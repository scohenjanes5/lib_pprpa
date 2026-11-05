"""GPU (gpu4pyscf) accelerated pp-RPA Gamma-point nuclear gradient.

Drop-in GPU counterpart of ``lib_pprpa.grad.pprpa_gamma``.  The relaxed 1-RDM /
energy-weighted density is built by the CPU routine ``make_rdm1_relaxed_rhf_pprpa``
(small MO-space block algebra) but driven by a GPU response (``make_gpu_vresp``:
J inside the fxc pass, only the occupied columns for the CPHF iterations), so the
CPHF/Z-vector solve runs on the GPU; the gradient *assembly* (hcore,
overlap, J/K, Vxc, fxc, pseudo-potential) also runs on the GPU.

Term -> gpu4pyscf primitive map (D=KS ref dm, P=relaxed corr dm, T=D+P,
W=energy-weighted dm, X=pp-RPA 2-RDM amplitude density):
  hcore   : hcore_contracted.hcore_deriv_contracted(T)  (density on the mesh once)
  ovlp    : krhf_g.contract_h1e_dm(s1, W)            (note +=, W carries -dme0)
  J ref   : polarization Q(T)-Q(P) via jk_energy_per_atom(j=1,sr=lr=0), FFTDF
  hyb K   : Q(T)-Q(P) = -1/2 [F_b(D, T) + F_b(P, D)], gpu_exchange.exchange_force with
            the hybrid's hyb SR + alpha LR kernel, FFT    (hybrids)
  pairing : gpu_pairing_force.pairing_k_force_lowrank(C, M) for X = C M C^T, FFT
  Vxc     : 2*einsum(v1ao[1:,atom], T) from grad_utils_gpu_pbc._contract_xc_kernel
  fxc     : 1*einsum(f1vo[1:,atom], D)  (same routine; factor 1 = CPU's 0.5*..*2)
  PP nl   : pp.vppnl_nuc_grad(T)

Key conventions / gpu4pyscf quirks (see also grad_utils_gpu_pbc):
* J derivatives use FFTDF (``get_j_e1`` exists, fast).  FFTDF has no ``get_k_e1``, so
  the K derivatives are the low-rank FFT force: the derivative of the SCF's own FFTDF
  exchange energy, -1/4 tr(D K[D]) = -1/2 times the force of X = D.  The difference of the
  T and P terms runs through pairs that hold an occupied orbital only (nocc nmo, not nmo^2).
* pp-RPA pairing exchange: X = [C_v, C_o] diag(x, y) [C_v, C_o]^T has rank <= nocc + nvir,
  so its force is one FFT pass over the active-MO codensities plus a gradient-AO
  pass; exact for a general X, no symmetric/antisymmetric split.
* The relaxed density's exchange K[X + Y] @ orbp is weight * ao^T W^T for the same
  potential grids W, so one FFT pass (``pair_potential_lowrank``) serves both, and W
  is released before the CPHF solve; the CPHF solution is kept on the ``Gradients``
  object (``cphf_x0``) as the warm start of the next geometry, and with
  ``cphf_chkfile`` in a chkfile as the warm start of a later run.

Requirements / scope
--------------------
* Gamma point, RKS (or RHF) reference only; LDA/GGA and (range-separated) hybrid functionals.
* Validated: vs CPU 2e-10 (C2 diamond, up to gth-tzv2p, singlet+triplet);
  vs finite difference 2.9e-7 (C2, off-grid).  Runs on the 63-atom NV cell.
"""
import numpy as np
import cupy as cp
from scipy.linalg import block_diag

from lib_pprpa.grad.pprpa import make_rdm1_relaxed_warm
from lib_pprpa.grad import pprpa_gamma as _cpu
from lib_pprpa.grad.grad_utils import get_xy_full
from lib_pprpa.grad.grad_utils_gpu_pbc import _contract_xc_kernel as _cxk_gpu
from lib_pprpa.gpu_coulomb import exchange_coulG
from lib_pprpa.gpu_emulation import emulation_available, fp64_emulation
from lib_pprpa.gpu_exchange import exchange_force, factor
from lib_pprpa.gpu_mem import pooled
from lib_pprpa.gpu_pairing_force import (pair_exchange, pair_potential_lowrank,
                                         pairing_k_force_lowrank)
from lib_pprpa.gpu_response import fused_response
from lib_pprpa.grad.hcore_contracted import hcore_deriv_contracted


def _fused_applies(kmf):
    '''``gpu_response.fused_response`` covers this KRKS: KNumInt on its uniform grids, FFTDF J
    and K, LDA/GGA (hybrid or not).'''
    from gpu4pyscf.pbc.df.fft import FFTDF
    from gpu4pyscf.pbc.dft.numint import KNumInt
    ni = kmf._numint
    return (type(ni) is KNumInt and type(kmf.with_df) is FFTDF
            and kmf.j_engine is None and kmf.rsjk is None
            and ni._xc_type(kmf.xc) in ('LDA', 'GGA'))


def make_gpu_vresp(cell, mf, gpu_mf=None, emulate=True):
    """GPU CPHF response (``singlet=None, hermi=1``) for Gamma KS.

    LDA/GGA (and their hybrids) on the uniform-grid KNumInt with FFTDF run as
    ``gpu_response.fused_response`` (J inside the fxc pass, exact exchange from the MO
    factors of D, and with ``ket`` only V @ ket, the columns the CPHF iterations
    use); everything else (MGGA, GDF, RSJK, other integrators) runs gpu4pyscf's
    ``KRKS.gen_response``.  Both cache the XC kernel once.  ``vresp(dm, ket=None)``:
    NumPy in -> NumPy out, CuPy in -> CuPy out.  With
    ``emulate`` the response runs under cuBLAS's ``performant`` fp64 emulation where the
    device gains from it (``gpu_emulation``).
    """
    from gpu4pyscf.pbc import dft as gdft

    kpts = np.asarray(mf.kpts).reshape(-1, 3)
    assert len(kpts) == 1 and abs(kpts).max() < 1e-9, \
        "GPU pp-RPA response is Gamma-point only"
    if gpu_mf is None:
        kmf = gdft.KRKS(cell, kpts=kpts, xc=mf.xc)
    else:
        kmf = gpu_mf
        if not isinstance(kmf, gdft.KRKS):
            raise TypeError("gpu_mf must be a gpu4pyscf periodic KRKS object")
        if kmf.cell is not cell:
            raise ValueError("gpu_mf and mf must use the same Cell object")
        if len(np.asarray(kmf.kpts).reshape(-1, 3)) != 1 or not np.allclose(
                np.asarray(kmf.kpts).reshape(-1, 3)[0], 0.0, atol=1e-9):
            raise ValueError("gpu_mf must be a Gamma-point KRKS object")
        if kmf.xc != mf.xc:
            raise ValueError("gpu_mf and mf must use the same XC functional")
    native_method = getattr(type(kmf), "gen_response", None)
    native_cache = getattr(kmf._numint, "cache_xc_kernel1", None)
    native_fxc = getattr(kmf._numint, "nr_rks_fxc", None)
    if not all(callable(fn) for fn in
               (native_method, native_cache, native_fxc)):
        raise RuntimeError(
            "GPU pp-RPA gradients require the native periodic "
            "KRKS.gen_response/cache_xc_kernel1/nr_rks_fxc implementation "
            "introduced in GPU4PySCF commit 1cffe4a. Install that revision "
            "or later together with its matching compiled libraries; a "
            "Python-only backport is ABI-incompatible.")

    kmf.exxdiv = mf.exxdiv
    kmf.max_memory = mf.max_memory
    kmf.verbose = mf.verbose
    kmf.mo_coeff = cp.asarray(mf.mo_coeff)[None]
    kmf.mo_energy = cp.asarray(mf.mo_energy)[None]
    kmf.mo_occ = cp.asarray(mf.mo_occ)[None]
    if kmf.grids.coords is None:
        kmf.grids.build()

    nao = cell.nao
    strategy = 'performant' if emulate and emulation_available('performant') else None
    if _fused_applies(kmf):
        respond = fused_response(cell, kmf)
    else:
        native_vresp = kmf.gen_response(singlet=None, hermi=1)

        def respond(dmg, ket=None, cross=None):
            v = native_vresp(dmg.reshape(1, 1, nao, nao))[0, 0]
            return v if ket is None else v @ ket

    def vresp(dm, ket=None, cross=None):
        # Return CuPy when handed CuPy so a device-resident CPHF loop can avoid
        # round-tripping the (nao, nao) matrices through the host every
        # iteration.  NumPy in -> NumPy out, unchanged, for every other caller.
        want_device = isinstance(dm, cp.ndarray)
        dmg = cp.asarray(dm)
        if dmg.shape != (nao, nao):
            raise ValueError(
                f"Gamma CPHF response expects ({nao}, {nao}); got {dmg.shape}")
        with fp64_emulation(strategy):
            v = respond(dmg, None if ket is None else cp.asarray(ket),
                        None if cross is None else cp.asarray(cross))
        return v if want_device else cp.asnumpy(v)

    vresp.accepts_device_arrays = True
    vresp.accepts_ket = True
    vresp.accepts_cross = True
    return vresp


def grad_elec(pprpa_grad, xy, mult, atmlst=None):
    mf = pprpa_grad.mf
    pprpa = pprpa_grad.base
    cell = mf.mol
    is_ks = hasattr(mf, "xc")
    kpts = mf.kpts
    if atmlst is None:
        atmlst = range(cell.natm)
    assert mult in ("t", "s"), "invalid mult %r" % mult
    assert len(kpts) == 1 and abs(np.asarray(kpts)).max() < 1e-9, \
        "GPU pp-RPA gradient is Gamma-point only"

    from gpu4pyscf.pbc import dft as gdft, scf as gscf
    from gpu4pyscf.pbc.grad import krhf as krhf_g
    # vppnl_nuc_grad lives in gpu4pyscf.pbc.grad.pp.  Until gpu4pyscf
    # d99e556 it was also re-exported by pbc.grad.krhf; import it from its
    # own module so this works on released gpu4pyscf.
    from gpu4pyscf.pbc.grad.pp import vppnl_nuc_grad

    nocc_all = cell.nelectron // 2
    nocc, nvir = pprpa.nocc, pprpa.nvir
    nfo = nocc_all - nocc
    mo = mf.mo_coeff

    # --- relaxed density / energy-weighted density (CPU, validated) ----------
    kmf_cpu = _cpu.rhf_to_krhf(mf)
    kg_cpu = kmf_cpu.nuc_grad_method()
    if is_ks:
        vresp = make_gpu_vresp(cell, mf)   # GPU grid response for the CPHF solve
    else:
        vresp = None
    # X + Y = orbp diag(y, x) orbp^T has rank <= the active space: the potential grids
    # of its codensities, built once, give both its exchange K[X + Y] @ orbp for the
    # relaxed density and the pp-RPA pairing exchange force
    # 2 sum_{i in A} sum_l (d_x K[X])_il X_il (exact for a general X)
    orbp = mo[:, nfo:nfo + nocc + nvir]
    occ_y, vir_x = get_xy_full(
        xy, pprpa.oo_dim, mult, nocc=nocc, nvir=nvir)
    Mp = block_diag(occ_y, vir_x)
    Wp = pair_potential_lowrank(cell, cell.mesh, orbp, Mp, devices=pprpa_grad.devices)
    pair_k = pair_exchange(cell, cell.mesh, Wp)
    de_pairing = pairing_k_force_lowrank(cell, cell.mesh, orbp, Mp, W=Wp)
    Wp = None
    P_mo, W_mo = make_rdm1_relaxed_warm(pprpa_grad, mf, xy, mult, vresp=vresp, pair_k=pair_k)
    W = mo @ W_mo @ mo.T \
        - kg_cpu.make_rdm1e(kmf_cpu.mo_energy, kmf_cpu.mo_coeff, kmf_cpu.mo_occ)[0]
    P = mo @ P_mo @ mo.T
    pprpa_grad.rdm1e = P
    D = kmf_cpu.make_rdm1()[0]
    T = D + P

    # --- GPU assembly --------------------------------------------------------
    if is_ks:
        kmf = gdft.KRKS(cell, kpts=kpts, xc=mf.xc)
    else:
        kmf = gscf.KRHF(cell, kpts=kpts)
    kmf.exxdiv = mf.exxdiv
    kmf.mo_coeff = [cp.asarray(mo)]
    kmf.mo_energy = [cp.asarray(mf.mo_energy)]
    kmf.mo_occ = [cp.asarray(mf.mo_occ)]
    gg = kmf.nuc_grad_method()

    ni = kmf._numint if is_ks else None
    # exchange factors as gpu4pyscf.pbc.grad.krks: hyb short range, alpha long range
    if is_ks:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc)
    else:
        omega, alpha, hyb = 0.0, 1.0, 1.0

    Dg, Pg, Tg, Wg = (cp.asarray(a) for a in (D, P, T, W))
    aoslices = cell.aoslice_by_atom()
    natm = cell.natm
    de = cp.zeros((natm, 3))

    # hcore (kinetic + local PP) contracted with the total density: the density
    # goes on the mesh once instead of one (3, nao, nao) derivative per atom
    de += cp.asarray(hcore_deriv_contracted(cell, kpts, T))

    # J reference via the polarization identity Q(D+P)-Q(P), batched (FFTDF).
    eJ = krhf_g.jk_energy_per_atom(
        kmf, cp.stack([Tg, Pg])[:, None], kpts, j_factor=1.0,
        sr_factor=0.0, lr_factor=0.0, omega=0.0, exxdiv=None)
    de += cp.asarray(eJ[0] - eJ[1])

    if abs(hyb) > 1e-12 or abs(alpha) > 1e-12:
        # hybrid reference exchange via the same polarization identity, on the SCF's mesh:
        # Q = -1/2 F with F(M) = F_b(M, M) bilinear (W from the first, L from the second), so
        # F(T) - F(P) = F_b(D, T) + F_b(P, D): W of D on every MO, W of P on the occupied ones
        coulG = exchange_coulG(cell, cell.mesh, omega, alpha, hyb, mf.exxdiv)
        occ = mf.mo_occ > 0
        n, cocc = mf.mo_occ[occ], mo[:, occ]
        lam_p, U_p = factor(P_mo, mo)
        de_k = exchange_force(cell, cell.mesh, n, cocc, mo, mo @ (P_mo + np.diag(mf.mo_occ)),
                              coulG)
        de_k += exchange_force(cell, cell.mesh, lam_p, U_p, cocc, cocc * n, coulG)
        de -= 0.5 * cp.asarray(de_k)

    de += cp.asarray(de_pairing)

    # Vxc skeleton (contract with T) + fxc.P skeleton (contract with D)
    if is_ks:
        f1vo, v1ao = _cxk_gpu(kmf, mf.xc, Pg, dm0=Dg, with_vxc=True)
        for ia in range(natm):
            p0, p1 = aoslices[ia, 2:]
            de[ia] += cp.einsum('xij,ij->x', v1ao[1:, p0:p1], Tg[p0:p1]).real * 2
            de[ia] += cp.einsum('xij,ij->x', f1vo[1:, p0:p1], Dg[p0:p1]).real

    de = de.get()

    # overlap (energy-weighted) and nonlocal pseudo-potential
    s1 = gg.get_ovlp(cell, kpts)
    de += krhf_g.contract_h1e_dm(cell, s1, Wg[None], hermi=1)
    de += vppnl_nuc_grad(cell, T[None], kpts=kpts)

    return de[list(atmlst)] if not isinstance(atmlst, range) else de


class Gradients(_cpu.Gradients):
    """GPU pp-RPA Gamma-point gradient (see module docstring)."""

    def __init__(self, pprpa, mf, mult="t", state=0):
        super().__init__(pprpa, mf, mult=mult, state=state)
        # CUDA devices the pair-potential FFT pass (relaxed-density exchange and
        # pairing force) is dispatched over; None is the current device.
        self.devices = None

    def grad_elec(self, xy, mult, atmlst):
        # the grid passes (CPHF response, hcore, Vxc) and the FFT batches allocate blocks
        # over gpu4pyscf's cudaMalloc threshold every iteration: pool them
        with pooled():
            return grad_elec(self, xy, mult, atmlst)


Grad = Gradients
