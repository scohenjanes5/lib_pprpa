"""Consistent GPU GDF ppRPA gradient for three-dimensional Gamma cells.

This backend keeps one integral approximation throughout the calculation:

* the CPU PySCF reference uses :class:`pyscf.pbc.df.GDF`;
* :func:`lib_pprpa.pbc_gdf.make_pprpa_gdf` builds the ppRPA Davidson operator
  from that object's three-index factors; and
* every Coulomb/exchange derivative, including the ppRPA pair-density term and
  the CPHF response, uses GPU4PySCF's periodic GDF implementation.

It deliberately does not fall back to FFTDF or AFTDF.  Range-separated
functionals are rejected until their long-range GDF derivative is available.
The initial scope is real restricted Gamma-point HF and LDA/GGA/global-hybrid
KS references in three-dimensional cells.
"""

from __future__ import annotations

import copy

import cupy as cp
import numpy as np

from lib_pprpa.grad import pprpa_gamma as _cpu
from lib_pprpa.grad.gdf_jk_deriv import ejk_derivatives
from lib_pprpa.grad.grad_utils import get_xy_full
from lib_pprpa.grad.grad_utils_gpu_pbc import relaxed_xc_gradient as _xcg_gpu
from lib_pprpa.grad.hcore_contracted import hcore_deriv_contracted
from lib_pprpa.grad.pprpa import make_rdm1_relaxed_warm
from lib_pprpa.grad.pprpa_gamma_gpu import make_gpu_vresp
from lib_pprpa.gpu_gdf import TieredCderi
from lib_pprpa.gpu_mem import pooled
from lib_pprpa.pprpa_util import start_clock, stop_clock
from lib_pprpa.pbc_gdf import validate_gdf_context, validate_gdf_mf


def _xc_coefficients(mf, gpu_mf):
    """Return ``(omega, alpha, hybrid)`` and enforce the supported XC scope."""
    if not hasattr(mf, "xc"):
        return 0.0, 1.0, 1.0
    xc_type = gpu_mf._numint._xc_type(mf.xc)
    if xc_type not in ("LDA", "GGA"):
        raise NotImplementedError(
            "GDF ppRPA gradients currently support LDA and GGA functionals only")
    omega, alpha, hybrid = gpu_mf._numint.rsh_and_hybrid_coeff(mf.xc)
    if abs(omega) > 1e-12:
        raise NotImplementedError(
            "range-separated GDF ppRPA gradients are not implemented; "
            "using AFT for the long-range term would mix integral backends")
    return float(omega), float(alpha), float(hybrid)


def make_gpu_gdf_mf(mf, gpu_df=None, gpu_grids=None):
    """Mirror a converged CPU Gamma GDF reference on GPU without changing it.

    The already-built CPU auxiliary basis (after any exponent pruning) is
    copied exactly.  The GPU metric threshold and reciprocal mesh are then set
    to the CPU values.
    """
    cpu_df = validate_gdf_mf(mf)
    # Building Lpq normally initializes auxcell already.  Keep this helper safe
    # for direct callers too.
    if cpu_df.auxcell is None:
        cpu_df.build(j_only=False)

    from gpu4pyscf.pbc import dft as gdft, scf as gscf
    from gpu4pyscf.pbc.dft import BeckeGrids as GPUBeckeGrids
    from gpu4pyscf.pbc.df.df import GDF as GPU_GDF
    from pyscf.pbc.dft.gen_grid import BeckeGrids as CPUBeckeGrids

    cell = mf.cell
    kpts = np.asarray(mf.kpts).reshape(-1, 3)
    if hasattr(mf, "xc"):
        gpu_mf = gdft.KRKS(cell, kpts=kpts, xc=mf.xc)
        source_grids = getattr(mf, "grids", None)
        if gpu_grids is not None:
            # The caller supplies already-built GPU grids, so skip
            # constructing (and later rebuilding) a fresh Becke grid.
            gpu_mf.grids = gpu_grids
            gpu_mf.pprpa_xc_density_epsilon = getattr(
                mf, "pprpa_xc_density_epsilon", 1e-3)
        elif isinstance(source_grids, CPUBeckeGrids):
            gpu_mf.grids = GPUBeckeGrids(cell)
            gpu_mf.grids.level = source_grids.level
            gpu_mf.grids.atom_grid = copy.deepcopy(source_grids.atom_grid)
            gpu_mf.grids.radi_method = source_grids.radi_method
            gpu_mf.grids.prune = source_grids.prune
            # A Becke relaxed-density response uses a central directional
            # derivative in density space.  Allow callers to tighten it while
            # keeping a validated default.
            gpu_mf.pprpa_xc_density_epsilon = getattr(
                mf, "pprpa_xc_density_epsilon", 1e-3)
        # Periodic CPU and GPU RKS both otherwise use UniformGrids by default.
        # Preserve an explicitly selected mesh because it defines the XC
        # energy/response.
        elif getattr(source_grids, "mesh", None) is not None:
            gpu_mf.grids.mesh = np.asarray(mf.grids.mesh, dtype=int)
    else:
        gpu_mf = gscf.KRHF(cell, kpts=kpts)

    mesh = cpu_df.mesh if cpu_df.mesh is not None else cell.mesh
    if gpu_df is None:
        gpu_df = GPU_GDF(cell, kpts)
        # Mirror the basis that actually generated the authoritative CPU
        # CDERI, not merely a possibly mutated user-facing basis name.
        gpu_df.auxbasis = copy.deepcopy(cpu_df.auxcell.basis)
        gpu_df.exp_to_discard = None
        gpu_df.linear_dep_threshold = cpu_df.linear_dep_threshold
        gpu_df.mesh = np.asarray(mesh, dtype=int)
        gpu_df.is_gamma_point = True
        gpu_df.build(j_only=False)
    else:
        # A caller that already built the identical GPU GDF passes it in;
        # verify it is the same object this function would have built.
        if (not np.array_equal(np.asarray(gpu_df.mesh, dtype=int),
                               np.asarray(mesh, dtype=int))
                or gpu_df.linear_dep_threshold != cpu_df.linear_dep_threshold):
            raise ValueError("supplied gpu_df does not match the CPU GDF "
                             "(mesh or linear_dep_threshold differ)")

    gpu_mf.with_df = gpu_df
    gpu_mf.rsjk = None
    gpu_mf.exxdiv = mf.exxdiv
    gpu_mf.max_memory = mf.max_memory
    gpu_mf.verbose = mf.verbose
    gpu_mf.mo_coeff = cp.asarray(mf.mo_coeff)[None]
    gpu_mf.mo_energy = cp.asarray(mf.mo_energy)[None]
    gpu_mf.mo_occ = cp.asarray(mf.mo_occ)[None]
    gpu_mf.e_tot = mf.e_tot
    gpu_mf.converged = mf.converged
    return gpu_mf


class _GDFDerivativeEngine:
    """One reusable GPU4PySCF three-center derivative engine (Gamma point)."""

    def __init__(self, gpu_mf):
        from gpu4pyscf.pbc.df.int3c2e import SRInt3c2eOpt
        from gpu4pyscf.pbc.df.rsdf_builder import _guess_omega
        from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh

        self.gpu_mf = gpu_mf
        self.kpts = np.asarray(gpu_mf.kpts).reshape(-1, 3)
        if len(self.kpts) != 1 or abs(self.kpts).max() > 0:
            raise NotImplementedError("the GDF derivative engine is Gamma-point only")
        with_df = gpu_mf.with_df
        cell = gpu_mf.cell
        kmesh = kpts_to_kmesh(
            cell, self.kpts, rcut=cell.rcut + 10, bound_by_supmol=False)
        self.opt = SRInt3c2eOpt(
            cell, with_df.auxcell, _guess_omega(cell), kmesh).build()
        self.linear_dep_threshold = with_df.linear_dep_threshold

    def jk_sum(self, terms, release_cderi=False):
        """Derivative of ``sum weight*(j_factor*E_J - k_factor/2*E_K)`` per atom over
        ``terms`` of (dm, hermi, j_factor, k_factor, exxdiv, weight), dm (nao, nao) symmetric
        for hermi=1 or antisymmetric for hermi=2 (no Coulomb part), with one pass of every
        integral (``gdf_jk_deriv.ejk_derivatives``).  ``release_cderi`` frees the device rows
        of a tiered cderi once the densities' fit is read from it."""
        nao = self.gpu_mf.cell.nao
        for dm, hermi, j_factor, *_ in terms:
            if hermi not in (1, 2):
                raise ValueError("the periodic GDF derivative supports hermi=1 or 2")
            if hermi == 2 and abs(j_factor) > 1e-15:
                raise ValueError("an antisymmetric density has no Coulomb contribution")
            if dm.shape != (nao, nao):
                raise ValueError(f"GDF derivative density must be ({nao}, {nao})")
        # the GPU reference's GDF keeps its cderi: the densities' fit is read from it
        with_df = self.gpu_mf.with_df
        if getattr(with_df, '_cderi', None) is None:
            with_df = None
        release_cderi = (release_cderi and with_df is not None
                         and isinstance(with_df._cderi[0], TieredCderi))
        return np.asarray(ejk_derivatives(
            self.opt, terms, omega=0.0,
            linear_dep_threshold=self.linear_dep_threshold, with_df=with_df,
            release_cderi=release_cderi)[0])


def grad_elec(pprpa_grad, xy, mult, atmlst=None):
    """Evaluate the electronic GDF ppRPA gradient."""
    mf = pprpa_grad.mf
    pprpa = pprpa_grad.base
    validate_gdf_context(pprpa, mf)

    cell = mf.cell
    if atmlst is None:
        atmlst = range(cell.natm)
    if mult not in ("s", "t"):
        raise ValueError(f"invalid multiplicity {mult!r}")

    from gpu4pyscf.pbc.grad import krhf as krhf_g
    # vppnl_nuc_grad lives in gpu4pyscf.pbc.grad.pp.  Until gpu4pyscf
    # d99e556 it was also re-exported by pbc.grad.krhf; import it from its
    # own module so this works on released gpu4pyscf.
    from gpu4pyscf.pbc.grad.pp import vppnl_nuc_grad

    is_ks = hasattr(mf, "xc")
    nocc_all = cell.nelectron // 2
    nocc, nvir = pprpa.nocc, pprpa.nvir
    nfo = nocc_all - nocc
    mo = np.asarray(mf.mo_coeff)

    gpu_mf = make_gpu_gdf_mf(mf, gpu_df=getattr(pprpa_grad, "gpu_df", None),
                             gpu_grids=getattr(pprpa_grad, "gpu_grids", None))
    _, _, hybrid = _xc_coefficients(mf, gpu_mf)
    gpu_grad = gpu_mf.nuc_grad_method()
    gdf_deriv = _GDFDerivativeEngine(gpu_mf)

    kmf_cpu = _cpu.rhf_to_krhf(mf)
    kg_cpu = kmf_cpu.nuc_grad_method()
    vresp = make_gpu_vresp(cell, mf, gpu_mf=gpu_mf) if is_ks else None

    # Contract the active pair density X + Y directly with the same GPU GDF
    # factors.  Reconstructing a full periodic Lpq tensor here would scale as
    # naux*nao**2 even when the Davidson problem uses a compact active space.
    def pair_get_k(dms, hermi):
        _, vk = gpu_mf.with_df.get_jk(
            cp.asarray(dms), hermi=hermi, with_j=False, with_k=True,
            exxdiv=None)
        return cp.asnumpy(vk)

    p_mo, w_mo = make_rdm1_relaxed_warm(pprpa_grad, mf, xy, mult, vresp=vresp,
                                        pair_get_k=pair_get_k)
    # the response and the pair exchange no longer read the cderi
    vresp = pair_get_k = None

    w = mo @ w_mo @ mo.T - kg_cpu.make_rdm1e(
        kmf_cpu.mo_energy, kmf_cpu.mo_coeff, kmf_cpu.mo_occ)[0]
    p = mo @ p_mo @ mo.T
    pprpa_grad.rdm1e = p
    d = kmf_cpu.make_rdm1()[0]
    total = d + p

    occ_y, vir_x = get_xy_full(
        xy, pprpa.oo_dim, mult, nocc=nocc, nvir=nvir)
    cocc = mo[:, nfo:nfo + nocc]
    cvir = mo[:, nfo + nocc:nfo + nocc + nvir]
    pair = cvir @ vir_x @ cvir.T + cocc @ occ_y @ cocc.T

    dg, pg, tg, wg, xg = (cp.asarray(a) for a in (d, p, total, w, pair))
    natm = cell.natm
    de = np.zeros((natm, 3))

    # One-electron skeleton (kinetic + local pseudopotential).  The per-atom
    # hcore_generator closure re-evaluates every AO on the full uniform mesh
    # for each atom; form the trace against the density once on the mesh and
    # contract it in reciprocal space instead.
    start_clock('Calculate hcore derivative')
    de += hcore_deriv_contracted(cell, gpu_mf.kpts, tg)
    stop_clock('Calculate hcore derivative')

    # All reference-density Coulomb/exchange terms use the same GDF metric as
    # the ppRPA operator.  Q(D+P)-Q(P) isolates Q(D)+the D/P cross derivative.
    terms = [(tg, 1, 1.0, hybrid, mf.exxdiv, 1.0),
             (pg, 1, 1.0, hybrid, mf.exxdiv, -1.0)]

    # Pair-density exchange.  GPU4PySCF's GDF derivative is the derivative of
    # -(k_factor/4) Tr[D K(D)].  k_factor=2 therefore has the ppRPA magnitude;
    # antisymmetric/triplet and symmetric/singlet sectors enter with opposite
    # signs under the ppRPA metric convention.
    xa = (xg - xg.T) * 0.5
    xs = (xg + xg.T) * 0.5
    if float(cp.max(cp.abs(xa))) > 1e-10:
        terms.append((xa, 2, 0.0, 2.0, None, 1.0))
    if float(cp.max(cp.abs(xs))) > 1e-10:
        terms.append((xs, 1, 0.0, 2.0, None, -1.0))
    # one pass of the three-center derivative integrals for all of them
    start_clock('Calculate GDF Coulomb and exchange derivatives')
    de += gdf_deriv.jk_sum(terms, release_cderi=pprpa_grad.release_cderi_device)
    stop_clock('Calculate GDF Coulomb and exchange derivatives')

    # XC potential skeleton and its response to the relaxed density.  This
    # helper contains no Coulomb/exchange term; those are entirely GDF above.
    if is_ks:
        start_clock('Calculate XC derivative')
        de += _xcg_gpu(gpu_mf, mf.xc, pg, dm0=dg)
        stop_clock('Calculate XC derivative')

    # Pulay overlap and nonlocal pseudopotential terms.
    start_clock('Calculate overlap and nonlocal PP derivatives')
    s1 = gpu_grad.get_ovlp(cell, gpu_mf.kpts)
    de += krhf_g.contract_h1e_dm(cell, s1, wg[None], hermi=1)
    de += vppnl_nuc_grad(cell, total[None], kpts=gpu_mf.kpts)
    stop_clock('Calculate overlap and nonlocal PP derivatives')

    return de[list(atmlst)]


class Gradients(_cpu.Gradients):
    """Consistent periodic GDF ppRPA gradient driver."""

    def __init__(self, pprpa, mf, mult="t", state=0, gpu_df=None, gpu_grids=None):
        validate_gdf_context(pprpa, mf)
        self.mf = mf
        # Optional pre-built GPU GDF and grids, so a caller driving many
        # evaluations does not rebuild them every time.
        self.gpu_df = gpu_df
        self.gpu_grids = gpu_grids
        # A tiered gpu_df's device rows go to the GDF derivative pass once its last reader
        # (the densities' fit) is done: its Fourier blocks need tens of GB at NV215.  That
        # gpu_df then serves no other gradient; a caller reusing it sets this False.
        self.release_cderi_device = True
        self.base = pprpa
        self.mol = mf.cell
        self.cell = mf.cell
        self.state = state
        self.verbose = self.mol.verbose
        self.mult = mult
        self.rdm1e = None
        self.atmlst = None
        self.de = None

    def grad_elec(self, xy, mult, atmlst):
        # the grid passes (CPHF response, hcore, Vxc) allocate blocks over gpu4pyscf's
        # cudaMalloc threshold every iteration: pool them
        with pooled():
            return grad_elec(self, xy, mult, atmlst)


Grad = Gradients


__all__ = ["Grad", "Gradients", "grad_elec", "make_gpu_gdf_mf"]
