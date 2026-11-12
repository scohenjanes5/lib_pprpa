import numpy as np
from scipy.linalg import block_diag

from pyscf import lib
from pyscf.lib import logger
from lib_pprpa.grad.pprpa import Gradients as pprpa_grad
from lib_pprpa.grad.pprpa import make_rdm1_relaxed_rhf_pprpa
from pyscf.pbc.gto.pseudo import pp_int
from lib_pprpa.grad.grad_utils import _contract_xc_kernel_krks, get_veff_krks, get_xy_full
from lib_pprpa.grad.fused_response import fused_applies, fused_response
from lib_pprpa.grad.hcore_deriv import hcore_deriv_contracted
from lib_pprpa.grad.pairing_force import pair_exchange_lowrank
from lib_pprpa.pprpa_davidson import ppRPA_Davidson


def grad_elec(pprpa_grad, xy, mult, atmlst=None):
    mf = pprpa_grad.mf
    pprpa = pprpa_grad.base
    cell = mf.mol
    kmf = rhf_to_krhf(mf)
    kmf_grad = kmf.nuc_grad_method()
    if atmlst is None:
        atmlst = range(cell.natm)
    assert mult in ['t', 's'], 'mult = {}. is not valid in grad_elec'.format(mult)

    nocc_all = mf.mol.nelectron // 2
    nocc = pprpa.nocc
    nvir = pprpa.nvir
    nfrozen_occ = nocc_all - nocc
    kpts = mf.kpts
    mo_coeff = mf.mo_coeff
    log = logger.Logger(kmf_grad.stdout, kmf_grad.verbose)

    if hasattr(mf, 'xc') and kmf_grad.grid_response:
        raise NotImplementedError('Grid response is not implemented in pprpa yet.')

    # LDA/GGA on the FFTDF mesh: J inside the fxc pass, the AO values kept across the
    # CPHF iterations and only the occupied columns computed for them (fused_response)
    if hasattr(mf, "xc") and fused_applies(mf):
        vresp = fused_response(mf)
    # Reuse one grid AO evaluation across all CPHF iterations: build an AO
    # cache for the reference and hand it to the stock response function.
    elif hasattr(mf, "xc"):
        ni_resp = mf._numint
        deriv_resp = 1 if ni_resp._xc_type(mf.xc) in ("GGA", "MGGA") else 0
        if hasattr(ni_resp, 'set_ao_cache'):
            # pyscf-warlocat: cache on the numint object
            ni_resp.set_ao_cache(cell, mf.grids, deriv_resp,
                                 kpts=None, max_memory=mf.max_memory)
            vresp = mf.gen_response(singlet=None, hermi=1)
        else:
            # FIX(audit): stock pyscf has no ao_cache support; plain response (no caching)
            vresp = mf.gen_response(singlet=None, hermi=1)
    else:
        vresp = None
    # X + Y = orbp diag(y, x) orbp^T has rank <= the active space: one FFT pass over its
    # codensities gives both its exchange K[X + Y] @ orbp for the relaxed density and the
    # pairing force
    occ_y_mat, vir_x_mat = get_xy_full(xy, pprpa.oo_dim, mult)
    orbp = mo_coeff[:, nfrozen_occ : nfrozen_occ + nocc + nvir]
    Mp = block_diag(occ_y_mat, vir_x_mat)
    pair_k, de_pairing = pair_exchange_lowrank(cell, orbp, Mp, mesh=mf.with_df.mesh,
                                           exxdiv=mf.exxdiv, mydf=mf.with_df)
    cphf_out = {}
    dm0, i_int = make_rdm1_relaxed_rhf_pprpa(
        pprpa, mf, xy=xy, mult=mult, cphf_max_cycle=pprpa_grad.cphf_max_cycle, cphf_conv_tol=pprpa_grad.cphf_conv_tol,
        vresp=vresp, pair_k=pair_k,
        cphf_x0=pprpa_grad.cphf_x0, cphf_x0_out=cphf_out,
    )
    pprpa_grad.cphf_x0 = cphf_out.get('cphf_x0_ao')
    pair_k = None
    i_int = mo_coeff @ i_int @ mo_coeff.T
    i_int -= kmf_grad.make_rdm1e(kmf.mo_energy, kmf.mo_coeff, kmf.mo_occ)[0]

    dm0 = mo_coeff @ dm0 @ mo_coeff.T
    pprpa_grad.rdm1e = dm0
    dm0_hf = kmf.make_rdm1()[0] # (nband,3,nao,nao)

    # kinetic + local PP derivative against the total density: the density goes on the mesh
    # once instead of one (3, nao, nao) derivative per atom
    T = dm0 + dm0_hf
    de_hcore = hcore_deriv_contracted(kmf_grad, cell, T)
    s1 = kmf_grad.get_ovlp(cell, kpts)[0]

    if not hasattr(mf, 'xc'):  # HF
        t0 = (logger.process_clock(), logger.perf_counter())
        log.debug('Computing Gradients of NR-HF Coulomb repulsion')
        vhf = kmf_grad.get_veff([np.array([dm0_hf]), np.array([dm0])]) # (3,nset,nband,nao,nao)
        vhf = vhf[:,:,0,:,:].transpose(1,0,2,3)
        log.timer('gradients of 2e part', *t0)

        aoslices = cell.aoslice_by_atom()
        de = np.zeros((len(atmlst), 3))
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            # vhf[0] on the bra (rows) and on the ket (columns) of atom ia
            de[k] += np.einsum('xij,ij->x', vhf[0, :, p0:p1], T[p0:p1])
            de[k] += np.einsum('xij,ji->x', vhf[0, :, p0:p1], T[:, p0:p1])
            # nabla was applied on bra in s1, *2 for the contributions of nabla|ket>
            de[k] += np.einsum('xij,ij->x', vhf[1, :, p0:p1], dm0_hf[p0:p1, :]) * 2

            de[k] += np.einsum('xij,ji->x', s1[:, p0:p1], i_int[:, p0:p1]) * 2
    else:  # KS
        # Precompute the uniform-grid AOs once and reuse them across the Vxc,
        # Coulomb, and fxc-kernel derivative passes. On pyscf-warlocat the cache
        # lives on the numint object; on legacy pyscf it is a standalone AOCache
        # passed as a kwarg to every downstream function.
        ni = kmf._numint
        ao_deriv = 2 if ni._xc_type(kmf.xc) in ('GGA', 'MGGA') else 1
        if hasattr(ni, 'set_ao_cache'):
            ni.set_ao_cache(cell, kmf.grids, ao_deriv, kpts=kmf.kpts,
                            max_memory=kmf_grad.max_memory)
        # J (and hybrid K) derivatives of D and P; the Vxc skeleton of D and the fxc skeleton
        # of P come from one grid pass
        _, vjk = get_veff_krks(kmf_grad, np.array([[dm0_hf], [dm0]]), with_vxc=False)
        vjk = vjk[:,:,0,:,:].transpose(1,0,2,3)
        f1vo, _, vxc0, _ = _contract_xc_kernel_krks(kmf, kmf.xc, dm0, with_vxc=True)
        vjk[1] += f1vo[1:]*0.5

        aoslices = cell.aoslice_by_atom()
        de = np.zeros((len(atmlst), 3))
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            # vxc[0] + vjk[0] on the bra (rows) and on the ket (columns) of atom ia
            v0 = vxc0[:, p0:p1] + vjk[0, :, p0:p1]
            de[k] += np.einsum('xij,ij->x', v0, T[p0:p1])
            de[k] += np.einsum('xij,ji->x', v0, T[:, p0:p1])
            # nabla was applied on bra in s1, *2 for the contributions of nabla|ket>
            de[k] += np.einsum('xij,ij->x', vjk[1, :, p0:p1], dm0_hf[p0:p1, :]) * 2

            de[k] += np.einsum('xij,ji->x', s1[:, p0:p1], i_int[:, p0:p1]) * 2

    de += de_hcore[list(atmlst)]
    de += de_pairing[list(atmlst)]
    de += pp_int.vppnl_nuc_grad(cell, np.array([T]), kpts)[list(atmlst)]

    return de


def rhf_to_krhf(myrhf):
    from pyscf.pbc import scf, dft
    if hasattr(myrhf, 'xc'):
        mykrhf = dft.KRKS(myrhf.mol, kpts = np.array([np.zeros(3)]))
        mykrhf.xc = myrhf.xc
    else:
        mykrhf = scf.KRHF(myrhf.mol, kpts = np.array([np.zeros(3)]))
    mykrhf.mo_coeff = [myrhf.mo_coeff]
    mykrhf.mo_energy = [myrhf.mo_energy]
    mykrhf.mo_occ = [myrhf.mo_occ]
    mykrhf.exxdiv = myrhf.exxdiv
    mykrhf.converged = myrhf.converged
    mykrhf.e_tot = myrhf.e_tot
    return mykrhf


class Gradients(pprpa_grad):
    def __init__(self, pprpa, mf, mult='t', state=0):
        from pyscf.pbc import scf
        self.mf = mf
        assert isinstance(mf, scf.rhf.SCF)
        assert len(mf.kpts) == 1 and np.allclose(mf.kpts[0], np.zeros(3)), "Only Gamma-point KSCF is supported in ppRPA gradients."
        assert pprpa._ao_direct or pprpa._use_eri, "PBC ppRPA gradients require either MO eri or AO direct approach."
        self.base = pprpa
        self.mol = mf.mol
        self.cell = mf.mol
        self.state = state
        self.verbose = self.mol.verbose
        self.mult = mult

        self.rdm1e = None
        self.atmlst = None
        self.de = None
        # AO-basis CPHF solution of the previous evaluation, the initial guess of the next
        # one (see make_rdm1_relaxed_rhf_pprpa)
        self.cphf_x0 = None

    def grad_nuc(self, cell=None, atmlst=None):
        if cell is None:
            cell = self.mol
        from pyscf.pbc.grad.krhf import grad_nuc
        return grad_nuc(cell, atmlst)

    def grad_elec(self, xy, mult, atmlst):
        return grad_elec(self, xy, mult, atmlst)
    
    def get_stress(self):
        raise NotImplementedError('ppRPA stress is not implemented yet.')

Grad = Gradients

ppRPA_Davidson.Gradients = lib.class_as_method(Gradients)
