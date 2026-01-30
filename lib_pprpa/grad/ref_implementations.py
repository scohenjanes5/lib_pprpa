"""
This module contains reference implementations of the gradient code before major optimizations.
Used only for testing purposes.
"""

import numpy as np
import scipy
from functools import reduce
from pyscf import lib, scf, dft
from pyscf.df.df_jk import _DFHF
from pyscf.lib import logger
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.grad import rks as rks_grad


def get_veff_df_rks_ref(ks_grad, mol=None, dm=None):
    if mol is None:
        mol = ks_grad.mol
    if dm is None:
        dm = ks_grad.base.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    grids, nlcgrids = rks_grad._initialize_grids(ks_grad)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory * 0.9 - mem_now)
    if ks_grad.grid_response:
        exc = []
        vxc = []
        for dmi in dm:
            exci, vxci = rks_grad.get_vxc_full_response(
                ni,
                mol,
                grids,
                mf.xc,
                dmi,
                max_memory=max_memory,
                verbose=ks_grad.verbose,
            )
            exc.append(exci)
            vxc.append(vxci)
        exc = np.asarray(exc)
        vxc = np.asarray(vxc)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc_full_response(
                ni,
                mol,
                nlcgrids,
                xc,
                dm,
                max_memory=max_memory,
                verbose=ks_grad.verbose,
            )
            exc += enlc
            vxc += vnlc
    else:
        exc, vxc = rks_grad.get_vxc(
            ni, mol, grids, mf.xc, dm, max_memory=max_memory, verbose=ks_grad.verbose
        )
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc(
                ni,
                mol,
                nlcgrids,
                xc,
                dm,
                max_memory=max_memory,
                verbose=ks_grad.verbose,
            )
            vxc += vnlc
    t0 = logger.timer(ks_grad, "vxc", *t0)

    vjk = np.zeros_like(vxc)
    if not ni.libxc.is_hybrid_xc(mf.xc):
        vj = ks_grad.get_j(mol, dm)
        vjk += vj
        if ks_grad.auxbasis_response:
            e1_aux = vj.aux
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        vj, vk = ks_grad.get_jk(mol, dm)
        if ks_grad.auxbasis_response:
            vk.aux *= hyb
        vk[:] *= hyb
        if omega != 0:
            vk_lr = ks_grad.get_k(mol, dm, omega=omega)
            vk[:] += vk_lr * (alpha - hyb)
            if ks_grad.auxbasis_response:
                vk.aux[:] += vk_lr.aux * (alpha - hyb)
        vjk += vj - vk * 0.5
        if ks_grad.auxbasis_response:
            e1_aux = vj.aux - vk.aux * 0.5

    if ks_grad.auxbasis_response:
        vjk = lib.tag_array(vjk, aux=e1_aux)
    if ks_grad.grid_response:
        vxc = lib.tag_array(vxc, exc1_grid=exc)
    return vxc, vjk


def get_veff_rks_ref(ks_grad, mol=None, dm=None):
    if mol is None:
        mol = ks_grad.mol
    if dm is None:
        dm = ks_grad.base.make_rdm1()
    mf = ks_grad.base
    ni = mf._numint
    grids, nlcgrids = rks_grad._initialize_grids(ks_grad)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory * 0.9 - mem_now)
    if ks_grad.grid_response:
        exc, vxc = rks_grad.get_vxc_full_response(
            ni, mol, grids, mf.xc, dm, max_memory=max_memory, verbose=ks_grad.verbose
        )
        if mf.do_nlc():
            xc = mf.xc if ni.libxc.is_nlc(mf.xc) else mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc_full_response(
                ni,
                mol,
                nlcgrids,
                xc,
                dm,
                max_memory=max_memory,
                verbose=ks_grad.verbose,
            )
            exc += enlc
            vxc += vnlc
    else:
        exc, vxc = rks_grad.get_vxc(
            ni, mol, grids, mf.xc, dm, max_memory=max_memory, verbose=ks_grad.verbose
        )
        if mf.do_nlc():
            xc = mf.xc if ni.libxc.is_nlc(mf.xc) else mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc(
                ni,
                mol,
                nlcgrids,
                xc,
                dm,
                max_memory=max_memory,
                verbose=ks_grad.verbose,
            )
            vxc += vnlc

    vjk = np.zeros_like(vxc)
    if not ni.libxc.is_hybrid_xc(mf.xc):
        vjk += ks_grad.get_j(mol, dm)
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        vj, vk = ks_grad.get_jk(mol, dm)
        vk *= hyb
        if omega != 0:
            vk += ks_grad.get_k(mol, dm, omega=omega) * (alpha - hyb)
        vjk += vj - vk * 0.5
    vxc = lib.tag_array(vxc, exc1_grid=exc)
    return vxc, vjk


def _contract_xc_kernel_ref(
    mf,
    xc_code,
    dmvo,
    dmoo=None,
    with_vxc=True,
    with_kxc=True,
    singlet=True,
    max_memory=2000,
):
    from pyscf.grad.tdrks import _lda_eval_mat_, _gga_eval_mat_, _mgga_eval_mat_

    mol, grids, ni = mf.mol, mf.grids, mf._numint
    xctype = ni._xc_type(xc_code)
    mo_coeff, mo_occ = mf.mo_coeff, mf.mo_occ
    nao, nmo = mo_coeff.shape
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    dmvo = (dmvo + dmvo.T) * 0.5
    f1vo = np.zeros((4, nao, nao))
    deriv = 2
    f1oo = np.zeros((4, nao, nao)) if dmoo is not None else None
    v1ao = np.zeros((4, nao, nao)) if with_vxc else None
    k1ao = np.zeros((4, nao, nao)) if with_kxc else None
    if with_kxc:
        deriv = 3
    if xctype == "HF":
        return f1vo, f1oo, v1ao, k1ao
    elif xctype == "LDA":
        fmat_, ao_deriv = _lda_eval_mat_, 1
    elif xctype == "GGA":
        fmat_, ao_deriv = _gga_eval_mat_, 2
    elif xctype == "MGGA":
        fmat_, ao_deriv = _mgga_eval_mat_, 2

    if singlet:
        for ao, mask, weight, coords in ni.block_loop(
            mol, grids, nao, ao_deriv, max_memory
        ):
            ao0 = ao[0] if xctype == "LDA" else ao
            rho = ni.eval_rho2(
                mol, ao0, mo_coeff, mo_occ, mask, xctype, with_lapl=False
            )
            vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]
            rho1 = (
                ni.eval_rho(mol, ao0, dmvo, mask, xctype, hermi=1, with_lapl=False) * 2
            )
            if xctype == "LDA":
                rho1 = rho1[np.newaxis]
            wv = np.einsum("yg,xyg,g->xg", rho1, fxc, weight)
            fmat_(mol, f1vo, ao, wv, mask, shls_slice, ao_loc)
            if dmoo is not None:
                rho2 = (
                    ni.eval_rho(mol, ao0, dmoo, mask, xctype, hermi=1, with_lapl=False)
                    * 2
                )
                if xctype == "LDA":
                    rho2 = rho2[np.newaxis]
                wv = np.einsum("yg,xyg,g->xg", rho2, fxc, weight)
                fmat_(mol, f1oo, ao, wv, mask, shls_slice, ao_loc)
            if with_vxc:
                fmat_(mol, v1ao, ao, vxc * weight, mask, shls_slice, ao_loc)
            if with_kxc:
                wv = np.einsum("yg,zg,xyzg,g->xg", rho1, rho1, kxc, weight)
                fmat_(mol, k1ao, ao, wv, mask, shls_slice, ao_loc)
    else:
        for ao, mask, weight, coords in ni.block_loop(
            mol, grids, nao, ao_deriv, max_memory
        ):
            ao0 = ao[0] if xctype == "LDA" else ao
            rho = ni.eval_rho2(
                mol, ao0, mo_coeff, mo_occ, mask, xctype, with_lapl=False
            )
            rho = np.repeat((rho * 0.5)[np.newaxis], 2, axis=0)
            vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]
            fxc_t = fxc[:, :, 0] - fxc[:, :, 1]
            fxc_t = fxc_t[0] - fxc_t[1]
            rho1 = ni.eval_rho(mol, ao0, dmvo, mask, xctype, hermi=1, with_lapl=False)
            if xctype == "LDA":
                rho1 = rho1[np.newaxis]
            wv = np.einsum("yg,xyg,g->xg", rho1, fxc_t, weight)
            fmat_(mol, f1vo, ao, wv, mask, shls_slice, ao_loc)
            if dmoo is not None:
                fxc_s = fxc[0, :, 0] + fxc[0, :, 1]
                rho2 = ni.eval_rho(
                    mol, ao0, dmoo, mask, xctype, hermi=1, with_lapl=False
                )
                if xctype == "LDA":
                    rho2 = rho2[np.newaxis]
                wv = np.einsum("yg,xyg,g->xg", rho2, fxc_s, weight)
                fmat_(mol, f1oo, ao, wv, mask, shls_slice, ao_loc)
            if with_vxc:
                fmat_(mol, v1ao, ao, vxc[0] * weight, mask, shls_slice, ao_loc)
            if with_kxc:
                kxc_t = kxc[0, :, 0] - kxc[0, :, 1]
                kxc_t = kxc_t[:, :, 0] - kxc_t[:, :, 1]
                wv = np.einsum("yg,zg,xyzg,g->xg", rho1, rho1, kxc_t, weight)
                fmat_(mol, k1ao, ao, wv, mask, shls_slice, ao_loc)
    for f in [f1vo, f1oo, v1ao, k1ao]:
        if f is not None:
            f[1:] *= -1
    return f1vo, f1oo, v1ao, k1ao


def make_rdm1_relaxed_rhf_pprpa_ref(
    pprpa, mf, xy=None, mult="t", istate=0, cphf_max_cycle=20, cphf_conv_tol=1.0e-8
):
    from lib_pprpa import pyscf_util
    from lib_pprpa.grad.grad_utils import (
        choose_slice,
        choose_range,
        contraction_2rdm_Lpq,
        contraction_2rdm_eri,
        get_xy_full,
        make_rdm1_unrelaxed_from_xy_full,
    )

    if xy is None:
        xy = pprpa.xy_s[istate] if mult == "s" else pprpa.xy_t[istate]
    nocc_all, nmo_all = mf.mol.nelectron // 2, mf.mol.nao
    nocc, nvir = pprpa.nocc, pprpa.nvir
    nfrozen_occ, nfrozen_vir = nocc_all - nocc, nmo_all - nocc_all - nvir
    oo_dim = (nocc + 1) * nocc // 2 if mult == "s" else (nocc - 1) * nocc // 2
    slice_p = choose_slice("p", nfrozen_occ, nocc, nvir, nfrozen_vir)
    slice_i = choose_slice("i", nfrozen_occ, nocc, nvir, nfrozen_vir)
    slice_a = choose_slice("a", nfrozen_occ, nocc, nvir, nfrozen_vir)
    slice_ip = choose_slice("ip", nfrozen_occ, nocc, nvir, nfrozen_vir)
    slice_ap = choose_slice("ap", nfrozen_occ, nocc, nvir, nfrozen_vir)
    slice_I = choose_slice("I", nfrozen_occ, nocc, nvir, nfrozen_vir)
    slice_A = choose_slice("A", nfrozen_occ, nocc, nvir, nfrozen_vir)
    orbA, orbI, orbp, orbi, orba = (
        mf.mo_coeff[:, slice_A],
        mf.mo_coeff[:, slice_I],
        mf.mo_coeff[:, slice_p],
        mf.mo_coeff[:, slice_i],
        mf.mo_coeff[:, slice_a],
    )
    occ_y_mat, vir_x_mat = get_xy_full(xy, oo_dim, mult)
    if pprpa._use_eri or pprpa._ao_direct:
        hermi = 1 if mult == "s" else 2
        X_ao = orba @ vir_x_mat @ orba.T
        X_eri = mf.mo_coeff.T @ mf.get_k(dm=X_ao, hermi=hermi) @ orbp
        Y_ao = orbi @ occ_y_mat @ orbi.T
        Y_eri = mf.mo_coeff.T @ mf.get_k(dm=Y_ao, hermi=hermi) @ orbp
        mo_ene_full = mf.mo_energy
    else:
        if nfrozen_occ > 0 or nfrozen_vir > 0:
            _, mo_ene_full, Lpq_full = pyscf_util.get_pyscf_input_mol(mf)
        else:
            mo_ene_full, Lpq_full = pprpa.mo_energy, pprpa.Lpq
    vresp = mf.gen_response(singlet=None, hermi=1)
    den_u = make_rdm1_unrelaxed_from_xy_full(occ_y_mat, vir_x_mat)
    den_u_ao = orbp @ np.diag(den_u) @ orbp.T
    veff_den_u = mf.mo_coeff.T @ vresp(den_u_ao) * 2 @ mf.mo_coeff
    i_prime = np.zeros((len(mo_ene_full), len(mo_ene_full)), dtype=occ_y_mat.dtype)
    if not pprpa._use_eri and not pprpa._ao_direct:
        i_prime[slice_p, slice_p] += contraction_2rdm_Lpq(
            occ_y_mat,
            vir_x_mat,
            Lpq_full,
            nocc,
            nvir,
            nfrozen_occ,
            nfrozen_vir,
            "p",
            "p",
        )
    else:
        i_prime[slice_p, slice_p] += contraction_2rdm_eri(
            occ_y_mat,
            vir_x_mat,
            X_eri,
            Y_eri,
            nocc,
            nvir,
            nfrozen_occ,
            nfrozen_vir,
            "p",
            "p",
        )
    i_prime[slice_a, slice_i] += veff_den_u[slice_a, slice_i]
    for p in choose_range("p", nfrozen_occ, nocc, nvir, nfrozen_vir):
        i_prime[p, p] += mo_ene_full[p] * den_u[p - nfrozen_occ]
    if nfrozen_vir > 0:
        if not pprpa._use_eri and not pprpa._ao_direct:
            i_prime[slice_ap, slice_p] += contraction_2rdm_Lpq(
                occ_y_mat,
                vir_x_mat,
                Lpq_full,
                nocc,
                nvir,
                nfrozen_occ,
                nfrozen_vir,
                "ap",
                "p",
            )
        else:
            i_prime[slice_ap, slice_p] += contraction_2rdm_eri(
                occ_y_mat,
                vir_x_mat,
                X_eri,
                Y_eri,
                nocc,
                nvir,
                nfrozen_occ,
                nfrozen_vir,
                "ap",
                "p",
            )
        i_prime[slice_ap, slice_i] += veff_den_u[slice_ap, slice_i]
    if nfrozen_occ > 0:
        if not pprpa._use_eri and not pprpa._ao_direct:
            i_prime[slice_ip, slice_p] += contraction_2rdm_Lpq(
                occ_y_mat,
                vir_x_mat,
                Lpq_full,
                nocc,
                nvir,
                nfrozen_occ,
                nfrozen_vir,
                "ip",
                "p",
            )
        else:
            i_prime[slice_ip, slice_p] += contraction_2rdm_eri(
                occ_y_mat,
                vir_x_mat,
                X_eri,
                Y_eri,
                nocc,
                nvir,
                nfrozen_occ,
                nfrozen_vir,
                "ip",
                "p",
            )
        i_prime[slice_A, slice_ip] += veff_den_u[slice_A, slice_ip]
    i_prime_prime = np.zeros_like(i_prime)
    i_prime_prime[slice_a, slice_I] = (
        i_prime[slice_a, slice_I] - i_prime[slice_I, slice_a].T
    )
    (
        i_prime_prime[slice_A, slice_a],
        i_prime_prime[slice_I, slice_i],
        i_prime_prime[slice_ap, slice_I],
    ) = (
        i_prime[slice_A, slice_a],
        i_prime[slice_I, slice_i],
        i_prime[slice_ap, slice_I],
    )
    d_prime = np.zeros_like(i_prime_prime)
    for i in choose_range("I", nfrozen_occ, nocc, nvir, nfrozen_vir):
        for j in choose_range("i", nfrozen_occ, nocc, nvir, nfrozen_vir):
            denorm = mo_ene_full[j] - mo_ene_full[i]
            if abs(denorm) >= 1e-6:
                d_prime[i, j] = i_prime_prime[i, j] / denorm
    for a in choose_range("A", nfrozen_occ, nocc, nvir, nfrozen_vir):
        for b in choose_range("a", nfrozen_occ, nocc, nvir, nfrozen_vir):
            denorm = mo_ene_full[b] - mo_ene_full[a]
            if abs(denorm) >= 1e-6:
                d_prime[a, b] = i_prime_prime[a, b] / denorm
    x_int = i_prime_prime[slice_A, slice_I].copy()
    d_ao = (
        orbI @ d_prime[slice_I, slice_i] @ orbi.T
        + orbA @ d_prime[slice_A, slice_a] @ orba.T
    )
    d_ao += d_ao.T
    x_int += orbA.T @ vresp(d_ao) * 2 @ orbI

    def fvind(x):
        dm = orbA @ x.reshape(nvir + nfrozen_vir, nocc + nfrozen_occ) * 2 @ orbI.T
        return (orbA.T @ vresp(dm + dm.T) @ orbI).ravel()

    from pyscf.scf import cphf

    d_prime[slice_A, slice_I] = cphf.solve(
        fvind,
        mo_ene_full,
        mf.mo_occ,
        x_int,
        max_cycle=cphf_max_cycle,
        tol=cphf_conv_tol,
    )[0].reshape(nvir + nfrozen_vir, nocc + nfrozen_occ)
    i_int = -np.einsum("qp,p->qp", d_prime, mo_ene_full)
    dp_ao = mf.mo_coeff @ d_prime @ mf.mo_coeff.T
    i_int[slice_I, slice_I] -= (
        0.5 * veff_den_u[slice_I, slice_I] + orbI.T @ vresp(dp_ao + dp_ao.T) @ orbI
    )
    i_int[slice_I, slice_a] -= i_prime[slice_I, slice_a]
    for i in choose_range("p", nfrozen_occ, nocc, nvir, nfrozen_vir):
        for j in choose_range("p", nfrozen_occ, nocc, nvir, nfrozen_vir):
            if abs(mo_ene_full[j] - mo_ene_full[i]) < 1e-6:
                i_int[i, j] -= 0.5 * i_prime[i, j]
    den_relaxed = d_prime
    for p in choose_range("p", nfrozen_occ, nocc, nvir, nfrozen_vir):
        den_relaxed[p, p] += 0.5 * den_u[p - nfrozen_occ]
    return den_relaxed + den_relaxed.T, i_int + i_int.T


def grad_elec_ref(pprpa_grad, xy, mult, atmlst=None):
    mf, pprpa = pprpa_grad.mf, pprpa_grad.base
    mol, mf_grad = mf.mol, mf.nuc_grad_method()
    if atmlst is None:
        atmlst = range(mol.natm)
    nocc_all, nocc, nvir = mf.mol.nelectron // 2, pprpa.nocc, pprpa.nvir
    nfrozen_occ = nocc_all - nocc
    hcore_deriv, s1 = mf_grad.hcore_generator(mol), mf_grad.get_ovlp(mol)
    dm0, i_int = make_rdm1_relaxed_rhf_pprpa_ref(
        pprpa,
        mf,
        xy=xy,
        mult=mult,
        cphf_max_cycle=pprpa_grad.cphf_max_cycle,
        cphf_conv_tol=pprpa_grad.cphf_conv_tol,
    )
    dm0 = mf.mo_coeff @ dm0 @ mf.mo_coeff.T
    pprpa_grad.rdm1e = dm0
    dm0_hf = mf.make_rdm1()
    i_int = mf.mo_coeff @ i_int @ mf.mo_coeff.T - mf_grad.make_rdm1e(
        mf.mo_energy, mf.mo_coeff, mf.mo_occ
    )
    from lib_pprpa.grad.grad_utils import get_xy_full

    occ_y_mat, vir_x_mat = get_xy_full(xy, pprpa.oo_dim, mult)
    coeff_occ, coeff_vir = (
        mf.mo_coeff[:, nfrozen_occ : nfrozen_occ + nocc],
        mf.mo_coeff[:, nfrozen_occ + nocc : nfrozen_occ + nocc + nvir],
    )
    xy_ao = coeff_vir @ vir_x_mat @ coeff_vir.T + coeff_occ @ occ_y_mat @ coeff_occ.T
    aux_response = mf_grad.auxbasis_response if isinstance(mf, _DFHF) else False
    if not hasattr(mf, "xc"):
        vj, vk = mf_grad.get_jk(mol, (dm0_hf, dm0, xy_ao), hermi=0)
        vhf = np.zeros_like(vj)
        vhf[:2] = vj[:2] - 0.5 * vk[:2]
        vhf[2] = vk[2]
        if aux_response:
            vhf_aux = np.zeros_like(vj.aux)
            vhf_aux[:2, :2] = vj.aux[:2, :2] - 0.5 * vk.aux[:2, :2]
            vhf_aux[2, 2] = vk.aux[2, 2] if mult == "s" else -vk.aux[2, 2]
            vhf = lib.tag_array(vhf, aux=vhf_aux)
        aoslices, de = mol.aoslice_by_atom(), np.zeros((len(atmlst), 3))
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            h1ao = hcore_deriv(ia)
            h1ao[:, p0:p1] += vhf[0, :, p0:p1]
            h1ao[:, :, p0:p1] += vhf[0, :, p0:p1].transpose(0, 2, 1)
            de[k] += (
                np.einsum("xij,ij->x", h1ao, dm0 + dm0_hf)
                + np.einsum("xij,ij->x", vhf[1, :, p0:p1], dm0_hf[p0:p1, :]) * 2
                + np.einsum("xij,ij->x", vhf[2, :, p0:p1], xy_ao[p0:p1, :]) * 2
                + np.einsum("xij,ji->x", s1[:, p0:p1], i_int[:, p0:p1]) * 2
            )
            if aux_response:
                de[k] += (
                    vhf.aux[0, 1, ia]
                    + 0.5 * vhf.aux[0, 0, ia]
                    + vhf.aux[1, 0, ia]
                    + 0.5 * vhf.aux[0, 0, ia]
                    + vhf.aux[2, 2, ia]
                )
    else:
        vj, vk = mf_grad.get_jk(mol, xy_ao, hermi=0)
        vhf = vk
        if aux_response:
            vxc, vjk = get_veff_df_rks_ref(mf_grad, mol, (dm0_hf, dm0))
            vhf = lib.tag_array(vhf, aux=vk.aux[0, 0] if mult == "s" else -vk.aux[0, 0])
        else:
            vxc, vjk = get_veff_rks_ref(mf_grad, mol, (dm0_hf, dm0))
        vjk[1] += (
            _contract_xc_kernel_ref(mf, mf.xc, dm0, None, False, False, True)[0][1:]
            * 0.5
        )
        aoslices, de = mol.aoslice_by_atom(), np.zeros((len(atmlst), 3))
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            h1ao = hcore_deriv(ia)
            h1ao[:, p0:p1] += vxc[0, :, p0:p1] + vjk[0, :, p0:p1]
            h1ao[:, :, p0:p1] += vxc[0, :, p0:p1].transpose(0, 2, 1) + vjk[
                0, :, p0:p1
            ].transpose(0, 2, 1)
            de[k] += (
                np.einsum("xij,ij->x", h1ao, dm0 + dm0_hf)
                + np.einsum("xij,ij->x", vjk[1, :, p0:p1], dm0_hf[p0:p1, :]) * 2
                + np.einsum("xij,ij->x", vhf[:, p0:p1], xy_ao[p0:p1, :]) * 2
                + np.einsum("xij,ji->x", s1[:, p0:p1], i_int[:, p0:p1]) * 2
            )
            if aux_response:
                de[k] += (
                    vjk.aux[0, 1, ia]
                    + 0.5 * vjk.aux[0, 0, ia]
                    + vjk.aux[1, 0, ia]
                    + 0.5 * vjk.aux[0, 0, ia]
                    + vhf.aux[ia]
                )
            if mf_grad.grid_response:
                de[k] += vxc.exc1_grid[0, ia]
    return de


def grad_elec_gamma_ref(pprpa_grad, xy, mult, atmlst=None):
    from lib_pprpa.grad.pprpa_gamma import rhf_to_krhf

    mf, pprpa, cell = pprpa_grad.mf, pprpa_grad.base, pprpa_grad.mf.mol
    kmf = rhf_to_krhf(mf)
    kmf_grad = kmf.nuc_grad_method()
    if atmlst is None:
        atmlst = range(cell.natm)
    nocc_all, nocc, nvir = cell.nelectron // 2, pprpa.nocc, pprpa.nvir
    nfrozen_occ, kpts, mo_coeff = nocc_all - nocc, mf.kpts, mf.mo_coeff
    dm0, i_int = make_rdm1_relaxed_rhf_pprpa_ref(
        pprpa,
        mf,
        xy=xy,
        mult=mult,
        cphf_max_cycle=pprpa_grad.cphf_max_cycle,
        cphf_conv_tol=pprpa_grad.cphf_conv_tol,
    )
    i_int = (
        mo_coeff @ i_int @ mo_coeff.T
        - kmf_grad.make_rdm1e(kmf.mo_energy, kmf.mo_coeff, kmf.mo_occ)[0]
    )
    dm0 = mo_coeff @ dm0 @ mo_coeff.T
    pprpa_grad.rdm1e = dm0
    dm0_hf = kmf.make_rdm1()[0]
    from lib_pprpa.grad.grad_utils import (
        get_xy_full,
        _contract_xc_kernel_krks,
        get_veff_krks,
    )

    occ_y_mat, vir_x_mat = get_xy_full(xy, pprpa.oo_dim, mult)
    xy_ao = (
        mo_coeff[:, nfrozen_occ : nfrozen_occ + nocc]
        @ occ_y_mat
        @ mo_coeff[:, nfrozen_occ : nfrozen_occ + nocc].T
        + mo_coeff[:, nfrozen_occ + nocc : nfrozen_occ + nocc + nvir]
        @ vir_x_mat
        @ mo_coeff[:, nfrozen_occ + nocc : nfrozen_occ + nocc + nvir].T
    )
    hcore_deriv, s1 = (
        kmf_grad.hcore_generator(cell, kpts),
        kmf_grad.get_ovlp(cell, kpts)[0],
    )
    if not hasattr(mf, "xc"):
        vhf = kmf_grad.get_veff([np.array([dm0_hf]), np.array([dm0])])
        vhf = vhf[:, :, 0, :, :].transpose(1, 0, 2, 3)
        vk = kmf_grad.get_k(np.array([xy_ao]))[:, 0, :, :]
        aoslices, de = cell.aoslice_by_atom(), np.zeros((len(atmlst), 3))
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            h1ao = hcore_deriv(ia)[:, 0]
            h1ao[:, p0:p1] += vhf[0, :, p0:p1]
            h1ao[:, :, p0:p1] += vhf[0, :, p0:p1].transpose(0, 2, 1)
            de[k] += (
                np.einsum("xij,ij->x", h1ao, dm0 + dm0_hf)
                + np.einsum("xij,ij->x", vhf[1, :, p0:p1], dm0_hf[p0:p1, :]) * 2
                + np.einsum("xij,ij->x", vk[:, p0:p1], xy_ao[p0:p1, :]) * 2
                + np.einsum("xij,ji->x", s1[:, p0:p1], i_int[:, p0:p1]) * 2
            )
    else:
        vk = kmf_grad.get_k(np.array([xy_ao]))[:, 0, :, :]
        vxc, vjk = get_veff_krks(kmf_grad, np.array([[dm0_hf], [dm0]]))
        vxc, vjk = vxc[:, :, 0, :, :].transpose(1, 0, 2, 3), vjk[
            :, :, 0, :, :
        ].transpose(1, 0, 2, 3)
        vjk[1] += _contract_xc_kernel_krks(kmf, kmf.xc, dm0)[0][1:] * 0.5
        aoslices, de = cell.aoslice_by_atom(), np.zeros((len(atmlst), 3))
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            h1ao = hcore_deriv(ia)[:, 0]
            h1ao[:, p0:p1] += vxc[0, :, p0:p1] + vjk[0, :, p0:p1]
            h1ao[:, :, p0:p1] += vxc[0, :, p0:p1].transpose(0, 2, 1) + vjk[
                0, :, p0:p1
            ].transpose(0, 2, 1)
            de[k] += (
                np.einsum("xij,ij->x", h1ao, dm0 + dm0_hf)
                + np.einsum("xij,ij->x", vjk[1, :, p0:p1], dm0_hf[p0:p1, :]) * 2
                + np.einsum("xij,ij->x", vk[:, p0:p1], xy_ao[p0:p1, :]) * 2
                + np.einsum("xij,ji->x", s1[:, p0:p1], i_int[:, p0:p1]) * 2
            )
    de += pp_int.vppnl_nuc_grad(cell, np.array([dm0 + dm0_hf]), kpts)
    return de


def make_rdm1_relaxed_ghf_pprpa_ref(
    pprpa, mf, xy=None, istate=0, cphf_max_cycle=20, cphf_conv_tol=1.0e-8
):
    from lib_pprpa import pyscf_util
    from lib_pprpa.grad.grad_utils import (
        choose_slice,
        choose_range,
        contraction_2rdm_Lpq,
        get_xy_full,
        make_rdm1_unrelaxed_from_xy_full,
    )

    if xy is None:
        xy = pprpa.xy[istate]
    nocc_all, nmo_all = mf.mol.nelectron, mf.mol.nao * 2
    nocc, nvir = pprpa.nocc, pprpa.nvir
    nfrozen_occ, nfrozen_vir = nocc_all - nocc, nmo_all - nocc_all - nvir
    if nfrozen_occ > 0 or nfrozen_vir > 0:
        _, mo_ene_full, Lpq_full = pyscf_util.get_pyscf_input_mol_g(mf)
    else:
        mo_ene_full, Lpq_full = pprpa.mo_energy, pprpa.Lpq
    slice_p, slice_i, slice_a, slice_ip, slice_ap, slice_I, slice_A = (
        choose_slice("p", nfrozen_occ, nocc, nvir, nfrozen_vir),
        choose_slice("i", nfrozen_occ, nocc, nvir, nfrozen_vir),
        choose_slice("a", nfrozen_occ, nocc, nvir, nfrozen_vir),
        choose_slice("ip", nfrozen_occ, nocc, nvir, nfrozen_vir),
        choose_slice("ap", nfrozen_occ, nocc, nvir, nfrozen_vir),
        choose_slice("I", nfrozen_occ, nocc, nvir, nfrozen_vir),
        choose_slice("A", nfrozen_occ, nocc, nvir, nfrozen_vir),
    )
    orbA, orbI, orbp, orbi, orba = (
        mf.mo_coeff[:, slice_A],
        mf.mo_coeff[:, slice_I],
        mf.mo_coeff[:, slice_p],
        mf.mo_coeff[:, slice_i],
        mf.mo_coeff[:, slice_a],
    )
    vresp, (occ_y_mat, vir_x_mat) = mf.gen_response(hermi=0), get_xy_full(
        xy, (nocc - 1) * nocc // 2
    )
    den_u = make_rdm1_unrelaxed_from_xy_full(occ_y_mat, vir_x_mat)
    den_u_ao = orbp @ np.diag(den_u) @ orbp.T.conj()
    veff_den_u = mf.mo_coeff.T.conj() @ vresp(den_u_ao) @ mf.mo_coeff
    i_prime = np.zeros((len(mo_ene_full), len(mo_ene_full)), dtype=occ_y_mat.dtype)
    i_prime[slice_p, slice_p] += contraction_2rdm_Lpq(
        occ_y_mat, vir_x_mat, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, "p", "p"
    )
    i_prime[slice_a, slice_i] += veff_den_u[slice_a, slice_i]
    for p in choose_range("p", nfrozen_occ, nocc, nvir, nfrozen_vir):
        i_prime[p, p] += mo_ene_full[p] * den_u[p - nfrozen_occ]
    if nfrozen_vir > 0:
        i_prime[slice_ap, slice_p] += contraction_2rdm_Lpq(
            occ_y_mat,
            vir_x_mat,
            Lpq_full,
            nocc,
            nvir,
            nfrozen_occ,
            nfrozen_vir,
            "ap",
            "p",
        )
        i_prime[slice_ap, slice_i] += veff_den_u[slice_ap, slice_i]
    if nfrozen_occ > 0:
        i_prime[slice_ip, slice_p] += contraction_2rdm_Lpq(
            occ_y_mat,
            vir_x_mat,
            Lpq_full,
            nocc,
            nvir,
            nfrozen_occ,
            nfrozen_vir,
            "ip",
            "p",
        )
        i_prime[slice_A, slice_ip] += veff_den_u[slice_A, slice_ip]
    i_prime_prime = np.zeros_like(i_prime)
    i_prime_prime[slice_a, slice_I] = (
        i_prime[slice_a, slice_I] - i_prime[slice_I, slice_a].T.conj()
    )
    (
        i_prime_prime[slice_A, slice_a],
        i_prime_prime[slice_I, slice_i],
        i_prime_prime[slice_ap, slice_I],
    ) = (
        i_prime[slice_A, slice_a],
        i_prime[slice_I, slice_i],
        i_prime[slice_ap, slice_I],
    )
    d_prime = np.zeros_like(i_prime_prime)
    for i in choose_range("I", nfrozen_occ, nocc, nvir, nfrozen_vir):
        for j in choose_range("i", nfrozen_occ, nocc, nvir, nfrozen_vir):
            denorm = mo_ene_full[j] - mo_ene_full[i]
            if abs(denorm) >= 1e-6:
                d_prime[i, j] = i_prime_prime[i, j] / denorm
    for a in choose_range("A", nfrozen_occ, nocc, nvir, nfrozen_vir):
        for b in choose_range("a", nfrozen_occ, nocc, nvir, nfrozen_vir):
            denorm = mo_ene_full[b] - mo_ene_full[a]
            if abs(denorm) >= 1e-6:
                d_prime[a, b] = i_prime_prime[a, b] / denorm
    x_int = i_prime_prime[slice_A, slice_I].copy()
    d_ao = (
        orbI @ d_prime[slice_I, slice_i] @ orbi.T.conj()
        + orbA @ d_prime[slice_A, slice_a] @ orba.T.conj()
    )
    d_ao += d_ao.T.conj()
    x_int += orbA.T.conj() @ vresp(d_ao) @ orbI

    def fvind(x):
        dm = orbA @ x.reshape(nvir + nfrozen_vir, nocc + nfrozen_occ) @ orbI.T.conj()
        return (orbA.T.conj() @ vresp(dm + dm.T.conj()) @ orbI).ravel()

    from pyscf.scf import cphf

    d_prime[slice_A, slice_I] = cphf.solve(
        fvind,
        mo_ene_full,
        mf.mo_occ,
        x_int,
        max_cycle=cphf_max_cycle,
        tol=cphf_conv_tol,
    )[0].reshape(nvir + nfrozen_vir, nocc + nfrozen_occ)
    i_int = -np.einsum("qp,p->qp", d_prime, mo_ene_full)
    dp_ao = mf.mo_coeff @ d_prime @ mf.mo_coeff.T.conj()
    i_int[slice_I, slice_I] -= (
        0.5 * veff_den_u[slice_I, slice_I] + orbI.T.conj() @ vresp(dp_ao) @ orbI
    )
    i_int[slice_I, slice_a] -= i_prime[slice_I, slice_a]
    for i in choose_range("p", nfrozen_occ, nocc, nvir, nfrozen_vir):
        for j in choose_range("p", nfrozen_occ, nocc, nvir, nfrozen_vir):
            if abs(mo_ene_full[j] - mo_ene_full[i]) < 1e-6:
                i_int[i, j] -= 0.5 * i_prime[i, j]
    return d_prime + d_prime.T.conj(), i_int + i_int.T.conj()


def grad_elec_ghf_ref(pprpa_grad, xy, atmlst=None, correlation_only=False):
    mf, pprpa, mol = pprpa_grad.mf, pprpa_grad.base, pprpa_grad.mf.mol
    nao, mf_grad = mol.nao_nr(), pprpa_grad.mf.Gradients()
    if atmlst is None:
        atmlst = range(mol.natm)
    nocc_all, nocc, nvir = mol.nelectron, pprpa.nocc, pprpa.nvir
    nfrozen_occ = nocc_all - nocc
    hcore_deriv, s1 = mf_grad.hcore_generator(mol), mf_grad.get_ovlp(mol)
    dm0, i_int = make_rdm1_relaxed_ghf_pprpa_ref(pprpa, mf, xy)
    dm0 = mf.mo_coeff @ dm0 @ mf.mo_coeff.conj().T
    pprpa_grad.rdm1e = dm0
    dm0_hf = mf.make_rdm1()
    i_int = mf.mo_coeff @ i_int @ mf.mo_coeff.conj().T
    if not correlation_only:
        i_int -= mf_grad.make_rdm1e(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
    from lib_pprpa.grad.grad_utils import get_xy_full

    occ_y_mat, vir_x_mat = get_xy_full(xy, pprpa.oo_dim)
    coeff_occ, coeff_vir = (
        mf.mo_coeff[:, nfrozen_occ : nfrozen_occ + nocc],
        mf.mo_coeff[:, nfrozen_occ + nocc : nfrozen_occ + nocc + nvir],
    )
    xy_ao = (
        coeff_vir @ vir_x_mat @ coeff_vir.conj().T
        + coeff_occ @ occ_y_mat @ coeff_occ.conj().T
    )
    dm0_1e, dm0_2e = (
        (dm0, dm0) if correlation_only else (dm0 + dm0_hf, dm0 + 0.5 * dm0_hf)
    )
    aux_response = isinstance(mf, _DFHF)
    if not hasattr(mf, "xc"):
        if xy.dtype == np.float64:
            vj, vk = mf_grad.get_jk(mol, (dm0_hf, dm0_2e, xy_ao))
        else:
            vj, vk = mf_grad.get_jk(mol, (dm0_hf, dm0_2e, xy_ao, xy_ao.conj()))
        vhf = np.zeros_like(vj)
        vhf[:2], vhf[2] = vj[:2] - vk[:2], -vk[2]
        if aux_response:
            vhf_aux = np.zeros_like(vj.aux)
            vhf_aux[:2, :2] = vj.aux[:2, :2] - vk.aux[:2, :2]
            vhf_aux[2, 2] = -vk.aux[2, 2] if xy.dtype == np.float64 else -vk.aux[2, 3]
            vhf = lib.tag_array(vhf, aux=vhf_aux[0, 1] + vhf_aux[1, 0] + vhf_aux[2, 2])
        aoslices, de = mol.aoslice_by_atom(), np.zeros((len(atmlst), 3), dtype=xy.dtype)
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            h1ao = hcore_deriv(ia)
            if h1ao.shape[-1] == dm0_1e.shape[-1] // 2:
                h1ao = np.asarray(
                    [scipy.linalg.block_diag(h1ao[i], h1ao[i]) for i in range(3)]
                )
            de[k] += (
                np.einsum("xij,ji->x", h1ao, dm0_1e)
                + np.einsum("xij,ji->x", vhf[0, :, p0:p1], dm0_2e[:, p0:p1]).real * 2
                + np.einsum("xij,ji->x", vhf[1, :, p0:p1], dm0_hf[:, p0:p1]).real * 2
                + np.einsum(
                    "xij,ji->x",
                    vhf[0, :, nao + p0 : nao + p1],
                    dm0_2e[:, nao + p0 : nao + p1],
                ).real
                * 2
                + np.einsum(
                    "xij,ji->x",
                    vhf[1, :, nao + p0 : nao + p1],
                    dm0_hf[:, nao + p0 : nao + p1],
                ).real
                * 2
                + np.einsum("xij,ji->x", vhf[2, :, p0:p1], xy_ao[:, p0:p1].conj()) * 2
                + np.einsum(
                    "xij,ji->x",
                    vhf[2, :, nao + p0 : nao + p1],
                    xy_ao[:, nao + p0 : nao + p1].conj(),
                )
                * 2
                + np.einsum("xij,ji->x", s1[:, p0:p1], i_int[:nao, p0:p1]).real * 2
                + np.einsum(
                    "xij,ji->x", s1[:, p0:p1], i_int[nao:, nao + p0 : nao + p1]
                ).real
                * 2
            )
            if aux_response:
                de[k] += vhf.aux[ia]
    return de.real
