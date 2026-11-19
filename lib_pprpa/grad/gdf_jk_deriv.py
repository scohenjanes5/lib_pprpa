"""Gamma-point GDF derivatives of several Coulomb/exchange energies in one integral pass.

Adapted from ``gpu4pyscf.pbc.df.grad.rhf`` (commit d65bd284) under the Apache License 2.0;
Copyright 2025-2026 The PySCF Developers.

``ejk_derivatives`` below is a multi-density counterpart of gpu4pyscf's
``_get_ejk_derivatives`` (and of ``_get_ej_derivatives`` for its Coulomb-only terms).  The
pp-RPA GDF gradient needs ``Q(D+P) - Q(P)`` of the reference Coulomb (and hybrid exchange)
energy and the exchange energy of the pair density, and upstream evaluates each of them with
every integral anew: the 3c2e integrals that fit the density, the 2c2e metric and its
derivative, the long-range Fourier terms and the short-range 3c2e derivative integrals.  On the
63-atom NV cell (gth-dzvp, naux 5292, B200) those are three calls of 48, 48 and 66 s.

Every term enters through the same four linear intermediates: the auxiliary weight ``dm_aux``
of the 2c2e derivative, the per-G auxiliary and AO-pair weights of the long-range response,
and the (AO pair, aux) weight that ``PBCsr_ejk_int3c2e_deriv`` contracts with the short-range
derivative integrals (its general mode, the one upstream uses for exchange).  Summing the
terms' contributions to these intermediates differentiates the whole sum with one pass of each
integral.

Differences from upstream ``_get_ejk_derivatives``
--------------------------------------------------
* A list of terms ``(dm, hermi, j_factor, k_factor, exxdiv, weight)`` instead of one density.
  A Coulomb term is fitted directly, ``metric @ (P|dm)``, instead of through the trace of its
  exchange factors, so Coulomb-only densities need no factorisation.
* The short-range weight is symmetrised once for all terms, ``W + W^T``.  Upstream doubles a
  symmetric (``hermi=1``) weight and symmetrises an antisymmetric (``hermi=2``) one; for a
  symmetric weight the two are the same.
* Gamma point and real densities only (the GDF pp-RPA gradient's scope).
"""

import ctypes
import math

import numpy as np
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, empty_aligned, ndarray
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.df.grad.rhf import factorize_dm
from gpu4pyscf.df.int3c2e_bdiv import _split_l_ctr_pattern, get_ao_pair_loc
from gpu4pyscf.gto.mole import RysIntEnvVars, _scale_sp_ctr_coeff
from gpu4pyscf.pbc.df import aft_jk, ft_ao
from gpu4pyscf.pbc.df.grad.rhf import _gen_metric_solver, _get_shl_pair_batch_size
from gpu4pyscf.pbc.df.int2c2e import Int2c2eOpt, _estimate_sr_2c2e_rcut
from gpu4pyscf.pbc.df.int3c2e import (MAX_IMGS_PER_TASK, POOL_SIZE, int3c2e_scheme,
                                      libpbc)
from gpu4pyscf.pbc.df.rsdf_builder import LINEAR_DEP_THR, _precontract_j2c_aux_coeff
from gpu4pyscf.pbc.grad.rhf import contract_h1e_dm
from gpu4pyscf.pbc.grad.rks_stress import (
    _get_weighted_coulG_strain_derivatives as get_wcoulG)
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.gto.cell import get_Gv_weights
from gpu4pyscf.pbc.tools.pbc import madelung

from lib_pprpa.gpu_gdf import cderi_fit
from lib_pprpa.gpu_mem import free_bytes

__all__ = ['ejk_derivatives']


def ejk_derivatives(int3c2e_opt, terms, omega=None, verbose=None,
                    linear_dep_threshold=LINEAR_DEP_THR, with_df=None, release_cderi=False):
    '''Nuclear (natm, 3) and strain (3, 3) derivatives of
    sum_t weight_t * (j_factor_t E_J[dm_t] - k_factor_t/2 E_K[dm_t]) per cell, each term as
    gpu4pyscf's ``pbc.df.grad.rhf._get_ejk_derivatives(int3c2e_opt, dm, hermi, j_factor,
    k_factor, exxdiv, omega)``.

    terms : iterable of (dm, hermi, j_factor, k_factor, exxdiv, weight); dm a real (nao, nao)
        density, symmetric for hermi=1, antisymmetric for hermi=2 (no Coulomb energy).
    with_df : a built gpu4pyscf GDF object of the same cell, auxiliary basis and metric
        threshold whose cderi L = C^T (raw|pq) is stored.  The fit of the densities, metric
        applied, is then read from it (C L.dm, C a^T L b, :func:`lib_pprpa.gpu_gdf.cderi_fit`)
        instead of evaluated with the 0th-order integrals; C is the eigendecomposed metric
        factor gpu4pyscf's cderi carries, the one the metric below is built from.
    release_cderi : free the device rows of with_df's :class:`lib_pprpa.gpu_gdf.TieredCderi`
        once the fit is read (``release_device``), for the Fourier blocks that follow.
    '''
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    if len(int3c2e_opt.bvkmesh_Ls) != 1:
        raise NotImplementedError('ejk_derivatives is Gamma-point only')
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()

    # Coulomb terms: (dm in the sorted AO order, weight * j_factor); exchange terms:
    # (left and right factors in the sorted AO order, weight * k_factor, dm, exxdiv)
    jterms = []
    kterms = []
    jdms_orig = []      # Coulomb densities in the original AO order (cderi fit)
    kpairs_orig = []    # exchange factors (r, l) in the original AO order (cderi fit)
    for dm, hermi, j_factor, k_factor, exxdiv, weight in terms:
        if hermi not in (1, 2):
            raise ValueError('ejk_derivatives supports hermi=1 or 2')
        dm = cp.asarray(dm, dtype=np.float64)
        if hermi == 1 and j_factor * weight != 0:
            jterms.append((cell.apply_C_mat_CT(dm), weight * j_factor))
            jdms_orig.append(dm)
        if k_factor * weight != 0:
            factor_l, factor_r = factorize_dm(dm, hermi)
            if factor_l.shape[-1] == 0:
                continue
            kpairs_orig.append((factor_l if factor_r is None else factor_r, factor_l))
            factor_l = cell.apply_C_dot(factor_l, axis=0)
            factor_r = (factor_l if factor_r is None
                        else cell.apply_C_dot(factor_r, axis=0))
            kterms.append((factor_l, factor_r, weight * k_factor, dm, exxdiv))
    if not jterms and not kterms:
        return np.zeros((cell.natm, 3)), np.zeros((3, 3))
    nocc2 = sum(fl.shape[1]**2 for fl, *_ in kterms)
    noccmax = max([fl.shape[1] for fl, *_ in kterms], default=1)
    nj = len(jterms)

    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    # AOs of the sorted cell, the basis of the integrals
    nao = kterms[0][0].shape[0] if kterms else jterms[0][0].shape[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)
    aux_loc = auxcell.ao_loc
    naux = int(aux_loc[-1])

    mem_free = free_bytes()
    mem_avail = mem_free - naux*nocc2*8 - nao**2*8*(nj+1)
    batch_size = max(1, min(naux, int(mem_avail*.5/(nao_pair*8))))
    assert batch_size < POOL_SIZE
    blksize = max(1, min(naux, int(mem_avail*.4/(nao**2*2*8))//8*8))
    log.debug1('%.3f GB free memory. nao_pair=%d naux=%d batch_size=%d blksize=%d, '
               '%d Coulomb and %d exchange terms', mem_free*1e-9, nao_pair, naux,
               batch_size, blksize, nj, len(kterms))

    jdms = cp.stack([dm.ravel() for dm, _ in jterms]) if nj else None
    precision = auxcell.precision * 1e-6
    auxcell.rcut = _estimate_sr_2c2e_rcut(auxcell, int3c2e_opt.omega, precision)
    int2c2e_opt = Int2c2eOpt(auxcell)

    ################################
    # LR part: Fourier transforms and kernels
    ft_opt = ft_ao.FTOpt.from_intopt(int3c2e_opt)
    eval_ft = ft_opt.ft_evaluator(
        compressing=False, cart=True, original_ao_order=False)[0]
    if omega is None:
        omega = 0
    else:
        omega = abs(omega)
    if omega < int3c2e_opt.omega:
        mesh = int3c2e_opt.mesh
    else:
        assert cell.dimension == 3
        mesh = [1] * 3
    Gv, _, kws = get_Gv_weights(cell, mesh)
    ngrids = len(Gv)
    wcoulG_LR0, wcoulG_LR1 = get_wcoulG(cell, Gv, int3c2e_opt.omega)
    if omega != 0:
        wcoulG_0, wcoulG_1 = get_wcoulG(cell, Gv, omega)
        wcoulG_LR0 -= wcoulG_0
        wcoulG_LR1 -= wcoulG_1
    wcoulG_SR_at_G0 = np.pi / int3c2e_opt.omega**2 * kws
    wcoulG_LR0[0] -= wcoulG_SR_at_G0
    wcoulG_LR1[:,:,0] += wcoulG_SR_at_G0 * cp.eye(3)

    def g_block_size():
        mem_avail = free_bytes()
        Gsize = (max(nao**2, naux) + max(nao*noccmax, naux) + naux + nao_pair
                 + nao**2 + noccmax**2)
        Gblksize = min(int(mem_avail*.8//(Gsize*16))//32*32, ngrids)
        if Gblksize < 1:
            raise RuntimeError('Insufficient GPU memory for GDF Fourier buffers')
        return Gblksize

    def pq_g(p0, p1):
        # (pq|G) of the G vectors [p0, p1) on the full AO square, real and imaginary parts
        # interleaved: the compressed lower triangle mirrored
        pqG = eval_ft(Gv[p0:p1]).view(np.float64).reshape(nao, nao, (p1 - p0)*2)
        pqG[j_addr, i_addr] = pqG[i_addr, j_addr]
        return pqG

    def ij_g(pqG, factor_l, factor_r):
        # ijG[i,j] = sum_pq r[p,i] (pq|G) l[q,j]
        tmp = contract('pqG,pi->iqG', pqG, factor_r)
        return contract('iqG,qj->ijG', tmp, factor_l)

    if with_df is None:
        # 0th-order integrals: the SR 3c2e fit, j2c and the LR fit, then the metric
        eval_j3c, _, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
            aux_batch_size=batch_size, cart=True)
        # SR 3c2e: (P|dm) of the Coulomb terms and (P|l_i r_j) of the exchange terms
        rho = cp.zeros((nj, naux))
        j3c_oo = [cp.empty((naux, fl.shape[1], fl.shape[1])) for fl, *_ in kterms]
        aux0 = aux1 = 0
        j3c_full = cp.zeros((nao, nao, blksize))
        buf = cp.empty((batch_size, nao_pair))
        for kbatch in range(len(aux_offsets) - 1):
            compressed = eval_j3c(aux_batch_id=kbatch, out=buf)[:,0,:]
            for k0, k1 in lib.prange(0, compressed.shape[1], blksize):
                dk = k1 - k0
                aux0, aux1 = aux1, aux1 + dk
                j3c = j3c_full[:,:,:dk]
                j3c[j_addr,i_addr] = j3c[i_addr,j_addr] = compressed[:,k0:k1]
                if nj:
                    rho[:,aux0:aux1] = jdms.dot(j3c.reshape(nao*nao, dk))
                for (factor_l, factor_r, *_), oo in zip(kterms, j3c_oo):
                    tmp = contract('pqr,pi->iqr', j3c, factor_r)
                    contract('iqr,qj->rij', tmp, factor_l, out=oo[aux0:aux1])
        j3c_full = buf = eval_j3c = j3c = tmp = compressed = None
        t0 = log.timer_debug1('contract dm', *t0)

        j2c = int2c2e_opt.int2c2e(sort_output=False)
        Gblksize = g_block_size()
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            nGv = p1 - p0
            auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1]).T
            auxGw = (auxG * wcoulG_LR0[p0:p1]).view(np.float64)
            contract('iG,jG->ij', auxG.view(np.float64), auxGw, beta=1, out=j2c)
            # conj((r|G)^{[0]}) (ij|G)^{[0]}
            pqG = pq_g(p0, p1)
            if nj:
                rhoG = jdms.dot(pqG.reshape(nao*nao, nGv*2))
                rho += rhoG.dot(auxGw.T)
            for (factor_l, factor_r, *_), oo in zip(kterms, j3c_oo):
                contract('rG,ijG->rij', auxGw, ij_g(pqG, factor_l, factor_r), beta=1, out=oo)
        pqG = auxG = auxGw = rhoG = None


        ################################
        # (d/dX P|Q) contributions
        j2c = auxcell.apply_CT_mat_C(j2c)
        if auxcell.cell.cart:
            raise NotImplementedError
        aux_coeff = cp.asarray(auxcell.ctr_coeff)
        solve_j2c = _gen_metric_solver(j2c, linear_dep_threshold, auxcell.dimension)
        metric = aux_coeff.dot(solve_j2c(aux_coeff.T))
        j2c = aux_coeff = solve_j2c = None
        # the metric acts on the auxiliary index only: transform the exchange blocks in place
        dm_oo = [_aux_apply(metric, oo, out=oo) for oo in j3c_oo]
        auxvec = rho.dot(metric.T)          # fitted Coulomb densities, (nj, naux)
        metric = rho = j3c_oo = None
    else:
        # the fit from the stored cderi L = C^T (raw|pq): (raw|.) metric-applied = C L.(.)
        aux_coeff = cp.asarray(_precontract_j2c_aux_coeff(
            auxcell, None, 0, int3c2e_opt.omega, linear_dep_threshold)[0][0])
        if aux_coeff.shape[0] != naux:
            raise ValueError("with_df's auxiliary basis is not the integrals' one")
        rho_df, blocks = cderi_fit(with_df, jdms_orig, kpairs_orig, verbose=verbose)
        if release_cderi:
            with_df._cderi[0].release_device()
        if rho_df.shape[1] != aux_coeff.shape[1]:
            raise ValueError('with_df: its cderi was fitted with another metric truncation')
        auxvec = rho_df.dot(aux_coeff.T)
        rho_df = None
        dm_oo = [_aux_apply(aux_coeff, blk) for blk in blocks]
        blocks = aux_coeff = None
        t0 = log.timer_debug1('fit from the cderi', *t0)
    jw = cp.asarray([w for _, w in jterms])

    dm_aux = contract('ti,tj->ij', auxvec * jw[:,None], auxvec) if nj else cp.zeros((naux, naux))
    for (*_, wk, _, _), oo in zip(kterms, dm_oo):
        contract('rij,sji->rs', oo, oo, alpha=-.5*wk, beta=1, out=dm_aux)
    ejk, sigma = int2c2e_opt.energy_derivatives(dm_aux, omega=-int3c2e_opt.omega)
    ejk = cp.asarray(-ejk)
    sigma = cp.asarray(-sigma)
    t0 = log.timer_debug1('contract int2c2e_deriv', *t0)

    ################################
    # LR part response
    bas_ij_idx, bas_ij_img_idx, shl_pair_offsets = aft_jk._generate_shl_pairs(ft_opt)
    aft_envs = ft_opt.aft_envs
    aux_ft_envs = RysIntEnvVars.new(
        auxcell.natm, auxcell.nbas, auxcell._atm, auxcell._bas,
        _scale_sp_ctr_coeff(auxcell), auxcell.ao_loc)
    shm_size = aft_jk._estimate_max_shm_size(cell, (1, 0))
    ejk_lr = cp.zeros((cell.natm, 3))
    ejk_aux = cp.zeros((cell.natm, 3))
    sigma_lr = cp.zeros((3, 3))
    sigma_aux = cp.zeros((3, 3))
    sigma_G = cp.zeros((3, 3))
    Gblksize = g_block_size()
    for p0, p1 in lib.prange(0, ngrids, Gblksize):
        nGv = p1 - p0
        pqG = pq_g(p0, p1)
        # all [0] order terms contracted to the auxiliary index: (ji|r)^{[0]} metric (G|ij)^{[0]}
        dm_auxG = cp.zeros((naux, nGv*2))
        if nj:
            rhoG = jdms.dot(pqG.reshape(nao*nao, nGv*2))
            dm_auxG += (auxvec * jw[:,None]).T.dot(rhoG)
        ijGs = []
        for (factor_l, factor_r, wk, *_), oo in zip(kterms, dm_oo):
            ijG = ij_g(pqG, factor_l, factor_r)
            contract('rji,ijG->rG', oo, ijG, -.5*wk, 1, out=dm_auxG)
            ijGs.append(ijG)
        ijG = ijGs = None
        dm_auxG = dm_auxG.view(np.complex128)

        auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1]).T
        vG = cp.einsum('rg,rg->g', dm_auxG, auxG.conj()).real
        # -(ji|r)^{[0]} metric (r|G)^{[1]} (G|s)^{[0]} metric (ij|s)^{[0]}
        dm_auxG1 = contract('sr,sG->rG', dm_aux, auxG.view(np.float64)).view(np.complex128)
        vG *= 2
        vG -= cp.einsum('rg,rg->g', dm_auxG1, auxG.conj()).real
        sigma_G += .5 * cp.einsum('g,xyg->xy', vG, wcoulG_LR1[:,:,p0:p1])
        dm_auxG -= dm_auxG1
        dm_auxG *= wcoulG_LR0[p0:p1]
        dm_auxG = dm_auxG.view(np.float64)
        GvT = cp.asarray(Gv[p0:p1].T.ravel())
        err = libpbc.PBC_ft_ao_deriv(
            ctypes.cast(ejk_aux.data.ptr, ctypes.c_void_p),
            ctypes.cast(sigma_aux.data.ptr, ctypes.c_void_p),
            lib.c_null_ptr(),
            ctypes.cast(dm_auxG.data.ptr, ctypes.c_void_p),
            ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
            ctypes.byref(aux_ft_envs),
            ctypes.c_int(nGv))
        if err != 0:
            raise RuntimeError('ft_ao_deriv failed')

        # (ji|r)^{[0]} metric (G|ij)^{[1]} (r|G)^{[0]}: the AO-pair weight, symmetric
        # (PBC_ft_aopair_ek_deriv processes the lower triangle)
        auxG_conj = (auxG.conj() * wcoulG_LR0[p0:p1]).view(np.float64)
        dm_vG = cp.zeros((nao, nao, nGv*2))
        if nj:
            vGj = (auxvec * jw[:,None]).dot(auxG_conj)
            dm_vG += contract('tpq,tG->pqG', jdms.reshape(nj, nao, nao), vGj)
        for (factor_l, factor_r, wk, *_), oo in zip(kterms, dm_oo):
            nocc = factor_l.shape[1]
            dm_ooG = oo.reshape(naux, nocc*nocc).T.dot(auxG_conj).reshape(nocc, nocc, nGv*2)
            tmp = contract('jiG,qi->jqG', dm_ooG, factor_r)
            contract('jqG,pj->pqG', tmp, factor_l, -.5*wk, 1, out=dm_vG)
        dm_ooG = tmp = auxG = auxG_conj = None
        err = libpbc.PBC_ft_aopair_ek_deriv(
            ctypes.cast(ejk_lr.data.ptr, ctypes.c_void_p),
            ctypes.cast(sigma_lr.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm_vG.data.ptr, ctypes.c_void_p),
            ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
            ctypes.byref(aft_envs),
            ctypes.c_int(len(shl_pair_offsets) - 1),
            ctypes.c_int(nGv),
            ctypes.c_int(shm_size),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ft_opt.permutation_symmetry))
        if err != 0:
            raise RuntimeError('PBC_ft_aopair_ek_deriv failed')
    dm_vG = dm_auxG = pqG = None
    ejk_lr *= 2  # due to i>=j symmetry in the CUDA kernel
    sigma_lr *= 2
    ejk += ejk_lr + ejk_aux
    sigma += sigma_lr + sigma_aux + sigma_G
    ft_opt = eval_ft = dm_aux = None
    t0 = log.timer_debug1('LR coulomb', *t0)

    ################################
    # SR int3c2e response: one pass of the derivative integrals contracted with the summed
    # (AO pair, aux) weight
    _, gout_stride, shm_size = int3c2e_scheme(gout_width=54, deriv=(1,0,0))
    lmax = cell.uniq_l_ctr[:,0].max()
    laux = auxcell.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
    l_ctr_aux_offsets = np.append(0, np.cumsum(auxcell.l_ctr_counts))
    l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
        l_ctr_aux_offsets, auxcell.uniq_l_ctr, batch_size)
    ksh_offsets_cpu = l_ctr_aux_offsets
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu, dtype=np.int32)
    shl_pair_batch_size = _get_shl_pair_batch_size(
        ksh_offsets_cpu[1:] - ksh_offsets_cpu[:-1], 1)
    bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block=shl_pair_batch_size)
    ao_pair_loc = get_ao_pair_loc(cell.uniq_l_ctr[:,0], int3c2e_opt.bas_ij_cache, cart=True)
    diffuse_exps = cp.asarray(int3c2e_opt.diffuse_exps)
    diffuse_coefs = cp.asarray(int3c2e_opt.diffuse_coefs)
    log_cutoff = math.log(int3c2e_opt.cutoff)

    assert cell.natm == auxcell.natm
    ejk_sr = cp.zeros((cell.natm, 3))
    ejk_aux_sr = cp.zeros((cell.natm, 3))
    sigma_sr = cp.zeros((3, 3))
    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty(workers * POOL_SIZE*(MAX_IMGS_PER_TASK+2) + 1, dtype=np.uint32)
    head = pool[-1:]
    task_pool = empty_aligned((workers, POOL_SIZE*16), np.int32, alignment=128)
    aux0 = aux1 = 0
    buf = cp.empty((nao_pair*batch_size))
    for kbatch in range(len(uniq_l_ctr_aux)):
        aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
        naux_in_batch = aux_loc[ksh_offsets_cpu[kbatch+1]] - aux_ao_offset
        compressed = ndarray((nao_pair, naux_in_batch), buffer=buf)
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            w = cp.zeros((nao, nao, dk))
            if nj:
                w += contract('tpq,tr->pqr', jdms.reshape(nj, nao, nao),
                              auxvec[:,aux0:aux1] * jw[:,None])
            for (factor_l, factor_r, wk, *_), oo in zip(kterms, dm_oo):
                tmp = contract('rji,qj->iqr', oo[aux0:aux1], factor_l)
                contract('iqr,pi->pqr', tmp, factor_r, -.5*wk, 1, out=w)
            w += w.transpose(1, 0, 2)
            cp.take(w.reshape(-1, dk), pair_addresses, axis=0, out=compressed[:,k0:k1])
            w = tmp = None
        err = libpbc.PBCsr_ejk_int3c2e_deriv(
            ctypes.cast(ejk_sr.data.ptr, ctypes.c_void_p),
            ctypes.cast(ejk_aux_sr.data.ptr, ctypes.c_void_p),
            ctypes.cast(sigma_sr.data.ptr, ctypes.c_void_p),
            lib.c_null_ptr(),
            ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
            ctypes.c_double(-int3c2e_opt.omega),
            ctypes.byref(int3c2e_opt.int3c2e_envs),
            ctypes.cast(pool.data.ptr, ctypes.c_void_p),
            ctypes.cast(task_pool.data.ptr, ctypes.c_void_p),
            ctypes.cast(head.data.ptr, ctypes.c_void_p),
            ctypes.c_int(shm_size_max),
            ctypes.c_int(len(shl_pair_offsets) - 1),
            ctypes.c_int(1),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets_gpu[kbatch:].data.ptr, ctypes.c_void_p),
            ctypes.cast(int3c2e_opt.img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(int3c2e_opt.img_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(aux_ao_offset),
            ctypes.c_int(auxcell.nbas),
            ctypes.c_int(naux_in_batch),
            ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_coefs.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff))
        if err != 0:
            raise RuntimeError('PBCsr_ejk_int3c2e_deriv failed')
    ejk += ejk_sr + ejk_aux_sr
    sigma += sigma_sr
    t0 = log.timer_debug1('contract int3c2e_ejk_deriv', *t0)
    ejk = ejk.get()
    sigma = sigma.get()

    # Ewald correction of the exchange terms with exxdiv='ewald'
    ewald = [(wk, dm) for _, _, wk, dm, exxdiv in kterms if exxdiv == 'ewald']
    if ewald and (cell.dimension == 3 or
                  (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum')):
        s0 = int1e.int1e_ovlp(cell)
        s1 = int1e.int1e_ipovlp(cell)
        kpts = np.zeros((1, 3))
        weighted_coulG_at_G0 = madelung(cell, kpts, omega=-omega)
        exx_0, exx_1 = aft_jk._exxdiv_ewald_strain_deriv(cell.cell, kpts, -omega)
        for wk, dm in ewald:
            k_dm = contract('pq,qr->pr', dm, s0)
            k_dm = contract('pr,rs->ps', k_dm, dm)
            # s1 and k_dm are in the original AO order: contract with the original cell
            ejk += (contract_h1e_dm(cell.cell, s1, k_dm, hermi=1)
                    * (.5 * wk * weighted_coulG_at_G0))
            ek_G0 = float(cp.einsum('ij,ji->', s0, k_dm).real.get())
            fac = wk * .25
            sigma -= fac * exx_1 * ek_G0
            sigma -= 2 * fac * exx_0 * int1e.ovlp_strain_deriv(cell.cell, k_dm, kpts)
    return ejk, sigma


def _aux_apply(m, blk, out=None):
    '''out[:, i] = m @ blk[:, i] over the auxiliary (leading) index of blk (n, ni, nj), in
    slices of i within the free memory; ``out`` may be ``blk`` itself when m is square.'''
    ni, nj = blk.shape[1], blk.shape[2]
    if out is None:
        out = cp.empty((m.shape[0], ni, nj))
    step = min(ni, max(1, int(free_bytes()*.4 // (m.shape[0]*nj*8))))
    for i0, i1 in lib.prange(0, ni, step):
        out[:, i0:i1] = contract('uv,vij->uij', m, blk[:, i0:i1])
    return out
