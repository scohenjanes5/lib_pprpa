'''Gamma-point GDF factors L[P, p, q] of the active orbitals on the GPU, without the AO cderi.

The integrals are those of gpu4pyscf's GDF build (``rsdf_builder.compressed_cderi_j_only``):
short-range int3c2e with lattice sums, the long-range part through the analytic Fourier
transform of the AO pairs on the rsdf mesh, and the j2c metric.  gpu4pyscf stores the
compressed AO cderi (naux x nao_pairs) in host memory; here the short-range integrals of one
block of raw auxiliary functions at a time are scattered into a dense AO slab and transformed to
the active orbitals, the long-range part is transformed to the active orbitals in blocks of G
vectors, and the metric is applied last.  Real orbitals, 3D cells, Gamma point only.
'''

import copy

import numpy as np
import cupy as cp
from cupy import cublas
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.df.df import make_auxcell
from gpu4pyscf.pbc.df import ft_ao
from gpu4pyscf.pbc.df.int3c2e import POOL_SIZE, SRInt3c2eOpt
from gpu4pyscf.pbc.df.rsdf_builder import (_guess_omega, _precontract_j2c_aux_coeff,
                                           _weighted_coulG_kpts)

from lib_pprpa.gpu_mem import empty_pinned, free_bytes, plan_budget

__all__ = ['gdf_active_factors']

# G vectors per long-range block are a multiple of this (gpu4pyscf's Gblksize alignment).
_G_ALIGN = 32

# src[P, k] (AO pair P of a batch, function or G vector k) into the symmetric AO slab
# slab[k, i, j] = slab[k, j, i], (i, j) = (rows[P], cols[P])
_SCATTER = cp.ElementwiseKernel(
    'raw T src, raw int64 rows, raw int64 cols, int64 nk, int64 nao',
    'raw T slab',
    '''
    const long long P = i / nk;
    const long long base = (i - P * nk) * nao * nao;
    slab[base + rows[P] * nao + cols[P]] = src[i];
    slab[base + cols[P] * nao + rows[P]] = src[i];
    ''',
    'pprpa_gdf_scatter_pairs')


def gdf_active_factors(with_df, mo_act, max_memory=None, verbose=None):
    '''GDF factors of the active orbitals ``mo_act`` (nao, n) for the auxiliary basis of the
    PySCF GDF object ``with_df`` (its ``auxcell`` built, ``with_j3c`` not needed).

    Kwargs:
        max_memory : device bytes the blocks may fill (default: the free device memory).
    Returns:
        A pinned NumPy array (naux, n, n), with (P|pq) = sum_P L[P, p, q] L[P, r, s].
    '''
    cell = with_df.cell
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    if with_df.auxcell is None:
        raise ValueError('with_df.auxcell is not built (with_df.build(with_j3c=False))')
    if cell.dimension != 3:
        raise NotImplementedError('gdf_active_factors supports 3D cells only')
    auxcell = make_auxcell(cell, copy.deepcopy(with_df.auxcell.basis), None)
    rsdf_omega = _guess_omega(cell)
    opt = SRInt3c2eOpt(cell, auxcell, omega=-rsdf_omega, bvk_kmesh=np.ones(3, dtype=int)).build()
    scell, sauxcell = opt.cell, opt.auxcell
    aux_coeff = _precontract_j2c_aux_coeff(sauxcell, None, 0, rsdf_omega,
                                           with_df.linear_dep_threshold)[0][0]
    naux_cart, naux = aux_coeff.shape
    nao = cell.nao
    rows, cols = cp.divmod(cp.asarray(opt.pair_and_diag_indices()[0], dtype=np.int64), nao)
    C = cp.asarray(mo_act, dtype=np.float64)
    n = C.shape[1]
    ip, iq = cp.tril_indices(n)
    npq = ip.size
    agg = scell.aggregate_shl_pairs(opt.bas_ij_cache, ft_ao.ft_ao_scheme()[0])
    raw = cp.zeros((naux_cart, npq))

    def budget():
        return plan_budget(free_bytes()) if max_memory is None else int(max_memory)

    # short range.  An int3c2e launch costs a large fixed part, so the launches are as few as
    # the memory allows: all AO pairs in one batch, and aux groups of the most functions whose
    # launch output (nao_pairs each) and slab (nao^2 each) fit.  gpu4pyscf takes the whole aux
    # basis in one launch only without an aux_batch_size (any value groups it per l pattern),
    # and caps a launch at POOL_SIZE aux shells.
    pair_blk = rows.size
    aux_blk = budget() // (8 * (nao * nao + pair_blk))
    if aux_blk < 1:
        raise MemoryError('GDF active factors: no room for one aux function')
    aux_batch = None if aux_blk >= naux_cart else min(aux_blk, POOL_SIZE)
    eval_j3c, _, pair_offsets, aux_offsets = opt.int3c2e_evaluator(
        ao_pair_batch_size=pair_blk, aux_batch_size=aux_batch, bas_ij_aggregated=agg)[:4]
    pair_offsets = np.asarray(pair_offsets)
    aux_offsets = np.asarray(aux_offsets)
    batch_max = int(np.diff(pair_offsets).max())
    group_max = int(np.diff(aux_offsets).max())
    buf = cp.empty(batch_max * group_max)
    aux_blk = budget() // (8 * nao * nao)
    if aux_blk < group_max:
        raise MemoryError(f'GDF active factors: no room for an aux group of {group_max}')
    blocks = _aux_blocks(aux_offsets, aux_blk)
    log.debug('GDF active factors: naux=%d (raw %d) nao_pairs=%d n=%d, %d aux blocks of <= %d, '
              '%d pair batches', naux, naux_cart, rows.size, n, len(blocks), aux_blk,
              len(pair_offsets) - 1)
    for g0, g1 in blocks:
        a0, a1 = int(aux_offsets[g0]), int(aux_offsets[g1])
        slab = cp.zeros((a1 - a0, nao, nao))
        for b in range(len(pair_offsets) - 1):
            p0, p1 = pair_offsets[b], pair_offsets[b + 1]
            for g in range(g0, g1):
                j3c = eval_j3c(shl_pair_batch_id=b, aux_batch_id=g, out=buf)
                _scatter_pairs(slab[int(aux_offsets[g]) - a0:], j3c, rows[p0:p1], cols[p0:p1])
        for k in range(a1 - a0):
            raw[a0 + k] = (C.T @ (slab[k] @ C))[ip, iq]
        slab = None
    buf = eval_j3c = None
    t0 = log.timer('GDF active factors, short range', *t0)

    # long range, once: sum_G conj(FT[P](G)) coulG(G) FT[pq](G), FT[pq] from the AO pairs
    mesh = opt.mesh
    Gv = scell.get_Gv(mesh)
    auxG = ft_ao.ft_ao(sauxcell, Gv).T.conj()
    auxG *= cp.asarray(_weighted_coulG_kpts(sauxcell, mesh, 0, rsdf_omega)).ravel()
    eval_ft, ft_offsets = ft_ao.FTOpt.from_intopt(opt).ft_evaluator(
        pair_blk, bas_ij_aggregated=agg)
    if not np.array_equal(pair_offsets, ft_offsets):
        raise RuntimeError('int3c2e and FT pair batches differ')
    Cz = C.astype(np.complex128)
    # per G vector: its complex AO slab, the FT batch, rho with its real and imaginary
    # copies, and those of the auxG column, for the two real GEMMs
    gblk = (budget() // (16 * (nao * nao + batch_max + 2 * npq + naux_cart))
            // _G_ALIGN * _G_ALIGN)
    if gblk < _G_ALIGN:
        raise MemoryError(f'GDF active factors: no room for {_G_ALIGN} G vectors')
    log.debug('GDF active factors: LR mesh %s, %d G vectors in blocks of %d',
              list(mesh), len(Gv), gblk)
    for G0, G1 in lib.prange(0, len(Gv), gblk):
        slab = cp.zeros((G1 - G0, nao, nao), dtype=np.complex128)
        for b in range(len(pair_offsets) - 1):
            p0, p1 = pair_offsets[b], pair_offsets[b + 1]
            _scatter_pairs(slab, eval_ft(Gv[G0:G1], b), rows[p0:p1], cols[p0:p1])
        rho = cp.empty((G1 - G0, npq), dtype=np.complex128)
        for k in range(G1 - G0):
            rho[k] = (Cz.T @ (slab[k] @ Cz))[ip, iq]
        slab = None
        # raw += Re(auxG rho) = auxG.real rho.real - auxG.imag rho.imag
        A = auxG[:, G0:G1]
        cublas.gemm('N', 'N', cp.ascontiguousarray(A.real), cp.ascontiguousarray(rho.real),
                    out=raw, beta=1.0)
        cublas.gemm('N', 'N', cp.ascontiguousarray(A.imag), cp.ascontiguousarray(rho.imag),
                    out=raw, alpha=-1.0, beta=1.0)
        rho = None
    auxG = eval_ft = None
    t0 = log.timer('GDF active factors, long range', *t0)

    # metric, in aux rows; unpacked to (naux, n, n) on the device before the copy out
    out = empty_pinned((naux, n, n))
    rblk = max(1, budget() // (8 * (npq + n * n)))
    for r0, r1 in lib.prange(0, naux, rblk):
        Lr = aux_coeff[:, r0:r1].T @ raw
        full = cp.empty((r1 - r0, n, n))
        full[:, ip, iq] = Lr
        full[:, iq, ip] = Lr
        full.get(out=out[r0:r1])
        Lr = full = None
    log.timer('GDF active factors, metric', *t0)
    return out


def _scatter_pairs(slab, src, rows, cols):
    '''Write the AO pairs of ``src`` (npair, ..., nk) into slab[k], k < nk, both triangles.'''
    src = cp.ascontiguousarray(src)
    nk = src.size // rows.size
    _SCATTER(src, rows, cols, nk, slab.shape[-1], slab, size=src.size)


def _aux_blocks(aux_offsets, aux_blk):
    '''[(g0, g1)]: runs of whole aux groups with at most ``aux_blk`` functions each.'''
    blocks, g0, ngroup = [], 0, len(aux_offsets) - 1
    while g0 < ngroup:
        g1 = g0 + 1
        while g1 < ngroup and aux_offsets[g1 + 1] - aux_offsets[g0] <= aux_blk:
            g1 += 1
        blocks.append((g0, g1))
        g0 = g1
    return blocks
