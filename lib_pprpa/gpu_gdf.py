'''Gamma-point GDF factors L[P, p, q] of the active orbitals on the GPU, without the AO cderi.

The integrals are those of gpu4pyscf's GDF build (``rsdf_builder.compressed_cderi_j_only``):
short-range int3c2e with lattice sums, the long-range part through the analytic Fourier
transform of the AO pairs on the rsdf mesh, and the j2c metric.  gpu4pyscf stores the
compressed AO cderi (naux x nao_pairs) in host memory; here the short-range integrals of one
block of raw auxiliary functions at a time are scattered into a dense AO slab and transformed to
the active orbitals, the long-range part is transformed to the active orbitals in blocks of G
vectors, and the metric is applied last.  Real orbitals, 3D cells, Gamma point only.

When a gpu4pyscf GDF object already holds that cderi (the GPU SCF built it),
:func:`cderi_active_factors` transforms it instead of evaluating the integrals again, and
:func:`cderi_pair_factors` builds the pair operator's packed factors from it on the device.
:func:`build_tiered_gdf` builds that cderi with its rows split between device memory, mapped
host memory and a scratch file (:class:`TieredCderi`), for cells whose cderi does not fit the
host; its builder is adapted from gpu4pyscf's ``rsdf_builder.compressed_cderi_j_only`` (commit
d65bd284, Apache License 2.0, Copyright 2025-2026 The PySCF Developers).
'''

import copy
import os
import tempfile
import weakref

import numpy as np
import cupy as cp
from cupy import cublas
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.df.df import make_auxcell
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf.lib.cupy_helper import empty_mapped
from gpu4pyscf.pbc.df import ft_ao
from gpu4pyscf.pbc.df.int3c2e import POOL_SIZE, SRInt3c2eOpt
from gpu4pyscf.pbc.df.rsdf_builder import (_guess_omega, _precontract_j2c_aux_coeff,
                                           _weighted_coulG_kpts)

from lib_pprpa.gpu_mem import empty_pinned, free_bytes, plan_budget
from lib_pprpa.pprpa_eri_gpu import gdf_pair_factors

__all__ = ['TieredCderi', 'build_tiered_gdf', 'cderi_active_factors', 'cderi_pair_factors',
           'gdf_active_factors']

# cudaMemcpyDefault: the direction follows from the (unified) addresses
_MEMCPY_DEFAULT = 4

# share of max_memory a tiered cderi leaves to the rest of the run (the Davidson subspace, the
# gradient's nao^2 matrices, the CPU reference objects)
_HOST_RESERVE = 0.1

# G vectors per long-range block are a multiple of this (gpu4pyscf's Gblksize alignment).
_G_ALIGN = 32

# src[P * sp + k * sk] (AO pair P of a batch; function or G vector k) into the symmetric AO
# slab slab[k, i, j] = slab[k, j, i], (i, j) = (rows[P], cols[P]); one thread per (k, P)
_SCATTER = cp.ElementwiseKernel(
    'raw T src, raw int64 rows, raw int64 cols, int64 npair, int64 sp, int64 sk, int64 nao',
    'raw T slab',
    '''
    const long long k = i / npair;
    const long long P = i - k * npair;
    const long long base = k * nao * nao;
    slab[base + rows[P] * nao + cols[P]] = src[P * sp + k * sk];
    slab[base + cols[P] * nao + rows[P]] = src[P * sp + k * sk];
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
    opt, rsdf_omega, aux_coeff = _rsdf_setup(cell, auxcell, with_df.linear_dep_threshold)
    scell = opt.cell
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
    Gv, auxG = _lr_aux(opt, rsdf_omega)
    eval_ft = _ft_evaluator(opt, pair_blk, agg, pair_offsets)
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


def cderi_active_factors(gpu_df, mo_act, max_memory=None, verbose=None):
    '''GDF factors of the active orbitals ``mo_act`` (nao, n) from the compressed AO cderi of a
    built gpu4pyscf Gamma-point GDF object ``gpu_df`` (``gpu_df.build()``, as its SCF does).

    The integrals are not evaluated again: rows of the host cderi (metric applied) go to the
    device a block at a time, are unpacked into a dense AO slab as gpu4pyscf's
    ``loop_gamma_point`` does, and are transformed per function.

    Kwargs:
        max_memory : device bytes the blocks may fill (default: the free device memory).
    Returns:
        A pinned NumPy array (naux, n, n), as :func:`gdf_active_factors`.
    '''
    log = logger.new_logger(gpu_df.cell, verbose)
    t0 = log.init_timer()
    naux = _cderi(gpu_df, 'cderi_active_factors').shape[0]
    C = cp.asarray(mo_act, dtype=np.float64)
    n = C.shape[1]
    out = empty_pinned((naux, n, n))
    # per function besides its cderi row and slab: its factor
    for p0, p1, blk in _cderi_blocks(gpu_df, 8 * n * n, max_memory, 'GDF active factors from cderi',
                                     log):
        _mo_transform(blk, C).get(out=out[p0:p1])
    log.timer('GDF active factors from cderi', *t0)
    return out


def cderi_pair_factors(gpu_df, mo_act, nocc, max_memory=None, verbose=None):
    '''Packed pair factors of the active orbitals ``mo_act`` (nao, nocc + nvir), occupied first,
    from the cderi of a built gpu4pyscf Gamma-point GDF object, on the device, for
    ``pprpa_eri_gpu.pair_operator_from_factors``: the blocks of ``gdf_pair_factors`` (its pair
    orders), filled an aux block at a time with no (naux, n, n) array on the host.
    '''
    log = logger.new_logger(gpu_df.cell, verbose)
    t0 = log.init_timer()
    naux = _cderi(gpu_df, 'cderi_pair_factors').shape[0]
    C = cp.asarray(mo_act, dtype=np.float64)
    n = C.shape[1]
    F = None
    # per function besides its cderi row and slab: its factor and the gathered pairs
    for p0, p1, blk in _cderi_blocks(gpu_df, 16 * n * n, max_memory,
                                     'GDF pair factors from cderi', log):
        part = gdf_pair_factors(_mo_transform(blk, C), nocc)
        if F is None:
            F = {key: cp.empty((Fk.shape[0], naux)) for key, Fk in part.items()}
        for key, Fk in part.items():
            F[key][:, p0:p1] = Fk
        part = None
    log.timer('GDF pair factors from cderi', *t0)
    return F


class TieredCderi:
    '''Gamma-point cderi (naux, npair) with rows [0, ndev) in device memory, [ndev, ndev + nhost)
    in mapped host memory and the rest in a scratch file, one (nfile, p1 - p0) chunk per pair
    column batch [p0, p1).  The file goes through the page cache, which the kernel reclaims under
    memory pressure, so spare host memory speeds its reads up.  ``store[p0:p1]`` returns those rows as a CuPy array on the current
    device, which is all gpu4pyscf's GDF readers ask of ``with_df._cderi[0]``
    (``loop_gamma_point``, ``loop``, ``get_naoaux``).
    '''

    def __init__(self, naux, npair, ndev, nhost, scratch_dir=None):
        if ndev + nhost > naux:
            raise ValueError('more device and host rows than auxiliary functions')
        self.shape = (naux, npair)
        self.dtype = np.dtype(np.float64)
        self.ndim = 2
        self.ndev, self.nhost, self.nfile = ndev, nhost, naux - ndev - nhost
        self.dev = cp.empty((ndev, npair))
        self.host = empty_mapped((nhost, npair)) if nhost else None
        self.batches = []
        self.fd = None
        if self.nfile:
            fd, path = tempfile.mkstemp(prefix='pprpa_cderi_', dir=scratch_dir or lib.param.TMPDIR)
            os.unlink(path)
            self.fd = fd
            weakref.finalize(self, os.close, fd)

    @property
    def nbytes(self):
        return self.shape[0] * self.shape[1] * 8

    def release_device(self):
        '''Free the device rows for the caller's last passes; reading them afterwards raises.'''
        self.dev = None

    def store(self, j3c, p0, p1):
        '''Write the columns [p0, p1) of every row from the device array j3c (naux, p1 - p0).'''
        npair = self.shape[1]
        w = p1 - p0
        j3c = cp.ascontiguousarray(j3c)
        h0, f0 = self.ndev, self.ndev + self.nhost
        stream = cp.cuda.get_current_stream().ptr
        if self.ndev:
            cp.cuda.runtime.memcpy2DAsync(self.dev.data.ptr + p0 * 8, npair * 8, j3c.data.ptr,
                                          w * 8, w * 8, self.ndev, _MEMCPY_DEFAULT, stream)
        if self.nhost:
            cp.cuda.runtime.memcpy2DAsync(self.host.ctypes.data + p0 * 8, npair * 8,
                                          j3c[h0:].data.ptr, w * 8, w * 8, self.nhost,
                                          _MEMCPY_DEFAULT, stream)
        if self.nfile:
            chunk = j3c[f0:].get()
            _pwrite_all(self.fd, chunk, self.nfile * p0 * 8)
            self.batches.append((p0, p1))
        cp.cuda.get_current_stream().synchronize()

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError('TieredCderi supports contiguous row slices only')
        q0, q1 = key.indices(self.shape[0])[:2]
        q1 = max(q0, q1)
        npair = self.shape[1]
        if self.dev is None and q0 < self.ndev:
            raise RuntimeError('the device rows of this cderi were released (release_device)')
        same_device = self.dev is not None and self.dev.device.id == cp.cuda.Device().id
        if q1 <= self.ndev and same_device:
            return self.dev[q0:q1]
        out = cp.empty((q1 - q0, npair))
        h0, f0 = self.ndev, self.ndev + self.nhost
        if q0 < h0:
            r1 = min(q1, h0)
            out[:r1 - q0].data.copy_from_device(self.dev[q0:r1].data, (r1 - q0) * npair * 8)
        if q0 < f0 and q1 > h0:
            r0, r1 = max(q0, h0), min(q1, f0)
            out[r0 - q0:r1 - q0].set(self.host[r0 - h0:r1 - h0])
        if q1 > f0:
            r0 = max(q0, f0)
            nr = q1 - r0
            for p0, p1 in self.batches:
                w = p1 - p0
                buf = np.empty((nr, w))
                off = (self.nfile * p0 + (r0 - f0) * w) * 8
                _pread_all(self.fd, buf, off)
                out[r0 - q0:, p0:p1] = cp.asarray(buf)
        return out


def _pwrite_all(fd, a, off):
    '''os.pwrite of all of ``a`` (one call writes at most ~2 GB on Linux).'''
    mv = memoryview(a).cast('B')
    while mv:
        k = os.pwrite(fd, mv, off)
        mv, off = mv[k:], off + k


def _pread_all(fd, a, off):
    '''os.preadv into all of ``a``.'''
    mv = memoryview(a).cast('B')
    while mv:
        k = os.preadv(fd, [mv], off)
        if k == 0:
            raise EOFError('cderi scratch file is shorter than expected')
        mv, off = mv[k:], off + k


def build_tiered_gdf(gpu_df, device_bytes=0, max_memory=None, scratch_dir=None, host_bytes=None,
                     verbose=None):
    '''Build the Gamma-point cderi of the gpu4pyscf GDF object ``gpu_df`` (what
    ``gpu_df.build()`` does) into a :class:`TieredCderi`: the first ``device_bytes`` of rows on the
    current device, the next rows in mapped host memory as far as ``max_memory`` (MB, default
    ``gpu_df.max_memory``) allows beside what the process already holds and a reserve
    (``_HOST_RESERVE``) for the rest of the run, the remaining rows in a file under ``scratch_dir``
    (default ``lib.param.TMPDIR``).  ``host_bytes`` sets the host share directly instead.  The
    integrals, their order and the attributes set on ``gpu_df`` are those of gpu4pyscf's build.
    3D cells, Gamma point, no range separation.
    '''
    cell = gpu_df.cell
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    if cell.dimension != 3 or not gpu_df.is_gamma_point or not is_zero(gpu_df.kpts):
        raise NotImplementedError('build_tiered_gdf supports 3D Gamma-point cells only')
    if cell.omega != 0:
        raise NotImplementedError('build_tiered_gdf does not support range-separated cells')
    gpu_df.check_sanity()
    auxcell = make_auxcell(cell, gpu_df.auxbasis, gpu_df.exp_to_discard)
    gpu_df.auxcell = auxcell
    gpu_df.nao = cell.nao
    gpu_df._omega = cell.omega
    gpu_df.kmesh = [1] * 3

    opt, rsdf_omega, aux_coeff = _rsdf_setup(cell, auxcell, gpu_df.linear_dep_threshold)
    scell = opt.cell
    naux_cart, naux = aux_coeff.shape
    mapping, diag = opt.pair_and_diag_indices()
    npair = len(mapping)
    row_bytes = 8 * npair
    ndev = min(naux, int(device_bytes) // row_bytes)
    if host_bytes is None:
        max_memory = gpu_df.max_memory if max_memory is None else max_memory
        host_bytes = max(0, (max_memory * (1 - _HOST_RESERVE) - lib.current_memory()[0]) * 1e6)
    nhost = min(naux - ndev, int(host_bytes) // row_bytes)
    store = TieredCderi(naux, npair, ndev, nhost, scratch_dir)
    log.info('GDF cderi naux=%d nao_pairs=%d: %d rows on the device (%.1f GB), %d in host '
             'memory (%.1f GB), %d in a scratch file (%.1f GB)', naux, npair,
             ndev, ndev * row_bytes * 1e-9, nhost, nhost * row_bytes * 1e-9,
             store.nfile, store.nfile * row_bytes * 1e-9)

    mesh = opt.mesh
    Gv, auxG = _lr_aux(opt, rsdf_omega)
    ngrids = len(Gv)
    auxG = aux_coeff.T.dot(auxG)
    budget = plan_budget(free_bytes())
    # per AO pair of a batch: the int3c2e output (complex, as gpu4pyscf's evaluator writes it),
    # the fitted rows and the slack of gpu4pyscf's sizing (naux_cart * 16 * 4 per pair)
    batch = min(npair, budget // (naux_cart * 16 * 4))
    if batch < 1:
        raise MemoryError('tiered GDF build: no room for one AO pair')
    agg = scell.aggregate_shl_pairs(opt.bas_ij_cache, ft_ao.ft_ao_scheme()[0])
    eval_j3c, _, pair_offsets = opt.int3c2e_evaluator(
        ao_pair_batch_size=batch, bas_ij_aggregated=agg)[:3]
    eval_ft = _ft_evaluator(opt, batch, agg, pair_offsets)
    gblk = (budget - naux_cart * batch * 32) // (16 * (batch + naux * 2)) // _G_ALIGN * _G_ALIGN
    if gblk < _G_ALIGN:
        raise MemoryError(f'tiered GDF build: no room for {_G_ALIGN} G vectors')
    gblk = min(gblk, ngrids)
    log.debug('tiered GDF build: %d pair batches of <= %d, LR mesh %s in blocks of %d G',
              len(pair_offsets) - 1, batch, list(mesh), gblk)
    buf0 = cp.empty(naux * batch)
    buf1 = cp.empty(batch * naux_cart, dtype=np.complex128)
    buf2 = cp.empty(batch * gblk, dtype=np.complex128)
    for b in range(len(pair_offsets) - 1):
        j3c = eval_j3c(shl_pair_batch_id=b, out=buf1)
        if j3c.size == 0:
            continue
        w = j3c.shape[0]
        fit = cp.ndarray((naux, w), memptr=buf0.data)
        aux_coeff.T.dot(j3c.sum(axis=1).T, out=fit)
        tmp = cp.ndarray((naux, w), dtype=np.complex128, memptr=buf1.data)
        for G0, G1 in lib.prange(0, ngrids, gblk):
            pqG = eval_ft(Gv[G0:G1], b, out=buf2)
            fit += auxG[:, G0:G1].dot(pqG.T, out=tmp).real
        store.store(fit, int(pair_offsets[b]), int(pair_offsets[b + 1]))
    buf0 = buf1 = buf2 = auxG = None
    gpu_df._cderi = {0: store}
    gpu_df._cderip = None
    gpu_df._cderi_idx = (cp.asarray(mapping), cp.asarray(diag))
    log.timer('tiered GDF build', *t0)
    return gpu_df


def _scatter_pairs(slab, src, rows, cols, pair_major=True):
    '''Write the AO pairs of ``src`` into slab[k], both triangles: src (npair, ..., nk) pair-major
    (an int3c2e or Fourier batch) or (nk, npair) function-major (cderi rows).'''
    src = cp.ascontiguousarray(src)
    npair = rows.size
    nk = src.size // npair
    sp, sk = (nk, 1) if pair_major else (1, npair)
    _SCATTER(src, rows, cols, npair, sp, sk, slab.shape[-1], slab, size=src.size)


def _rsdf_setup(cell, auxcell, linear_dep_threshold):
    '''(opt, rsdf_omega, aux_coeff) of gpu4pyscf's Gamma-point GDF build: the short-range
    int3c2e optimiser, its range-separation omega and the metric's aux coefficients
    (naux_cart, naux), as ``rsdf_builder.compressed_cderi_j_only`` sets them up.'''
    rsdf_omega = _guess_omega(cell)
    opt = SRInt3c2eOpt(cell, auxcell, omega=-rsdf_omega, bvk_kmesh=np.ones(3, dtype=int)).build()
    aux_coeff = cp.asarray(_precontract_j2c_aux_coeff(opt.auxcell, None, 0, rsdf_omega,
                                                      linear_dep_threshold)[0][0])
    return opt, rsdf_omega, aux_coeff


def _lr_aux(opt, rsdf_omega):
    '''(Gv, auxG): the G vectors of the rsdf mesh and the raw aux functions' Fourier
    transforms weighted by the long-range kernel, conj(FT[P](G)) coulG(G) (naux_cart, nG).'''
    Gv = opt.cell.get_Gv(opt.mesh)
    auxG = ft_ao.ft_ao(opt.auxcell, Gv).T.conj()
    auxG *= cp.asarray(_weighted_coulG_kpts(opt.auxcell, opt.mesh, 0, rsdf_omega)).ravel()
    return Gv, auxG


def _ft_evaluator(opt, pair_blk, agg, pair_offsets):
    '''The AO-pair Fourier evaluator in the pair batches of the int3c2e evaluator.'''
    eval_ft, ft_offsets = ft_ao.FTOpt.from_intopt(opt).ft_evaluator(
        pair_blk, bas_ij_aggregated=agg)
    if not np.array_equal(np.asarray(pair_offsets), np.asarray(ft_offsets)):
        raise RuntimeError('int3c2e and FT pair batches differ')
    return eval_ft


def _cderi(gpu_df, what):
    '''The compressed Gamma-point cderi (naux, nao_pairs) of the built gpu4pyscf GDF ``gpu_df``.'''
    if gpu_df._cderi is None:
        raise ValueError('gpu_df holds no cderi (gpu_df.build())')
    if gpu_df.cell.dimension != 3 or not is_zero(gpu_df.kpts):
        raise NotImplementedError(f'{what} supports 3D Gamma-point cells only')
    return gpu_df._cderi[0]


def _cderi_blocks(gpu_df, per_aux, max_memory, what, log):
    '''(p0, p1, slab) over the cderi of ``gpu_df``: its rows [p0, p1) unpacked into dense AO
    slabs slab[k] (nao, nao) on the device, as gpu4pyscf's ``loop_gamma_point`` does.  A block
    has as many functions as fit ``max_memory`` (default: the free device memory) with their
    rows, their slabs and ``per_aux`` more bytes each; the slab buffer is reused.'''
    cderi = _cderi(gpu_df, what)
    naux, npair = cderi.shape
    nao = gpu_df.cell.nao
    rows, cols = cp.divmod(cp.asarray(gpu_df._cderi_idx[0], dtype=np.int64), nao)
    budget = plan_budget(free_bytes()) if max_memory is None else int(max_memory)
    aux_blk = min(naux, budget // (8 * (npair + nao * nao) + per_aux))
    if aux_blk < 1:
        raise MemoryError(f'{what}: no room for one aux function')
    log.debug('%s: naux=%d nao_pairs=%d, blocks of %d', what, naux, npair, aux_blk)
    slab = cp.empty((aux_blk, nao, nao))
    for p0, p1 in lib.prange(0, naux, aux_blk):
        blk = slab[:p1 - p0]
        blk.fill(0)
        _scatter_pairs(blk, cp.asarray(cderi[p0:p1]), rows, cols, pair_major=False)
        yield p0, p1, blk


def _mo_transform(slabs, C):
    '''C^T slabs[k] C (nk, n, n) on the device.'''
    out = cp.empty((len(slabs), C.shape[1], C.shape[1]))
    for k in range(len(slabs)):
        out[k] = C.T @ (slabs[k] @ C)
    return out


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
