'''Gamma-point exact exchange on the uniform grid, W_i = sum_k lam_k u_k V[u_k v_i] for
D = U diag(lam) U^T and kets v_i (V[rho] = ifft(w fft(rho))): K[D] v_i = (vol/ngrid) ao^T W_i,
the exchange force the same W against the gradient AOs.  One real-to-complex codensity FFT per
pair, by ``gpu_fft_chain`` on 7-smooth meshes, else cuFFT; ket grids and W stay on the device
while the factor grids pass in blocks.  On the occupied kets o, D's blocks over the MOs
(o M_oo o^T + c o^T + o c^T + D_vv, c = C_v M_vo) need the pairs o_j o_i once (W_i += s_j
V[o_j o_i], s = c + o M_oo), n^2 pairs c_j o_i (W_i += o_j V[c_j o_i]) and D_vv's factors:
``exchange_occ_ket`` takes these or D's factors, whichever has fewer pairs;
``exchange_cross_ket`` a CPHF density (1.5 n^2 pairs, not 2 n^2).  With ``L`` the ket and
occupied passes give the exchange force instead of K C.
'''

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.lib import logger

from lib_pprpa import gpu_fft_chain
from lib_pprpa.gpu_coulomb import (ao_force, ao_project, codensity, coulomb_potential,
                                   half_kernel, mo_on_grid)
from lib_pprpa.gpu_mem import EXCHANGE_FFT_BYTES, free_bytes, max_fft_batch, plan_budget

__all__ = ['factor', 'projector', 'exchange_ket', 'exchange_cross_ket', 'exchange_occ_ket',
           'reference_exchange_force']

# Shares of the per-grid-point budget: ket grids + their W, factor grids; the rest is the FFT batch.
_KET_SHARE = 0.3
_FACTOR_SHARE = 0.4
# Eigenvalues of a factored density below this fraction of the largest are dropped.
_RANK_TOL = 1e-13

# W[rows[t]] += sum_k scale[src[k]] S[src[k]] V[pair[k]] over k in [ptr[t], ptr[t + 1]): one
# thread per (target row, grid point), so each element of W is read and written once per batch.
_ACCUMULATE = cp.ElementwiseKernel(
    'raw float64 V, raw float64 S, raw float64 scale, raw int32 rows, raw int32 ptr, '
    'raw int32 pair, raw int32 src, int64 ngrid',
    'raw float64 W',
    '''
    const long long t = i / ngrid;
    const long long g = i - t * ngrid;
    double acc = 0.0;
    for (int k = ptr[t]; k < ptr[t + 1]; ++k)
        acc += scale[src[k]] * S[(long long)src[k] * ngrid + g] * V[(long long)pair[k] * ngrid + g];
    W[(long long)rows[t] * ngrid + g] += acc;
    ''',
    'pprpa_exchange_accumulate')


def _csr(target, pair, src):
    '''(rows, ptr, pair, src), int32 device arrays: the contributions k grouped by target row,
    those of rows[t] at [ptr[t], ptr[t + 1]).'''
    order = np.argsort(target, kind='stable')
    rows, start = np.unique(target[order], return_index=True)
    ptr = np.append(start, len(order))
    return tuple(cp.asarray(a, dtype=np.int32) for a in (rows, ptr, pair[order], src[order]))


def _accumulate(W, V, S, scale, target, pair, src):
    '''W[target_k] += scale[src_k] S[src_k] V[pair_k] for the contributions k of one FFT batch
    (NumPy integer arrays; V the batch's potentials, S the source grids): one pass over V,
    each target row of W read and written once, no temporaries of the grid's size.'''
    rows, ptr, pair, src = _csr(target, pair, src)
    ngrid = V.shape[1]
    _ACCUMULATE(V, S, scale, rows, ptr, pair, src, ngrid, W, size=len(rows) * ngrid)


class _PotentialSum:
    '''W[target_k] += scale[src_k] S[src_k] V[A[a_P] B[b_P]], P = p0 + pair_k, for one FFT batch
    of at most ``pb`` codensities: gpu_fft_chain on a mesh it applies to, else codensity,
    cuFFT and _accumulate.'''

    def __init__(self, mesh, w_half, pb):
        self.mesh, self.w_half, self.pb = mesh, w_half, pb
        self.chain = (gpu_fft_chain.FFTChain(mesh, w_half, pb)
                      if gpu_fft_chain.applies(mesh) else None)

    def add(self, W, A, B, aidx, bidx, p0, p1, S, scale, target, pair, src):
        if self.chain is None:
            V = coulomb_potential(codensity(A, B, aidx, bidx, p0, p1), self.mesh, self.w_half)
            _accumulate(W, V, S, scale, target, pair, src)
        else:
            self.chain.add(W, A, B, aidx, bidx, p0, p1, S, scale, *_csr(target, pair, src))


def factor(M, C):
    '''(lam, U), CuPy, with C M C^T = U diag(lam) U^T for a symmetric M over the columns of C:
    U = C E, E the eigenvectors of M, |lam| below _RANK_TOL of the largest dropped.'''
    lam, e = cp.linalg.eigh(cp.asarray(M))
    keep = cp.abs(lam) > _RANK_TOL * float(cp.abs(lam).max())
    return lam[keep], cp.asarray(C) @ e[:, keep]


def projector(C, ovlp):
    '''(C^T S C)^-1 C^T S, CuPy: C @ it is the S-projector onto the span of C, so a density's
    coefficients over C are proj dm proj^T, exact whether or not C is S-orthonormal (SCF MOs
    are only to ~1e-8 in an ill-conditioned basis).  Zero columns of C get zero rows: gpu4pyscf
    pads the MOs of an overlap with eigenvalues it removed (below 1e-6) with zero columns.'''
    C, ovlp = cp.asarray(C), cp.asarray(ovlp)
    live = cp.abs(C).max(axis=0) > 0
    cs = C[:, live].T @ ovlp
    proj = cp.zeros((C.shape[1], C.shape[0]))
    proj[live] = cp.linalg.solve(cs @ C[:, live], cs)
    return proj


def _plan(cell, mesh, nket, rank):
    '''Ket block, factor block and FFT batch sizes from the free device memory.'''
    ngrid = int(np.prod(mesh))
    per_point = plan_budget(free_bytes()) // ngrid
    kb = max(1, min(nket, int(_KET_SHARE * per_point) // 16))     # ket grid + W
    ub = max(1, min(rank, int(_FACTOR_SHARE * per_point) // 8))
    return kb, ub, _fft_batch(ngrid, 16 * kb + 8 * ub)


def _fft_batch(ngrid, held=0):
    '''Codensities per FFT batch from the free device memory, beside ``held`` more bytes per
    grid point still to be allocated.'''
    pb = (plan_budget(free_bytes()) // ngrid - held) // EXCHANGE_FFT_BYTES
    if pb < 1:
        raise MemoryError(f'exchange: no FFT batch fits beside {held} bytes per grid point')
    return min(pb, max_fft_batch(ngrid))


def _grids(cell, mesh, lam, U, ket_grid, W, ub, pb, w_half):
    '''W (nket, ngrid) += the factor terms of U diag(lam) U^T on the ket grids ``ket_grid``,
    the factor grids evaluated ``ub`` columns of U at a time, ``pb`` codensities per FFT.'''
    nket = ket_grid.shape[0]
    sums = _PotentialSum(mesh, w_half, pb)
    for k0, k1 in lib.prange(0, U.shape[1], ub):
        u, = mo_on_grid(cell, [U[:, k0:k1]], mesh)
        lu = lam[k0:k1]
        nk = k1 - k0
        # pairs (i, k) ket-major, P = i * nk + k: a batch adds into a few rows of W
        iidx = cp.repeat(cp.arange(nket, dtype=np.int32), nk)
        kidx = cp.tile(cp.arange(nk, dtype=np.int32), nket)
        for p0, p1 in lib.prange(0, nk * nket, pb):
            P = np.arange(p0, p1)
            # lam_k u_k V[u_k v_i]
            sums.add(W, u, ket_grid, kidx, iidx, p0, p1, u, lu, P // nk, P - p0, P % nk)
        u = None
    return W


def _diagonal(oo, cross):
    '''True for D's occupied block alone and diagonal (an SCF density): its s grids are o's.'''
    return (cross is None and oo is not None
            and not bool(cp.any(cp.asarray(oo) - cp.diag(cp.diagonal(cp.asarray(oo))))))


def _occ_grids(cell, mesh, o, w_half, oo=None, cross=None, lam=None, U=None, log=None):
    '''W (n, ngrid) on the kets o (nao, n) of D = o oo o^T + cross o^T + o cross^T
    + U diag(lam) U^T, the parts given: the pairs o_j o_i (j >= i) once with s = cross + o oo,
    the n^2 pairs c_j o_i, the factor pairs u_k o_i.  Holds o, W and s or c on the grid,
    unblocked: 24 n bytes per grid point, 16 n for a diagonal oo alone (s_j = oo_jj o_j).'''
    n = o.shape[1]
    ngrid = int(np.prod(mesh))
    ones = cp.ones(n)
    paired = oo is not None or cross is not None
    sg, scale = None, ones
    if _diagonal(oo, cross):
        og, = mo_on_grid(cell, [o], mesh)
        sg, scale = og, cp.ascontiguousarray(cp.diagonal(cp.asarray(oo)))
    elif paired:
        s = cross if oo is None else (o @ oo if cross is None else cross + o @ oo)
        og, sg = mo_on_grid(cell, [o, s], mesh)
        s = None
    else:
        og, = mo_on_grid(cell, [o], mesh)
    W = cp.zeros_like(og)
    if paired:
        # pairs (j, i <= j) row-major; W_i += s_j V_ji and, off the diagonal, W_j += s_i V_ji
        sums = _PotentialSum(mesh, w_half, _fft_batch(ngrid))
        jj, ii = np.tril_indices(n)
        jidx, iidx = (cp.asarray(a, dtype=np.int32) for a in (jj, ii))
        if log is not None:
            log.debug('exchange: %d o o pairs, %d per FFT', len(jj), sums.pb)
        for p0, p1 in lib.prange(0, len(jj), sums.pb):
            j, i = jj[p0:p1], ii[p0:p1]
            P = np.arange(p1 - p0)
            off = i < j
            sums.add(W, og, og, jidx, iidx, p0, p1, sg, scale, np.concatenate([i, j[off]]),
                     np.concatenate([P, P[off]]), np.concatenate([j, i[off]]))
        sums = jidx = iidx = None
    if cross is not None:
        if oo is not None:                 # sg held s, not the cross grids
            sg = None
            sg, = mo_on_grid(cell, [cross], mesh)
        # pairs (i, j) ket-major, P = i * n + j; W_i += o_j V[c_j o_i]
        sums = _PotentialSum(mesh, w_half, _fft_batch(ngrid))
        iidx = cp.repeat(cp.arange(n, dtype=np.int32), n)
        jidx = cp.tile(cp.arange(n, dtype=np.int32), n)
        if log is not None:
            log.debug('exchange: %d c o pairs, %d per FFT', n * n, sums.pb)
        for p0, p1 in lib.prange(0, n * n, sums.pb):
            P = np.arange(p0, p1)
            sums.add(W, sg, og, jidx, iidx, p0, p1, og, ones, P // n, P - p0, P % n)
        sums = jidx = iidx = None
    sg = None
    if U is not None and U.shape[1]:
        per_point = plan_budget(free_bytes()) // ngrid
        # og and W are held: the factor grids get _plan's factor share of the rest
        ub = int(_FACTOR_SHARE / (1 - _KET_SHARE) * per_point) // 8
        ub = max(1, min(U.shape[1], ub))
        pb = _fft_batch(ngrid, 8 * ub)
        if log is not None:
            log.debug('exchange: %d factor pairs, factor blocks %d, %d per FFT',
                      U.shape[1] * n, ub, pb)
        _grids(cell, mesh, lam, U, og, W, ub, pb, w_half)
    return W


def _occ_parts(M, mo, occ):
    '''(keyword arguments of _occ_grids, pairs) for D = mo M mo^T on the kets mo[:, occ]:
    D's blocks (oo, cross, the virtual block's factors) or D's factors whole, whichever has
    fewer pairs.  A block below _RANK_TOL of M's largest element is left out.'''
    M, mo = (cp.asarray(a, dtype=np.float64) for a in (M, mo))
    occ = np.asarray(occ, dtype=bool)
    o, v = (cp.asarray(np.flatnonzero(m)) for m in (occ, ~occ))
    n = len(o)
    lam, U = factor(M, mo)
    tiny = _RANK_TOL * float(cp.abs(M).max())
    m_oo, m_vo = M[o][:, o], M[v][:, o]
    oo = m_oo if float(cp.abs(m_oo).max()) > tiny else None
    cross = mo[:, v] @ m_vo if len(v) and float(cp.abs(m_vo).max()) > tiny else None
    lam_v, U_v = factor(M[v][:, v], mo[:, v]) if len(v) else (None, None)
    pairs = ((n * (n + 1) // 2 if oo is not None or cross is not None else 0)
             + (n * n if cross is not None else 0) + (0 if U_v is None else U_v.shape[1] * n))
    if len(lam) * n <= pairs:
        return dict(lam=lam, U=U), len(lam) * n
    return dict(oo=oo, cross=cross, lam=lam_v, U=U_v), pairs


def exchange_ket(cell, mesh, lam, U, ket, coulG, L=None, verbose=None):
    '''K[U diag(lam) U^T] @ ket, CuPy (nao, nket), for the kernel ``coulG`` (``get_coulG``'s
    normalisation); with L (nao, nket) the exchange force instead, NumPy (natm, 3), dE/dR:
    -2 sum_{i in A} sum_m L_im (vol/ngrid) sum_g d_x ao_i(g) W_m(g), W of D on the ket columns
    m (the force of ``gpu_pairing_force`` with W from D and L = C M from another density).
    The ket columns go in blocks sized to the free memory (_plan).'''
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    what = 'exchange' if L is None else 'exchange force'
    mesh = np.asarray(mesh, dtype=int)
    U, ket, lam = (cp.asarray(a, dtype=np.float64) for a in (U, ket, lam))
    w_half = half_kernel(coulG, mesh)
    nket = ket.shape[1]
    kb, ub, pb = _plan(cell, mesh, nket, U.shape[1])
    log.debug('%s: rank %d, ket %d in blocks of %d, factor blocks %d, %d pairs per FFT%s', what,
              U.shape[1], nket, kb, ub, pb,
              ' (five-pass FFT)' if gpu_fft_chain.applies(mesh) else '')
    out = cp.empty((cell.nao, nket)) if L is None else np.zeros((cell.natm, 3))
    for i0, i1 in lib.prange(0, nket, kb):
        ket_grid, = mo_on_grid(cell, [ket[:, i0:i1]], mesh)
        W = _grids(cell, mesh, lam, U, ket_grid, cp.zeros_like(ket_grid), ub, pb, w_half)
        ket_grid = None
        if L is None:
            out[:, i0:i1] = ao_project(cell, mesh, W)
        else:
            out += ao_force(cell, mesh, W, L[:, i0:i1])
        W = None
    log.timer(f'{what}: {U.shape[1] * nket} codensity pairs', *t0)
    return out


def exchange_cross_ket(cell, mesh, ket, cross, coulG, verbose=None):
    '''K[cross ket^T + ket cross^T] @ ket, CuPy (nao, n), for ket and cross (nao, n) and the
    kernel ``coulG`` (``get_coulG``'s normalisation): a CPHF density C_v x C_o^T + h.c. on the
    occupied orbitals (ket C_o, cross C_v x), 1.5 n^2 pairs (_occ_grids).'''
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    mesh = np.asarray(mesh, dtype=int)
    n = ket.shape[1]
    ket, cross = (cp.asarray(a, dtype=np.float64) for a in (ket, cross))
    W = _occ_grids(cell, mesh, ket, half_kernel(coulG, mesh), cross=cross, log=log)
    K = ao_project(cell, mesh, W)
    log.timer(f'cross exchange: {n * (n + 1) // 2 + n * n} codensity pairs', *t0)
    return K


def exchange_occ_ket(cell, mesh, M, mo, occ, coulG, L=None, verbose=None):
    '''K[mo M mo^T] @ mo[:, occ], CuPy (nao, n), for a symmetric M (nmo, nmo) over the MOs
    ``mo`` and their occupied mask ``occ`` (NumPy bool); with L (nao, n) the exchange force
    against it instead, as ``exchange_ket``.  D's blocks or its factors, whichever has fewer
    pairs (_occ_parts).'''
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    what = 'exchange' if L is None else 'exchange force'
    mesh = np.asarray(mesh, dtype=int)
    parts, pairs = _occ_parts(M, mo, occ)
    o = cp.asarray(mo, dtype=np.float64)[:, cp.asarray(np.flatnonzero(occ))]
    W = _occ_grids(cell, mesh, o, half_kernel(coulG, mesh), log=log, **parts)
    out = ao_project(cell, mesh, W) if L is None else ao_force(cell, mesh, W, L)
    log.timer(f'{what} on the occupied kets: {pairs} codensity pairs', *t0)
    return out


def reference_exchange_force(cell, mesh, mo, mo_occ, coulG, P=None, verbose=None):
    '''dE/dR, NumPy (natm, 3), of the exchange energy -1/4 tr(D K[D]) of the closed-shell
    reference D = mo diag(mo_occ) mo^T (``coulG`` with the hybrid's factors); with the relaxed
    density P (MO basis) the polarisation identity's Q(D + P) - Q(P) instead.  Q = -F/2 with
    F(M) = F_b(M, M), F_b(A, B) the force of A's W against B's L, so the difference is
    -1/2 [F_b(D, D + P) + F_b(P, D)]: W of D on the virtual MOs from its factors and on the
    occupied ones from the pairs o_j o_i once, W of P on the occupied ones.'''
    mo, mo_occ = np.asarray(mo, dtype=np.float64), np.asarray(mo_occ, dtype=np.float64)
    occ = mo_occ > 0
    D = np.diag(mo_occ)
    L = mo @ (D if P is None else D + P)
    de = exchange_occ_ket(cell, mesh, D, mo, occ, coulG, L=L[:, occ], verbose=verbose)
    if P is not None:
        n, cocc = mo_occ[occ], mo[:, occ]
        de += exchange_ket(cell, mesh, n, cocc, mo[:, ~occ], coulG, L=L[:, ~occ], verbose=verbose)
        de += exchange_occ_ket(cell, mesh, P, mo, occ, coulG, L=cocc * n, verbose=verbose)
    return -0.5 * de
