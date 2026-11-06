'''Gamma-point exact exchange on the uniform grid from the factors of a density, in bounded memory.

For D = sum_k lam_k u_k u_k^T and ket orbitals v_i (columns of AO coefficient matrices),

    W_i(g) = sum_k lam_k u_k(g) V[u_k v_i](g),      V[rho] = ifft(w fft(rho)),

K[D] v_i = (vol / ngrid) sum_g ao(g) W_i(g) (K_pq = sum_jk (pj|kq) D_jk, gpu4pyscf's fft_jk.get_k)
and the exchange force of D is the same W against the gradient AOs.  One codensity FFT per
(k, i) pair, real-to-complex with ``gpu_coulomb``'s symmetrised kernel.  A block of ket grids
and its W stay on the device while the factor grids are evaluated block by block, so memory is
set by the ket block, not by the rank of D.  ``factor`` gives (lam, U) of a symmetric density
from its coefficients over a basis (``projector``).  NumPy or CuPy coefficients; one device.

A CPHF density D = c o^T + o c^T (o the occupied orbitals, c = C_v x) on the occupied kets:
the c o^T half has the symmetric codensities o_j o_i, taken once, the o c^T half the c_j o_i,
1.5 n^2 pairs for the 2n x n of D's factors (``exchange_cross_ket``).
'''

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.lib import logger

from lib_pprpa.gpu_coulomb import ao_loop, codensity, coulomb_potential, half_kernel, mo_on_grid
from lib_pprpa.gpu_mem import free_bytes, max_fft_batch, plan_budget

__all__ = ['factor', 'projector', 'exchange_ket', 'exchange_cross_ket', 'exchange_force']

# Bytes per (codensity, grid point) of one coulomb_potential batch: 56.4 (codensities, half
# spectrum, real output, two cuFFT work areas, measured on the 159^3 mesh, see gpu_ao2mo).
_FFT_BYTES = 64
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


def _accumulate(W, V, S, scale, target, pair, src):
    '''W[target_k] += scale[src_k] S[src_k] V[pair_k] for the contributions k of one FFT batch
    (NumPy integer arrays; V the batch's potentials, S the source grids): one pass over V,
    each target row of W read and written once, no temporaries of the grid's size.'''
    order = np.argsort(target, kind='stable')
    rows, start = np.unique(target[order], return_index=True)
    ptr = np.append(start, len(order))
    rows, ptr, pair, src = (cp.asarray(a, dtype=np.int32)
                            for a in (rows, ptr, pair[order], src[order]))
    ngrid = V.shape[1]
    _ACCUMULATE(V, S, scale, rows, ptr, pair, src, ngrid, W, size=len(rows) * ngrid)


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
    pb = (per_point - 16 * kb - 8 * ub) // _FFT_BYTES
    if pb < 1:
        raise MemoryError(f'exchange: {per_point} bytes per grid point leave no FFT batch')
    return kb, ub, max(1, min(pb, max_fft_batch(ngrid)))


def _grids(cell, mesh, lam, U, ket_grid, ub, pb, w_half):
    '''W (nket, ngrid) for the ket grids ``ket_grid`` (nket, ngrid), the factor grids
    evaluated ``ub`` columns of U at a time, ``pb`` codensities per FFT batch.'''
    nket = ket_grid.shape[0]
    W = cp.zeros_like(ket_grid)
    for k0, k1 in lib.prange(0, U.shape[1], ub):
        u, = mo_on_grid(cell, [U[:, k0:k1]], mesh)
        lu = lam[k0:k1]
        nk = k1 - k0
        # pairs (i, k) ket-major, P = i * nk + k: a batch adds into a few rows of W
        iidx = cp.repeat(cp.arange(nket, dtype=np.int32), nk)
        kidx = cp.tile(cp.arange(nk, dtype=np.int32), nket)
        for p0, p1 in lib.prange(0, nk * nket, pb):
            V = coulomb_potential(codensity(u, ket_grid, kidx, iidx, p0, p1), mesh, w_half)
            P = np.arange(p0, p1)
            _accumulate(W, V, u, lu, P // nk, P - p0, P % nk)     # lam_k u_k V[u_k v_i]
            V = None
        u = None
    return W


def exchange_ket(cell, mesh, lam, U, ket, coulG, verbose=None):
    '''K[U diag(lam) U^T] @ ket, CuPy (nao, nket), for the kernel ``coulG`` (``get_coulG``'s
    normalisation).'''
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    mesh = np.asarray(mesh, dtype=int)
    ngrid = int(np.prod(mesh))
    U, ket, lam = (cp.asarray(a, dtype=np.float64) for a in (U, ket, lam))
    w_half = half_kernel(coulG, mesh)
    nket = ket.shape[1]
    kb, ub, pb = _plan(cell, mesh, nket, U.shape[1])
    log.debug('exchange: rank %d, ket %d in blocks of %d, factor blocks %d, %d pairs per FFT',
              U.shape[1], nket, kb, ub, pb)
    K = cp.empty((cell.nao, nket))
    for i0, i1 in lib.prange(0, nket, kb):
        ket_grid, = mo_on_grid(cell, [ket[:, i0:i1]], mesh)
        W = _grids(cell, mesh, lam, U, ket_grid, ub, pb, w_half)
        ket_grid = None
        Kb = cp.zeros((cell.nao, i1 - i0))
        for g0, g1, aoT in ao_loop(cell, mesh, extra_per_point=8 * (i1 - i0)):
            Kb += aoT @ W[:, g0:g1].T
        # the loop variable keeps the last AO chunk, as large as the plan allowed, through
        # the next ket block, whose budget was planned without it
        aoT = None
        K[:, i0:i1] = Kb * (cell.vol / ngrid)
        W = None
    log.timer(f'exchange: {U.shape[1] * nket} codensity pairs', *t0)
    return K


def exchange_cross_ket(cell, mesh, ket, cross, coulG, verbose=None):
    '''K[cross ket^T + ket cross^T] @ ket, CuPy (nao, n), for ket and cross (nao, n) and the
    kernel ``coulG`` (``get_coulG``'s normalisation): a CPHF density C_v x C_o^T + h.c. on the
    occupied orbitals (ket C_o, cross C_v x).  W_i = sum_j c_j V[o_j o_i] + o_j V[c_j o_i]
    with both orbital sets on the grid once, the o_j o_i pairs j >= i once, one AO projection.
    Holds the two orbital grids and W, unblocked: 24 n bytes per grid point.'''
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    mesh = np.asarray(mesh, dtype=int)
    ngrid = int(np.prod(mesh))
    n = ket.shape[1]
    per_point = plan_budget(free_bytes()) // ngrid - 24 * n
    pb = per_point // _FFT_BYTES
    if pb < 1:
        raise MemoryError(f'cross exchange: {per_point} bytes per grid point leave no FFT batch')
    pb = min(pb, max_fft_batch(ngrid))
    log.debug('cross exchange: %d kets, %d pairs per FFT', n, pb)
    w_half = half_kernel(coulG, mesh)
    o, c = mo_on_grid(cell, [cp.asarray(ket, dtype=np.float64),
                             cp.asarray(cross, dtype=np.float64)], mesh)
    W = cp.zeros_like(o)
    ones = cp.ones(n)
    # c o^T: pairs (j, i <= j) row-major; W_i += c_j V_ji and, off the diagonal, W_j += c_i V_ji
    jj, ii = np.tril_indices(n)
    jidx, iidx = cp.asarray(jj, dtype=np.int32), cp.asarray(ii, dtype=np.int32)
    for p0, p1 in lib.prange(0, len(jj), pb):
        V = coulomb_potential(codensity(o, o, jidx, iidx, p0, p1), mesh, w_half)
        j, i = jj[p0:p1], ii[p0:p1]
        P = np.arange(p1 - p0)
        off = i < j
        _accumulate(W, V, c, ones, np.concatenate([i, j[off]]), np.concatenate([P, P[off]]),
                    np.concatenate([j, i[off]]))
        V = None
    # o c^T: pairs (i, j) ket-major, P = i * n + j; W_i += o_j V[c_j o_i]
    iidx = cp.repeat(cp.arange(n, dtype=np.int32), n)
    jidx = cp.tile(cp.arange(n, dtype=np.int32), n)
    for p0, p1 in lib.prange(0, n * n, pb):
        V = coulomb_potential(codensity(c, o, jidx, iidx, p0, p1), mesh, w_half)
        P = np.arange(p0, p1)
        _accumulate(W, V, o, ones, P // n, P - p0, P % n)
        V = None
    o = c = jidx = iidx = None
    K = cp.zeros((cell.nao, n))
    for g0, g1, aoT in ao_loop(cell, mesh, extra_per_point=8 * n):
        K += aoT @ W[:, g0:g1].T
    aoT = W = None
    log.timer(f'cross exchange: {n * (n + 1) // 2 + n * n} codensity pairs', *t0)
    return K * (cell.vol / ngrid)


def exchange_force(cell, mesh, lam, U, ket, L, coulG, verbose=None):
    '''-2 sum_{i in A} sum_m L_im G[x, i, m], NumPy (natm, 3), with
    G[x, i, m] = (vol/ngrid) sum_g d_x ao_i(g) W_m(g) and W of D = U diag(lam) U^T on the ket
    columns m: the exchange force of ``gpu_pairing_force`` with W from D and L = C M from
    another density (L (nao, nket) matches the ket columns).  dE/dR convention.'''
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    mesh = np.asarray(mesh, dtype=int)
    ngrid = int(np.prod(mesh))
    U, ket, lam, L = (cp.asarray(a, dtype=np.float64) for a in (U, ket, lam, L))
    w_half = half_kernel(coulG, mesh)
    nket = ket.shape[1]
    kb, ub, pb = _plan(cell, mesh, nket, U.shape[1])
    t = cp.zeros((3, cell.nao))
    for i0, i1 in lib.prange(0, nket, kb):
        ket_grid, = mo_on_grid(cell, [ket[:, i0:i1]], mesh)
        W = _grids(cell, mesh, lam, U, ket_grid, ub, pb, w_half)
        ket_grid = None
        G = cp.zeros((3, cell.nao, i1 - i0))
        for g0, g1, ao in ao_loop(cell, mesh, deriv=1, extra_per_point=8 * (i1 - i0)):
            Wc = cp.ascontiguousarray(W[:, g0:g1])
            for x in range(3):
                G[x] += ao[1 + x].T @ Wc.T
        ao = Wc = W = None             # the last AO chunk too (see exchange_ket)
        t += cp.einsum('xim,im->xi', G, L[:, i0:i1]) * (cell.vol / ngrid)
    t = t.get()
    de = np.zeros((cell.natm, 3))
    for ia, (_, _, p0, p1) in enumerate(cell.aoslice_by_atom()):
        de[ia] = -2.0 * t[:, p0:p1].sum(axis=1)
    log.timer(f'exchange force: {U.shape[1] * nket} codensity pairs', *t0)
    return de
