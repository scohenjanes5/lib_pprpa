'''Pair-density exchange and pairing-exchange force of the Gamma-point pp-RPA gradient for
X = C M C^T on the CPU (the NumPy counterpart of ``gpu_pairing_force``).

C holds the active orbitals and M = diag(y, x), so X has rank r <= nocc + nvir.  With the
orbitals on the uniform grid, one FFT pass over the r (r + 1) / 2 codensities gives

    W_m = sum_n phiL_n v[phiC_n phiC_m],   phiL = M^T phiC,

and one pass over the AOs and their gradients turns W into both terms the gradient needs:

    K[X] C     = (vol/ngrid) sum_g ao W^T                        (relaxed density)
    de[A, x]   = -2 sum_{i in A} sum_m (C M)_im G[x, i, m]        (pairing force)
    G[x, i, m] = (vol/ngrid) sum_g d_x ao_i W_m

pyscf's FFTDF ``get_k`` and ``get_k_e1`` spend nao^2 and 3 nao^2 complex transforms on the
same two terms.  The kernel is pyscf's ``get_coulG`` with the mean field's ``exxdiv``, as in
those calls.  Exact for a general M.
'''

import numpy as np
import scipy.fft
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.dft import numint

__all__ = ['pair_exchange_lowrank']

# Bytes per (codensity, grid point) in one FFT batch: the codensity, its half spectrum
# (16 B over half the mesh) and the potential, plus the scratch of the W updates.
_FFT_BYTES = 40
# Smallest grid block of an AO pass, so the BLAS products stay efficient.
_GRID_BLK_MIN = 1024


def _free_bytes(cell):
    return max(0., cell.max_memory - lib.current_memory()[0]) * 1e6


def _ao_blocks(cell, mesh, deriv, extra_per_point):
    '''(g0, g1, ao) over the uniform grid; ao is (g1-g0, nao), or (4, g1-g0, nao) with
    its gradients for ``deriv=1``, sized so two blocks and their per-point arrays fit.'''
    coords = cell.gen_uniform_grids(mesh)
    ncomp = 1 + 3 * deriv
    blk = int(_free_bytes(cell) // (2 * (8 * ncomp * cell.nao + extra_per_point)))
    blk = max(_GRID_BLK_MIN, min(len(coords), blk))
    for g0, g1 in lib.prange(0, len(coords), blk):
        ao = numint.eval_ao(cell, coords[g0:g1], deriv=deriv)
        yield g0, g1, np.asarray(ao.real if np.iscomplexobj(ao) else ao)


def _half_kernel(cell, mesh, exxdiv, mydf):
    '''get_coulG symmetrised in G -> -G and cut to the real-to-complex half mesh.'''
    nx, ny, nz = mesh
    w = tools.get_coulG(cell, np.zeros(3), exxdiv, mydf, mesh).reshape(nx, ny, nz)
    ix, iy, iz = ((-np.arange(n)) % n for n in mesh)
    w = 0.5 * (w + w[ix][:, iy][:, :, iz])
    return np.ascontiguousarray(w[:, :, :nz // 2 + 1])


def _potential(cell, mesh, phiC, phiL, w_half, log):
    '''W_m = sum_n phiL_n v[phiC_n phiC_m] for every row m, (r, ngrid).'''
    t0 = log.init_timer()
    r, ngrid = phiC.shape
    nx, ny, nz = mesh
    workers = lib.num_threads()
    n_idx, m_idx = np.tril_indices(r)               # row n contiguous, m <= n
    npair = len(n_idx)
    blk = int(_free_bytes(cell) // (_FFT_BYTES * ngrid))
    blk = max(1, min(npair, blk))
    log.debug('pairing potential: rank %d, %d codensity pairs per FFT batch', r, blk)
    W = np.zeros((r, ngrid))
    for p0, p1 in lib.prange(0, npair, blk):
        rho = phiC[n_idx[p0:p1]]
        rho *= phiC[m_idx[p0:p1]]
        vG = scipy.fft.rfftn(rho.reshape(-1, nx, ny, nz), axes=(1, 2, 3), workers=workers)
        rho = None
        vG *= w_half
        V = scipy.fft.irfftn(vG, s=(nx, ny, nz), axes=(1, 2, 3), workers=workers,
                             overwrite_x=True).reshape(-1, ngrid)
        vG = None
        rows, starts = np.unique(n_idx[p0:p1], return_index=True)
        for n, s0 in zip(rows, starts):
            s1 = s0 + 1
            while s1 < p1 - p0 and n_idx[p0 + s1] == n:
                s1 += 1
            m0, m1 = int(m_idx[p0 + s0]), int(m_idx[p0 + s1 - 1]) + 1
            Vs = V[s0:s1]                                    # pairs (n, m0..m1-1)
            k = m1 - m0 - (1 if m1 - 1 == n else 0)          # the diagonal pair once
            W[m0:m1] += phiL[n] * Vs                         # W_m += phiL_n V_nm
            if k > 0:
                W[n] += np.einsum('mg,mg->g', phiL[m0:m0 + k], Vs[:k])   # W_n += phiL_m V_nm
        V = None
    log.timer(f'pairing potential: {npair} codensity pairs (rank {r})', *t0)
    return W


def pair_exchange_lowrank(cell, C, M, mesh=None, exxdiv=None, mydf=None, verbose=None):
    '''K[X] C and the pairing-exchange force for X = C M C^T, Gamma point.

    Args:
        C : (nao, r) real orbitals.
        M : (r, r) real mixing matrix.

    Kwargs:
        mesh, exxdiv, mydf : the FFT mesh and the ``get_coulG`` arguments of the exchange
            being reproduced (default ``cell.mesh``, no G = 0 correction).

    Returns:
        kc : (nao, r) K[X] @ C, the ``pair_k`` of ``make_rdm1_relaxed_rhf_pprpa``.
        de : (natm, 3) the pairing-exchange contribution to dE/dR, the CPU reference's
            2 sum_{i in A} sum_l (d_x K[X])_il X_il.
    '''
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    C = np.asarray(C, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    mesh = np.asarray(cell.mesh if mesh is None else mesh, dtype=int)
    ngrid = int(np.prod(mesh))
    nao, r = C.shape

    phiC = np.empty((r, ngrid))
    for g0, g1, ao in _ao_blocks(cell, mesh, 0, 8 * r):
        phiC[:, g0:g1] = lib.dot(C.T, ao.T)
    phiL = lib.dot(M.T, phiC)
    W = _potential(cell, mesh, phiC, phiL, _half_kernel(cell, mesh, exxdiv, mydf), log)
    phiC = phiL = None

    # one AO pass with gradients for both K[X] C (values) and G (gradients)
    kc = np.zeros((nao, r))
    G = np.zeros((3, nao, r))
    for g0, g1, ao in _ao_blocks(cell, mesh, 1, 8 * r):
        Wt = np.ascontiguousarray(W[:, g0:g1].T)
        kc += lib.dot(ao[0].T, Wt)
        for x in range(3):
            G[x] += lib.dot(ao[1 + x].T, Wt)
    W = None
    scale = cell.vol / ngrid
    kc *= scale
    G *= scale
    # eval_ao differentiates with respect to the electron coordinate; the nuclear
    # derivative flips the sign
    t = np.einsum('xim,im->xi', G, C @ M)
    de = np.array([-2.0 * t[:, p0:p1].sum(axis=1)
                   for _, _, p0, p1 in cell.aoslice_by_atom()])
    log.timer('pair exchange and pairing force', *t0)
    return kc, de

