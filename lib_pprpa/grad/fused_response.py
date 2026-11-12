'''Gamma-point KS response (J + fxc)[D] on the uniform grid, CPU counterpart of
``gpu_response``.

pyscf's ``gen_response`` builds fxc[D] in one AO pass and the FFTDF J[D] in two more (its
density, then its matrix), evaluating the AOs on the grid each time.  Here one density
pass serves both (rho1[0] is J's density), J's potential joins the fxc weights, and one
potential pass builds the matrix; the AO values are evaluated once and kept while they fit
in ``max_memory``.  With a ``ket`` (the occupied orbitals of a CPHF iteration) it returns
V @ ket directly from Z = ao0^T (aow ket) + aow^T (ao0 ket), 2 nk / nao of a nao^2
product.  LDA/GGA without exact exchange, real symmetric D, FFTDF J on the XC mesh; matches
``gen_response`` up to the summation order.
'''

import numpy as np
import scipy.fft
from pyscf import lib
from pyscf.dft.numint import _scale_ao
from pyscf.pbc import tools
from pyscf.pbc.dft import numint

__all__ = ['fused_applies', 'fused_response']

# Smallest grid block of an AO pass, so the BLAS products stay efficient.
_GRID_BLK_MIN = 1024


def fused_applies(mf):
    '''``fused_response`` covers this Gamma KS mean field: uniform grids on the FFTDF mesh,
    LDA/GGA without exact exchange.'''
    from pyscf.pbc.df.fft import FFTDF
    from pyscf.pbc.dft.gen_grid import UniformGrids
    ni = mf._numint
    return (isinstance(mf.grids, UniformGrids) and type(mf.with_df) is FFTDF
            and np.array_equal(np.asarray(mf.grids.mesh), np.asarray(mf.with_df.mesh))
            and ni._xc_type(mf.xc) in ('LDA', 'GGA') and not ni.libxc.is_hybrid_xc(mf.xc))


def fused_response(mf):
    '''``respond(dm, ket=None)`` for the Gamma KS mean field ``mf`` (see ``fused_applies``):
    (J + fxc)[dm] (nao, nao) for a real symmetric dm, as
    ``mf.gen_response(singlet=None, hermi=1)``, or that matrix times ``ket`` (nao, nk).
    '''
    cell, ni, grids, xc = mf.cell, mf._numint, mf.grids, mf.xc
    xctype = ni._xc_type(xc)
    deriv = 0 if xctype == 'LDA' else 1
    ncomp = 1 + 3 * deriv
    mesh = np.asarray(grids.mesh)
    nx, ny, nz = mesh
    coords = cell.gen_uniform_grids(mesh)
    ngrid = len(coords)
    weight = cell.vol / ngrid
    nao = cell.nao
    fxc = ni.cache_xc_kernel(cell, grids, xc, np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ),
                             0)[2]
    coulG = tools.get_coulG(cell, mesh=mesh).reshape(nx, ny, nz)[:, :, :nz // 2 + 1]
    workers = lib.num_threads()

    free = max(0., mf.max_memory - lib.current_memory()[0]) * 1e6
    blk = int(free // (2 * 8 * (ncomp * nao + 2 * ncomp)))
    blk = max(_GRID_BLK_MIN, min(ngrid, blk))
    spans = list(lib.prange(0, ngrid, blk))
    # the AO values of every block, kept when they fit beside the passes
    keep = 8 * ngrid * ncomp * nao <= 0.5 * free
    cache = {}

    def blocks():
        for g0, g1 in spans:
            ao = cache.get((g0, g1))
            if ao is None:
                ao = numint.eval_ao(cell, coords[g0:g1], deriv=deriv)
                ao = np.asarray(ao.real if np.iscomplexobj(ao) else ao).reshape(ncomp, g1 - g0, nao)
                if keep:
                    cache[g0, g1] = ao
            yield g0, g1, ao

    def potential_weights(dm):
        rho = np.empty((ncomp, ngrid))
        for g0, g1, ao in blocks():
            c0 = lib.dot(ao[0], dm)
            rho[0, g0:g1] = np.einsum('gi,gi->g', c0, ao[0])
            for x in range(1, ncomp):
                rho[x, g0:g1] = 2 * np.einsum('gi,gi->g', c0, ao[x])
        rhoG = scipy.fft.rfftn(rho[0].reshape(nx, ny, nz), workers=workers)
        vj = scipy.fft.irfftn(rhoG * coulG, s=(nx, ny, nz), workers=workers).ravel()
        if xctype == 'LDA':
            wv = rho[:1] * fxc[0, 0] * weight
        else:
            wv = np.einsum('xg,xyg->yg', rho, fxc) * weight
        wv[0] += vj * weight
        wv[0] *= .5                    # the matrix is Z + Z^T, as _vxc_mat with hermi=1
        return wv

    def scaled(ao, wv):
        return _scale_ao(ao[0], wv[0]) if xctype == 'LDA' else _scale_ao(ao, wv)

    def respond(dm, ket=None):
        wv = potential_weights(np.asarray(dm))
        if ket is None:
            Z = np.zeros((nao, nao))
            for g0, g1, ao in blocks():
                Z += lib.dot(ao[0].T, scaled(ao, wv[:, g0:g1]))
            return Z + Z.T
        ket = np.asarray(ket)
        Z = np.zeros((nao, ket.shape[1]))
        for g0, g1, ao in blocks():
            aow = scaled(ao, wv[:, g0:g1])
            Z += lib.dot(ao[0].T, lib.dot(aow, ket))
            Z += lib.dot(aow.T, lib.dot(ao[0], ket))
        return Z

    respond.accepts_ket = True
    return respond
