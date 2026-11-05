'''Gamma-point KS response (J + fxc)[D] on the uniform grid in two AO passes.

gpu4pyscf's ``KRKS.gen_response`` builds fxc[D] (density and potential passes) and
J[D] (FFTDF ``get_j``: two more) apart: three passes, four nao^2 x ngrid products.
Here one density pass serves both (rho1[0] is J's density), J's potential joins the
fxc weights, and one potential pass builds the matrix: two passes, two products.  With
a ``ket`` (the occupied orbitals of a CPHF iteration) it returns V @ ket directly from
Z = ao0^T (aow ket) + aow^T (ao0 ket), 2 nk / nao of a nao^2 product.  LDA/GGA, real
symmetric D (``hermi=1``), FFTDF J; matches ``gen_response`` up to the summation order.

A hybrid's exchange -K[D]/2 comes from the eigen-factors of D over the MOs
(``gpu_exchange``): rank x nket codensity pairs, for a CPHF iteration 2 nocc x nocc instead
of the nao^2 of FFTDF ``get_k``.  Exact on the MO span: the ket, or, without one, every MO
(V P with P the S-projector onto the MOs).  Given the CPHF form D = cross ket^T + ket cross^T
the exchange takes 1.5 nocc^2 pairs (``exchange_cross_ket``).
'''

import numpy as np
import cupy as cp
from gpu4pyscf.pbc import tools as gtools
from gpu4pyscf.pbc.dft.numint import _scale_ao

from lib_pprpa.gpu_coulomb import exchange_coulG
from lib_pprpa.gpu_exchange import exchange_cross_ket, exchange_ket, factor, projector

__all__ = ['fused_response']

_KPTS0 = np.zeros((1, 3))


def fused_response(cell, kmf):
    '''``respond(dm, ket=None, cross=None)`` for the Gamma KRKS ``kmf`` (KNumInt on its uniform
    grids, LDA/GGA or their hybrids, ``kmf.mo_coeff`` set): (J + fxc - K/2)[dm] (nao, nao) for
    a real symmetric CuPy dm, or that matrix times ``ket`` (nao, nk), CuPy.  ``cross`` (nao, nk)
    with a ket states dm = cross ket^T + ket cross^T, for the exchange's pairs.
    '''
    ni, grids, xc = kmf._numint, kmf.grids, kmf.xc
    if grids.coords is None:
        grids.build()
    xctype = ni._xc_type(xc)
    deriv = 0 if xctype == 'LDA' else 1
    ncomp = 1 + 3 * deriv
    mesh = np.asarray(cell.mesh)
    ngrid = int(np.prod(mesh))
    weight = cell.vol / ngrid
    fxc = ni.cache_xc_kernel1(cell, grids, xc, kmf.make_rdm1(), 0, _KPTS0, is_rhf=True)[2]
    order = cp.asarray(grids.argsort(tile=8))              # block_loop(sort_grids=True)
    coulG = gtools.get_coulG(cell, mesh=mesh)

    def blocks():
        p0 = 0
        for ao_ks, w, _ in ni.block_loop(cell, grids, deriv, _KPTS0, sort_grids=True):
            p1 = p0 + w.size
            yield p0, p1, ao_ks
            p0 = p1

    def potential_weights(dm):
        rho = cp.empty((ncomp, ngrid))
        for p0, p1, ao_ks in blocks():
            rho[:, p0:p1] = ni.eval_rho(cell, ao_ks, dm[None], xctype=xctype,
                                        hermi=1).real.reshape(ncomp, -1)
        rho_g = cp.empty(ngrid)
        rho_g[order] = rho[0]
        vj = gtools.ifft(coulG * gtools.fft(rho_g[None], mesh), mesh).real[0]
        wv = cp.einsum('xg,xyg->yg', rho, fxc) * weight
        wv[0] += vj[order] * weight
        wv[0] *= .5                    # the matrix is Z + Z^T, as _vxc_mat with hermi=1
        return wv

    def scaled(ao, wv):
        return _scale_ao(ao, wv[0]) if xctype == 'LDA' else _scale_ao(ao[:4], wv)

    hybrid = ni.libxc.is_hybrid_xc(xc)
    if hybrid:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc)
        coulG_k = exchange_coulG(cell, mesh, omega, alpha, hyb, kmf.exxdiv)
        mo = cp.asarray(kmf.mo_coeff[0])
        proj = projector(mo, cell.pbc_intor('int1e_ovlp', hermi=1))

    def exchange(dm, ket, cross):
        if cross is not None:
            return exchange_cross_ket(cell, mesh, ket, cross, coulG_k, verbose=kmf.verbose)
        lam, U = factor(proj @ dm @ proj.T, mo)
        K = exchange_ket(cell, mesh, lam, U, mo if ket is None else ket, coulG_k,
                         verbose=kmf.verbose)
        return K if ket is not None else K @ proj

    def respond(dm, ket=None, cross=None):
        wv = potential_weights(dm)
        nk = cell.nao if ket is None else ket.shape[1]
        Z = cp.zeros((cell.nao, nk))
        for p0, p1, ao_ks in blocks():
            ao = ao_ks[0]
            ao0 = ao if xctype == 'LDA' else ao[0]
            aow = scaled(ao, wv[:, p0:p1])
            if ket is None:
                Z += ao0.T @ aow
            else:
                Z += ao0.T @ (aow @ ket)
                Z += aow.T @ (ao0 @ ket)
        if ket is None:
            Z = Z + Z.T
        if hybrid:
            Z -= .5 * exchange(dm, ket, cross)
        return Z

    return respond
