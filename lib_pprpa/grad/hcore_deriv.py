'''Density-contracted one-electron nuclear derivative at the Gamma point on the CPU (the
counterpart of ``hcore_contracted``).

``pyscf.pbc.grad.krhf.hcore_generator(ia)`` evaluates every AO on the uniform mesh and
forms three (nao, nao) products per atom, which the gradient then traces against a
density.  The trace only needs the density on the mesh: each atom's local-pseudopotential
term is (1/N) Re sum_G vloc_A^x(G) rho(G)^* by Parseval, and the AO-centre terms are slices
of ``get_hcore``, built once.  Pseudopotential cells, Gamma point only, as ``get_hcore``.
'''

import numpy as np
from pyscf import lib
from pyscf.pbc import tools
from pyscf.pbc.dft import numint
from pyscf.pbc.gto.pseudo.pp import get_vlocG

__all__ = ['hcore_deriv_contracted']


def hcore_deriv_contracted(mf_grad, cell, dm):
    '''sum_ij hcore_generator(ia)[x, 0, i, j] dm[i, j] for every atom ia, (natm, 3).

    mf_grad : the Gamma KRHF/KRKS gradient object whose ``get_hcore`` supplies the
        AO-centre derivative integrals.
    dm : (nao, nao) real AO density.
    '''
    dm = np.asarray(dm)
    mesh = cell.mesh
    coords = cell.get_uniform_grids()
    ngrid = len(coords)
    free = max(0., cell.max_memory - lib.current_memory()[0]) * 1e6
    blk = max(1024, min(ngrid, int(free // (4 * 8 * cell.nao))))
    rho = np.empty(ngrid)
    for g0, g1 in lib.prange(0, ngrid, blk):
        ao = numint.eval_ao(cell, coords[g0:g1])
        ao = ao.real if np.iscomplexobj(ao) else ao
        rho[g0:g1] = np.einsum('gi,gi->g', lib.dot(ao, dm), ao)
    rho_G = tools.fft(rho, mesh)

    # hcore_generator's local term of atom a, sum_g rho(g) ifft(1j G_x SI_a vlocG_a)(g)
    w = cell.get_SI() * get_vlocG(cell)
    w *= rho_G.conj()
    de = lib.dot(1j * w, cell.Gv).real / ngrid

    h1 = np.asarray(mf_grad.get_hcore(cell, np.zeros((1, 3))))[0]
    if np.iscomplexobj(h1):
        h1 = h1.real
    for ia, (_, _, p0, p1) in enumerate(cell.aoslice_by_atom()):
        de[ia] -= np.einsum('xij,ij->x', h1[:, p0:p1], dm[p0:p1])
        de[ia] -= np.einsum('xij,ji->x', h1[:, p0:p1], dm[:, p0:p1])
    return de
