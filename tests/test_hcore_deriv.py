"""The CPU density-contracted hcore derivative against pyscf's per-atom hcore_generator."""

import numpy as np

from lib_pprpa.grad.hcore_deriv import hcore_deriv_contracted


def test_hcore_deriv_contracted_matches_generator():
    from pyscf.pbc import dft, gto
    a0 = 3.5668
    cell = gto.Cell()
    cell.atom = [["C", [0.0, 0.0, 0.0]], ["C", [a0 / 4 + 0.05, a0 / 4, a0 / 4 - 0.03]]]
    cell.a = np.array([[0, a0 / 2, a0 / 2], [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]])
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.mesh = [15, 15, 15]
    cell.verbose = 0
    cell.build()
    kpts = np.zeros((1, 3))
    kg = dft.KRKS(cell, kpts=kpts, xc="pbe").nuc_grad_method()
    dm = np.random.default_rng(2).standard_normal((cell.nao, cell.nao))  # both slice terms
    hcore_deriv = kg.hcore_generator(cell, kpts)
    ref = np.array([np.einsum('xij,ij->x', hcore_deriv(ia)[:, 0], dm)
                    for ia in range(cell.natm)])
    np.testing.assert_allclose(hcore_deriv_contracted(kg, cell, dm), ref, rtol=0, atol=1e-10)
