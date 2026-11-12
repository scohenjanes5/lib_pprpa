"""The CPU low-rank pair exchange and pairing force against the FFTDF get_k / get_k_e1 they
replace, for symmetric (singlet) and antisymmetric (triplet) amplitudes."""

import numpy as np
import pytest

from lib_pprpa.grad.pairing_force import pair_exchange_lowrank


@pytest.fixture(scope="module")
def cell():
    from pyscf.pbc import gto
    a0 = 3.5668
    cell = gto.Cell()
    cell.atom = [["C", [0.0, 0.0, 0.0]], ["C", [a0 / 4 + 0.05, a0 / 4, a0 / 4 - 0.03]]]
    cell.a = np.array([[0, a0 / 2, a0 / 2], [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]])
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.mesh = [15, 15, 15]
    cell.verbose = 0
    cell.build()
    return cell


def _amplitudes(cell, sym, seed=3):
    rng = np.random.default_rng(seed)
    nao, r = cell.nao, 6
    C = np.linalg.qr(rng.standard_normal((nao, nao)))[0][:, :r]
    M = rng.standard_normal((r, r))
    M = M + M.T if sym else M - M.T
    return C, M


@pytest.mark.parametrize("exxdiv", (None, "ewald"))
@pytest.mark.parametrize("sym", (True, False))
def test_pair_exchange_matches_fftdf(cell, sym, exxdiv):
    from pyscf.pbc import df
    mydf = df.FFTDF(cell)
    C, M = _amplitudes(cell, sym)
    X = C @ M @ C.T
    kc, de = pair_exchange_lowrank(cell, C, M, exxdiv=exxdiv, mydf=mydf)

    vk = mydf.get_jk(X, hermi=0, with_j=False, exxdiv=exxdiv)[1]
    np.testing.assert_allclose(kc, vk @ C, rtol=0, atol=1e-10)

    vk1 = np.asarray(mydf.get_k_e1(X[None], np.zeros((1, 3)), exxdiv=exxdiv))
    vk1 = vk1.reshape(3, cell.nao, cell.nao)
    ref = np.array([2 * np.einsum('xij,ij->x', vk1[:, p0:p1], X[p0:p1])
                    for _, _, p0, p1 in cell.aoslice_by_atom()])
    np.testing.assert_allclose(de, ref, rtol=0, atol=1e-10)

