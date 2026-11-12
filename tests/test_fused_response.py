"""The CPU fused Gamma KS response against pyscf's gen_response(singlet=None, hermi=1),
for the full matrix and the occupied columns the CPHF iterations use."""

import numpy as np
import pytest

from lib_pprpa.grad.fused_response import fused_applies, fused_response


def _mf(xc):
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
    mf = dft.RKS(cell, xc=xc)
    mf.exxdiv = None
    mf.conv_tol = 1e-10
    return mf.run()


@pytest.mark.parametrize("max_memory", (4000, 1))
@pytest.mark.parametrize("xc", ("lda,vwn", "pbe"))
def test_fused_response_matches_gen_response(xc, max_memory):
    mf = _mf(xc)
    assert fused_applies(mf)
    ref = mf.gen_response(singlet=None, hermi=1)
    mf.max_memory = max_memory              # 1 MB: AO values recomputed in every pass
    respond = fused_response(mf)
    rng = np.random.default_rng(1)
    nao = mf.cell.nao
    dm = rng.standard_normal((nao, nao))
    dm = dm + dm.T
    np.testing.assert_allclose(respond(dm), ref(dm), rtol=0, atol=1e-10)

    C_A = mf.mo_coeff[:, mf.mo_occ == 0]
    C_I = mf.mo_coeff[:, mf.mo_occ > 0]
    t = rng.standard_normal((C_A.shape[1], C_I.shape[1]))
    d = C_A @ t @ C_I.T
    np.testing.assert_allclose(respond(d + d.T, ket=C_I), ref(d + d.T) @ C_I, rtol=0, atol=1e-10)


def test_hybrids_use_gen_response():
    assert not fused_applies(_mf("pbe0"))
