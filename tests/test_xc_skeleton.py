"""The Vxc skeleton of the reference density taken in the fxc-skeleton grid pass
(_contract_xc_kernel_krks with_vxc) against pyscf's pbc krks.get_vxc."""

import numpy as np
import pytest

from lib_pprpa.grad.grad_utils import _contract_xc_kernel_krks


@pytest.mark.parametrize("xc", ("lda,vwn", "pbe"))
def test_vxc_skeleton_in_the_fxc_pass(xc):
    from pyscf.pbc import dft, gto
    from pyscf.pbc.grad.krks import get_vxc
    a0 = 3.5668
    cell = gto.Cell()
    cell.atom = [["C", [0.0, 0.0, 0.0]], ["C", [a0 / 4 + 0.05, a0 / 4, a0 / 4 - 0.03]]]
    cell.a = np.array([[0, a0 / 2, a0 / 2], [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]])
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.mesh = [15, 15, 15]
    cell.verbose = 0
    cell.build()
    kmf = dft.KRKS(cell, kpts=np.zeros((1, 3)), xc=xc)
    kmf.exxdiv = None
    kmf.run()
    P = np.random.default_rng(4).standard_normal((cell.nao, cell.nao))
    P = P + P.T
    f1vo, _, v1ao, _ = _contract_xc_kernel_krks(kmf, xc, P, with_vxc=True)
    ref = get_vxc(kmf._numint, cell, kmf.grids, xc, kmf.make_rdm1(), kmf.kpts)[:, 0]
    np.testing.assert_allclose(v1ao, ref, rtol=0, atol=1e-10)
    np.testing.assert_allclose(f1vo, _contract_xc_kernel_krks(kmf, xc, P)[0], rtol=0, atol=1e-12)
