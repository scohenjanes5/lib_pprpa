"""The fused Gamma response (J inside the fxc pass, a hybrid's exchange from the MO factors)
against gpu4pyscf's KRKS.gen_response: the full matrix and the columns of the occupied ket,
for a CPHF (vir-occ) and a general density, LDA, GGA and hybrids; make_gpu_vresp's routing."""

import numpy as np
import pytest


def _gpu4pyscf_available():
    try:
        import cupy as cp
        import gpu4pyscf  # noqa: F401
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _gpu4pyscf_available(), reason="GPU4PySCF CUDA device is unavailable")


def _diamond(xc):
    from pyscf.pbc import dft, gto
    a0 = 3.370137329
    cell = gto.M(atom=[["C", [0., 0., 0.]], ["C", [a0 / 2, a0 / 2, a0 / 2]]],
                 a=np.array([[0, a0, a0], [a0, 0, a0], [a0, a0, 0]]),
                 unit="bohr", basis="gth-dzv", pseudo="gth-pade", mesh=[20] * 3, verbose=0)
    mf = dft.RKS(cell, xc=xc)
    mf.exxdiv = None
    mf.conv_tol = 1e-10
    mf.kernel()
    return cell, mf


def _trial(mf):
    occ = mf.mo_occ
    o, v = mf.mo_coeff[:, occ > 0], mf.mo_coeff[:, occ == 0]
    t = np.random.default_rng(0).standard_normal((v.shape[1], o.shape[1]))
    dm = v @ t @ o.T
    return o, v, t, dm + dm.T


@pytest.mark.parametrize("xc", ("pbe", "lda,vwn", "pbe0", "b3lyp",
                                "0.25*SR_HF(1.0) + 0.75*PBE, PBE",
                                "0.4*LR_HF(1.0) + 0.6*PBE, PBE"))
def test_fused_response_matches_gen_response(xc):
    import cupy as cp
    from gpu4pyscf.pbc import dft as gdft
    from lib_pprpa.gpu_response import fused_response
    cell, mf = _diamond(xc)
    o, v, t, dm = _trial(mf)
    r = np.random.default_rng(1).standard_normal((cell.nao,) * 2)
    dm_general = mf.mo_coeff @ (r + r.T) @ mf.mo_coeff.T
    kmf = gdft.KRKS(cell, kpts=np.zeros((1, 3)), xc=xc)
    kmf.exxdiv = None
    kmf.mo_coeff = cp.asarray(mf.mo_coeff)[None]
    kmf.mo_energy = cp.asarray(mf.mo_energy)[None]
    kmf.mo_occ = cp.asarray(mf.mo_occ)[None]
    kmf.grids.build()
    nao = cell.nao
    native = kmf.gen_response(singlet=None, hermi=1)
    respond = fused_response(cell, kmf)
    # the exchange's FFTs run real-to-complex: 1e-9 for hybrids
    tol = 1e-9 if mf._numint.libxc.is_hybrid_xc(xc) else 1e-10
    for d in (dm, dm_general):
        ref = cp.asnumpy(native(cp.asarray(d).reshape(1, 1, nao, nao))[0, 0])
        got = cp.asnumpy(respond(cp.asarray(d)))
        assert np.abs(got - ref).max() < tol * np.abs(ref).max()
        got_ket = cp.asnumpy(respond(cp.asarray(d), ket=cp.asarray(o)))
        assert np.abs(got_ket - ref @ o).max() < tol * np.abs(ref @ o).max()


def test_make_gpu_vresp_routes():
    """LDA/GGA and their hybrids on FFTDF get the fused response, GDF gen_response; the
    response serves ``ket``."""
    from gpu4pyscf.pbc import dft as gdft
    from lib_pprpa.grad.pprpa_gamma_gpu import _fused_applies, make_gpu_vresp
    cell, mf = _diamond("pbe")
    o, v, t, dm = _trial(mf)
    assert not _fused_applies(gdft.KRKS(cell, kpts=np.zeros((1, 3)), xc="pbe").density_fit())
    # the ket and the full matrix factor a hybrid's density differently: 1e-9
    for xc, tol in (("pbe", 1e-10), ("pbe0", 1e-9)):
        mf.xc = xc
        assert _fused_applies(gdft.KRKS(cell, kpts=np.zeros((1, 3)), xc=xc))
        vresp = make_gpu_vresp(cell, mf, emulate=False)
        ref = vresp(dm) @ o
        np.testing.assert_allclose(vresp(dm, ket=o), ref, rtol=0, atol=tol * np.abs(ref).max())
