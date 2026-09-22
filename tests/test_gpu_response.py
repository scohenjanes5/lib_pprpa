"""Grouped Gamma RKS response (grad.gpu_response) vs the dense GPU response
(pprpa_gamma_gpu.make_gpu_vresp), LDA and GGA, one and two virtual slots."""
from __future__ import annotations

import numpy as np

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None


def _need_cupy():
    try:
        import cupy as _cp
        _cp.cuda.runtime.getDevice()
    except Exception:
        if pytest is None:
            raise RuntimeError("CUDA GPU required for this test") from None
        pytest.skip("CUDA GPU required")


def _two_slots():
    import cupy as _cp
    return [0, 1] if _cp.cuda.runtime.getDeviceCount() >= 2 else [0, 0]


def _cell_and_mf(xc):
    from pyscf.pbc import gto, dft as cdft
    a0 = 3.5668
    cell = gto.Cell()
    cell.atom = [["C", [0.0, 0.0, 0.0]], ["C", [a0 / 4 + 0.03, a0 / 4, a0 / 4 - 0.02]]]
    cell.a = np.array([[0, a0 / 2, a0 / 2], [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]])
    cell.basis = "gth-dzvp"
    cell.pseudo = "gth-pade"
    cell.ke_cutoff = 120.0
    cell.verbose = 0
    cell.build()
    mf = cdft.RKS(cell, xc=xc)
    mf.exxdiv = None
    mf.conv_tol = 1e-9
    mf.kernel()
    return cell, mf


def _random_sym_dm(nao, rng, rank=None):
    if rank is None:
        A = rng.standard_normal((nao, nao))
    else:
        A = rng.standard_normal((nao, rank)) @ rng.standard_normal((rank, nao))
    return (A + A.T) * 0.05


def _check(xc, slots, gchunk=None):
    from lib_pprpa.grad.pprpa_gamma_gpu import make_gpu_vresp
    from lib_pprpa.grad.gpu_response import GammaResponse
    from lib_pprpa.gpu_multi import DeviceGroup
    cell, mf = _cell_and_mf(xc)
    rng = np.random.default_rng(3)
    dense = make_gpu_vresp(cell, mf)
    resp = GammaResponse(cell, mf, group=DeviceGroup(slots), gchunk=gchunk, verbose=False)
    for dm in (_random_sym_dm(cell.nao, rng), _random_sym_dm(cell.nao, rng, rank=3)):
        v_ref = dense(dm)
        v_new = resp(dm)
        scale = max(np.abs(v_ref).max(), 1e-12)
        err = np.abs(v_new - v_ref).max() / scale
        assert err < 1e-10, (xc, slots, err)
        assert np.abs(v_new - v_new.T).max() < 1e-12 * scale
    assert resp.stats["calls"] == 2
    assert "calls" in resp.summary()
    resp.release()


def test_gga_matches_dense_one_slot():
    _need_cupy()
    _check("pbe", [0])


def test_gga_matches_dense_two_slots_small_chunks():
    _need_cupy()
    _check("pbe", _two_slots(), gchunk=777)


def test_lda_matches_dense():
    _need_cupy()
    _check("lda,vwn", _two_slots())


def test_survives_other_kernels_on_the_shared_group():
    """The exchange build runs on the same DeviceGroup between construction and
    the first call and frees its own state keys; ours must not be among them."""
    _need_cupy()
    from lib_pprpa.grad.pprpa_gamma_gpu import make_gpu_vresp
    from lib_pprpa.grad.gpu_response import GammaResponse
    from lib_pprpa.gpu_fft_k import get_k_lowrank
    from lib_pprpa.gpu_multi import DeviceGroup
    cell, mf = _cell_and_mf("pbe")
    g = DeviceGroup(_two_slots())
    resp = GammaResponse(cell, mf, group=g, verbose=False)
    rng = np.random.default_rng(8)
    L = rng.standard_normal((cell.nao, 3))
    get_k_lowrank(cell, cell.mesh, (L, L), ket=L, group=g, verbose=False)   # frees "gchunk" etc.
    dm = _random_sym_dm(cell.nao, rng)
    v_ref = make_gpu_vresp(cell, mf)(dm)
    v_new = resp(dm)
    assert np.abs(v_new - v_ref).max() < 1e-10 * np.abs(v_ref).max()
    resp.release()


def test_factory_falls_back_for_unsupported_functional():
    _need_cupy()
    import os
    from lib_pprpa.grad.gpu_response import make_gpu_vresp_grouped
    cell, mf = _cell_and_mf("pbe")
    assert make_gpu_vresp_grouped(cell, mf, verbose=False) is not None
    os.environ["PPRPA_RESPONSE"] = "dense"
    try:
        assert make_gpu_vresp_grouped(cell, mf, verbose=False) is None
    finally:
        os.environ.pop("PPRPA_RESPONSE", None)


if __name__ == "__main__":
    _need_cupy()
    test_gga_matches_dense_one_slot()
    print("OK  test_gga_matches_dense_one_slot")
    test_gga_matches_dense_two_slots_small_chunks()
    print("OK  test_gga_matches_dense_two_slots_small_chunks")
    test_lda_matches_dense()
    print("OK  test_lda_matches_dense")
    test_survives_other_kernels_on_the_shared_group()
    print("OK  test_survives_other_kernels_on_the_shared_group")
    test_factory_falls_back_for_unsupported_functional()
    print("OK  test_factory_falls_back_for_unsupported_functional")
