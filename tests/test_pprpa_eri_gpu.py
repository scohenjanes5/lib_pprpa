"""Tiled vs resident GPU ERI contraction, plus the ao2mo VRAM fit helper."""
from __future__ import annotations

import numpy as np

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

from lib_pprpa.gpu_mem import eri_bytes, fits_resident
from lib_pprpa.pprpa_eri_gpu import (
    attach_gpu_eri_contraction,
    eri_mvp_tiled,
    estimate_eri_tile,
    get_last_telemetry,
    release_gpu_eri,
)


def _need_cupy():
    try:
        import cupy as _cp
        _cp.cuda.runtime.getDevice()
    except Exception:
        if pytest is None:
            raise RuntimeError("CUDA GPU required for this test") from None
        pytest.skip("CUDA GPU required")


def _parametrize(*args):
    if pytest is None:
        return lambda f: f
    return pytest.mark.parametrize(*args)


def test_eri_bytes_as300():
    assert eri_bytes(300, 300) == 3 * (300 ** 4) * 8
    assert eri_bytes(128, 124) == (124 ** 4 + 128 ** 4 + (128 ** 2) * (124 ** 2)) * 8


def test_fits_resident_matches_ao2mo_rule():
    nocc = nvir = 300
    total = 183359 * 1024 * 1024  # B200 nvidia-smi
    extra = 0
    assert not fits_resident(nocc, nvir, extra_bytes=extra, total_bytes=total)
    assert fits_resident(128, 124, extra_bytes=0, total_bytes=total)
    tiny = eri_bytes(8, 8)
    assert fits_resident(8, 8, extra_bytes=0, total_bytes=int(tiny / 0.75) + 8)
    assert not fits_resident(8, 8, extra_bytes=0, total_bytes=int(tiny / 0.75) - 8)


def _random_eri(rng, nocc, nvir):
    vvvv = rng.standard_normal((nvir, nvir, nvir, nvir))
    oooo = rng.standard_normal((nocc, nocc, nocc, nocc))
    oovv = rng.standard_normal((nocc, nocc, nvir, nvir))
    return vvvv, oooo, oovv


def test_eri_mvp_tiled_matches_full_numpy():
    rng = np.random.default_rng(1)
    nocc, nvir, ntri = 5, 7, 11
    no2, nv2 = nocc * nocc, nvir * nvir
    vvvv, oooo, oovv = _random_eri(rng, nocc, nvir)
    V = np.ascontiguousarray(vvvv.reshape(nv2, nv2))
    O = np.ascontiguousarray(oooo.reshape(no2, no2))
    OV = np.ascontiguousarray(oovv.reshape(no2, nv2))
    zvvT = rng.standard_normal((ntri, nv2))
    zooT = rng.standard_normal((ntri, no2))
    full_vv = zvvT @ V.T + zooT @ OV
    full_oo = zooT @ O.T + zvvT @ OV.T
    for tile in (1, 3, 17, nv2, no2 * nv2):
        t_vv, t_oo = eri_mvp_tiled(zvvT, zooT, V, OV, O, tile, xp=np)
        np.testing.assert_allclose(t_vv, full_vv, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(t_oo, full_oo, rtol=1e-12, atol=1e-12)


def test_estimate_eri_tile_forced():
    assert estimate_eri_tile(90, tile=17) == 17
    assert estimate_eri_tile(10, tile=99) == 10
    assert estimate_eri_tile(10, tile=0) == 1


def _make_solver(nocc, nvir, moe, vvvv, oovv, oooo, multi):
    from lib_pprpa.pprpa_davidson import ppRPA_Davidson

    mp = ppRPA_Davidson(
        nocc, moe, Lpq=None, channel="hh",
        nroot=2, residue_thresh=1e-10, trial="identity",
    )
    mp.mu = 0.0
    mp.multi = multi
    mp.use_eri(vvvv, oovv, oooo)
    mp.check_parameter()
    return mp


def test_auto_selects_tiled_when_vram_too_small():
    _need_cupy()
    rng = np.random.default_rng(2)
    nocc, nvir = 4, 5
    nmo = nocc + nvir
    moe = rng.standard_normal(nmo)
    vvvv, oooo, oovv = _random_eri(rng, nocc, nvir)
    gpu = _make_solver(nocc, nvir, moe, vvvv, oovv, oooo, "t")
    attach_gpu_eri_contraction(
        gpu, vvvv, oovv, oooo, mode="auto", tile=3, total_bytes=1,
    )
    assert get_last_telemetry()["mode"] == "tiled"
    release_gpu_eri(gpu)


def test_auto_selects_resident_when_vram_is_plenty():
    _need_cupy()
    rng = np.random.default_rng(3)
    nocc, nvir = 4, 5
    nmo = nocc + nvir
    moe = rng.standard_normal(nmo)
    vvvv, oooo, oovv = _random_eri(rng, nocc, nvir)
    gpu = _make_solver(nocc, nvir, moe, vvvv, oovv, oooo, "t")
    attach_gpu_eri_contraction(
        gpu, vvvv, oovv, oooo, mode="auto",
        total_bytes=10 * 1024 ** 3,
    )
    assert get_last_telemetry()["mode"] == "resident"
    release_gpu_eri(gpu)


@_parametrize("multi", ("s", "t"))
@_parametrize("ntri", (1, 7, 16))
@_parametrize("tile", (1, 5, 64))
def test_tiled_mvp_matches_resident_and_cpu(multi, ntri, tile):
    _need_cupy()
    from lib_pprpa.pprpa_davidson import _pprpa_contraction

    rng = np.random.default_rng(4)
    nocc, nvir = 6, 8
    nmo = nocc + nvir
    moe = rng.standard_normal(nmo)
    vvvv, oooo, oovv = _random_eri(rng, nocc, nvir)
    cpu = _make_solver(nocc, nvir, moe, vvvv, oovv, oooo, multi)
    tv = rng.standard_normal((ntri, cpu.full_dim))
    mv_cpu = _pprpa_contraction(cpu, tv)

    gpu = _make_solver(nocc, nvir, moe, vvvv, oovv, oooo, multi)
    attach_gpu_eri_contraction(gpu, vvvv, oovv, oooo, mode="resident")
    mv_res = gpu.contraction(tv)
    attach_gpu_eri_contraction(gpu, vvvv, oovv, oooo, mode="tiled", tile=tile)
    mv_tile = gpu.contraction(tv)
    np.testing.assert_allclose(mv_res, mv_cpu, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(mv_tile, mv_res, rtol=1e-11, atol=1e-11)
    release_gpu_eri(gpu)


if __name__ == "__main__":
    import traceback

    def _run(fn, *a):
        label = fn.__name__ if not a else f"{fn.__name__}{a}"
        try:
            fn(*a)
            print("PASS", label)
            return True
        except RuntimeError as exc:
            if "CUDA GPU required" in str(exc):
                print("SKIP", label)
                return True
            print("FAIL", label)
            traceback.print_exc()
            return False
        except Exception:
            print("FAIL", label)
            traceback.print_exc()
            return False

    ok = True
    ok &= _run(test_eri_bytes_as300)
    ok &= _run(test_fits_resident_matches_ao2mo_rule)
    ok &= _run(test_eri_mvp_tiled_matches_full_numpy)
    ok &= _run(test_estimate_eri_tile_forced)
    ok &= _run(test_auto_selects_tiled_when_vram_too_small)
    ok &= _run(test_auto_selects_resident_when_vram_is_plenty)
    for multi in ("s", "t"):
        for ntri in (1, 7, 16):
            for tile in (1, 5, 64):
                ok &= _run(test_tiled_mvp_matches_resident_and_cpu, multi, ntri, tile)
    raise SystemExit(0 if ok else 1)
