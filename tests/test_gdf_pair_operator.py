"""The pair-space operator built from three-index factors (gdf_pair_operator, and
pair_operator_from_factors from device factor blocks) against the one converted from their
packed Gram blocks and the CPU use_eri products."""

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


def _factors(rng, naux, n):
    L = rng.standard_normal((naux, n, n))
    return L + L.transpose(0, 2, 1)


def _physicist(L, nocc):
    """vvvv, oovv, oooo with <pq|rs> = (pr|qs) = sum_L L[L, p, r] L[L, q, s]."""
    phys = np.einsum("Lpr,Lqs->pqrs", L, L)
    o, v = slice(0, nocc), slice(nocc, None)
    return (np.ascontiguousarray(phys[v, v, v, v]), np.ascontiguousarray(phys[o, o, v, v]),
            np.ascontiguousarray(phys[o, o, o, o]))


def _solver(nocc, mo_energy, multi, eri=None):
    from lib_pprpa.pprpa_davidson import ppRPA_Davidson
    mp = ppRPA_Davidson(nocc, mo_energy, Lpq=None, channel="hh", nroot=2,
                        residue_thresh=1e-10, trial="identity")
    mp.mu = 0.0
    mp.multi = multi
    if eri is not None:
        mp.use_eri(*eri)
    mp.check_parameter()
    return mp


@pytest.mark.parametrize("multi", ("s", "t"))
@pytest.mark.parametrize("rows, pair_blk", ((None, None), (2, 3), (1, 40)))
def test_gdf_operator_matches_gram_path(multi, rows, pair_blk):
    import cupy as cp
    from lib_pprpa.pprpa_davidson import _pprpa_contraction
    from lib_pprpa.pair_operator import pack_eri
    from lib_pprpa.pprpa_eri_gpu import (attach_gpu_eri_contraction, gdf_pair_operator,
                                         release_gpu_eri)
    rng = np.random.default_rng(3)
    naux, nocc, nvir = 30, 5, 7
    L = _factors(rng, naux, nocc + nvir)
    moe = np.sort(rng.standard_normal(nocc + nvir))
    eri = _physicist(L, nocc)
    cpu = _solver(nocc, moe, multi, eri=eri)
    tv = rng.standard_normal((4, cpu.full_dim))
    expect = _pprpa_contraction(cpu, tv)

    gram = _solver(nocc, moe, multi)
    attach_gpu_eri_contraction(gram, *pack_eri(*eri), rows=rows)
    from_gram = gram.contraction(tv)
    release_gpu_eri(gram)
    np.testing.assert_allclose(from_gram, expect, rtol=1e-11, atol=1e-11)

    for lpq in (L, cp.asarray(L)):
        for resident, kind in ((False, np.ndarray), (True, cp.ndarray)):
            op = gdf_pair_operator(lpq, nocc, multi, rows=rows, pair_blk=pair_blk,
                                   resident=resident)
            assert op.multi == multi and all(isinstance(s[3], kind) for s in op.strips)
            gpu = _solver(nocc, moe, multi)
            attach_gpu_eri_contraction(gpu, op)
            if resident:
                assert gpu._gpu_eri["mode"] == "resident"
            np.testing.assert_allclose(gpu.contraction(tv), from_gram, rtol=1e-12, atol=1e-12)
            release_gpu_eri(gpu)


def test_gdf_operator_under_a_tight_budget(monkeypatch):
    """With no free memory the strips stay in pinned host memory, the factors go up one
    aux function at a time and every Gram strip is one row (so every operator strip is
    one orbital); the products are unchanged."""
    import cupy as cp
    from lib_pprpa import pprpa_eri_gpu as mod
    rng = np.random.default_rng(5)
    naux, nocc, nvir = 12, 4, 6
    L = _factors(rng, naux, nocc + nvir)
    moe = np.sort(rng.standard_normal(nocc + nvir))
    ref = mod.gdf_pair_operator(L, nocc, "s")
    assert all(isinstance(s[3], cp.ndarray) for s in ref.strips)
    monkeypatch.setattr(mod, "free_bytes", lambda: 0)
    tight = mod.gdf_pair_operator(L, nocc, "s")
    assert all(isinstance(s[3], np.ndarray) for s in tight.strips)
    assert all(R1 - R0 <= nvir for _, R0, R1, _ in tight.strips)
    assert len(tight.strips) > len(ref.strips)
    monkeypatch.undo()
    out = []
    for op in (ref, tight):
        gpu = _solver(nocc, moe, "s")
        tv = np.random.default_rng(7).standard_normal((3, gpu.full_dim))
        mod.attach_gpu_eri_contraction(gpu, op)
        out.append(gpu.contraction(tv))
        mod.release_gpu_eri(gpu)
    np.testing.assert_allclose(out[1], out[0], rtol=1e-12, atol=1e-12)


def test_pair_factors_layout():
    """gdf_pair_factors stores F[P, L] = lpq[L, p, q] in the documented pair orders."""
    import cupy as cp
    from lib_pprpa.pprpa_eri_gpu import gdf_pair_factors
    rng = np.random.default_rng(9)
    naux, nocc, nvir = 6, 3, 4
    L = _factors(rng, naux, nocc + nvir)
    F = gdf_pair_factors(L, nocc)
    assert sorted(F) == ["oo", "ov", "vv"]
    assert all(isinstance(f, cp.ndarray) and f.flags.c_contiguous for f in F.values())
    a, b = np.tril_indices(nvir)
    i, j = np.tril_indices(nocc)
    io, av = np.divmod(np.arange(nocc * nvir), nvir)
    np.testing.assert_array_equal(F["vv"].get(), L[:, nocc + a, nocc + b].T)
    np.testing.assert_array_equal(F["oo"].get(), L[:, i, j].T)
    np.testing.assert_array_equal(F["ov"].get(), L[:, io, nocc + av].T)


@pytest.mark.parametrize("multi", ("s", "t"))
@pytest.mark.parametrize("pair_blk, resident", ((None, None), (3, False)))
def test_operator_from_device_factors(multi, pair_blk, resident):
    """pair_operator_from_factors gives the operator of gdf_pair_operator and lets every
    block go once it is used: afterwards the pool holds the device strips, not the factors."""
    import cupy as cp
    from lib_pprpa import pprpa_eri_gpu as mod
    rng = np.random.default_rng(11)
    naux, nocc, nvir = 4000, 4, 6                # factors 1.8 MB, well above pool rounding
    L = _factors(rng, naux, nocc + nvir)
    moe = np.sort(rng.standard_normal(nocc + nvir))
    pool = cp.get_default_memory_pool()
    factors = mod.gdf_pair_factors(L, nocc)
    held = sum(f.nbytes for f in factors.values())
    before = pool.used_bytes()
    op = mod.pair_operator_from_factors(factors, nocc, multi, pair_blk=pair_blk,
                                        resident=resident)
    assert factors == {}
    on_device = sum(s[3].nbytes for s in op.strips if isinstance(s[3], cp.ndarray))
    assert pool.used_bytes() <= before - held + on_device + 64 * 1024
    if resident is False:
        assert on_device == 0
    out = []
    for o in (mod.gdf_pair_operator(L, nocc, multi, pair_blk=pair_blk), op):
        gpu = _solver(nocc, moe, multi)
        tv = np.random.default_rng(13).standard_normal((3, gpu.full_dim))
        mod.attach_gpu_eri_contraction(gpu, o)
        out.append(gpu.contraction(tv))
        mod.release_gpu_eri(gpu)
    np.testing.assert_allclose(out[1], out[0], rtol=1e-13, atol=1e-13)


def test_operator_from_factors_rejects_other_layouts():
    """A wrong key set, block shape, dtype, memory order or host array is refused before
    any block is taken from the dict."""
    import cupy as cp
    from lib_pprpa.pprpa_eri_gpu import gdf_pair_factors, pair_operator_from_factors
    rng = np.random.default_rng(15)
    nocc = 3
    good = gdf_pair_factors(_factors(rng, 5, nocc + 4), nocc)
    bad = [{k: v for k, v in good.items() if k != "oo"},
           dict(good, vv=good["vv"][:-1]),
           dict(good, oo=good["oo"].astype(cp.float32)),
           dict(good, ov=cp.asfortranarray(good["ov"])),
           dict(good, vv=good["vv"].get())]
    for factors in bad:
        keys = sorted(factors)
        with pytest.raises(ValueError):
            pair_operator_from_factors(factors, nocc, "s")
        assert sorted(factors) == keys
    with pytest.raises(ValueError):
        pair_operator_from_factors(dict(good), nocc, "x")
