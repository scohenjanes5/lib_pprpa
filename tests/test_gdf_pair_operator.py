"""The pair-space operator built from three-index factors (gdf_pair_operator) against
the one converted from their packed Gram blocks and the CPU use_eri products."""

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
