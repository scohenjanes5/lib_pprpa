"""``gpu_fft_k._plan_blocks``: FFT batch and row block sized separately.

The old planner sized one quantity -- the FFT batch (row_blk x rank_blk
transforms, 64 B per point) -- and derived everything from it: the cuFFT plan
cap held row_blk at 1-2 on 159^3 whatever the VRAM, and the final contraction
inherited that shape.  Now ``fft_blk`` (codensities per transform batch, from
the chain's own per-point cost) and ``row_blk`` (rows of the accumulator U per
task) are chosen independently; the balanced within-row chunks are kept
(300 under a cap of 269 -> 150 + 150, never 269 + 31).

The module imports cupy, so these need a GPU present, but the planner itself
is pure arithmetic and is driven here with an explicit ``free``.
"""
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


# NV216 relaxed-density numbers (job 26659127): the AO grid + phiL/phiR resident
NAO, RANK, NGRID = 2795, 300, 159 ** 3
NACT = 600
FREE_AT_PLAN = 81_600_000_000
MESH = np.array([159, 159, 159])


def test_fft_batch_is_capped_and_shaped():
    _need_cupy()
    from lib_pprpa.gpu_coulomb import FFT_BYTES_R2C, FFT_BYTES_C2C
    from lib_pprpa.gpu_fft_k import _plan_blocks
    from lib_pprpa.gpu_mem import max_fft_batch
    for rfft, bpp in ((True, FFT_BYTES_R2C), (False, FFT_BYTES_C2C)):
        rb, fblk = _plan_blocks(NAO, RANK, NGRID, free=FREE_AT_PLAN, mesh=MESH, rfft=rfft)
        assert fblk <= max_fft_batch(NGRID, mesh=MESH)
        assert fblk * bpp * NGRID <= 0.5 * FREE_AT_PLAN          # within the FFT share
        if fblk >= RANK:
            assert fblk % RANK == 0                              # whole rows
        else:
            nchunk = -(-RANK // fblk)
            assert fblk == -(-RANK // nchunk)                    # balanced chunks
        assert 1 <= rb <= NAO


def test_nv216_within_row_chunks_are_balanced_and_rows_batch():
    """At 81.6 GB free the R2C chain affords a batch below one row (300), so the
    row splits into balanced chunks -- and, unlike before, the row block is no
    longer pinned to 1 by that."""
    _need_cupy()
    from lib_pprpa.gpu_fft_k import _plan_blocks, _ROW_BLK_CAP
    rb, fblk = _plan_blocks(NAO, RANK, NGRID, free=FREE_AT_PLAN, mesh=MESH, rfft=True)
    assert fblk < RANK
    assert fblk == 150 and -(-RANK // fblk) == 2, (rb, fblk)
    assert rb > 1 and rb <= _ROW_BLK_CAP
    # ket path: 600 active columns instead of 2795 AO rows, still >= 8 tasks
    rb_k, _f = _plan_blocks(NACT, RANK, NGRID, free=FREE_AT_PLAN, mesh=MESH, rfft=True)
    assert 1 <= rb_k <= -(-NACT // 8)


def test_whole_row_batches_when_memory_allows():
    _need_cupy()
    from lib_pprpa.gpu_fft_k import _plan_blocks
    small = 10 ** 6           # 100^3-ish grid: one row of 300 transforms is 12 GB (R2C)
    rb, fblk = _plan_blocks(NAO, RANK, small, free=200_000_000_000, mesh=np.array([100] * 3),
                            rfft=True)
    assert fblk >= RANK and fblk % RANK == 0
    assert rb >= fblk // RANK                    # one FFT batch never spans two tasks


def test_balancing_is_never_worse_than_ceil_division():
    _need_cupy()
    from lib_pprpa.gpu_coulomb import balanced_chunk
    for rank in (1, 2, 3, 7, 16, 31, 64, 150, 299, 300, 301, 600):
        for cap in (1, 2, 5, 11, 100, 149, 150, 269, 299, 300, 10 ** 6):
            c = balanced_chunk(rank, cap)
            cap_eff = max(1, min(rank, cap))
            n_old, n_new = -(-rank // cap_eff), -(-rank // c)
            assert 1 <= c <= cap_eff, (rank, cap, c)
            assert n_new == n_old, (rank, cap, c)
            assert c == -(-rank // n_new), (rank, cap, c)
            assert rank - c * (n_new - 1) >= rank - cap_eff * (n_old - 1)


def test_forced_blocks_are_not_rebalanced():
    _need_cupy()
    from lib_pprpa.gpu_fft_k import _plan_blocks
    rb, fblk = _plan_blocks(NAO, RANK, NGRID, row_blk=3, rank_blk=269, free=FREE_AT_PLAN, mesh=MESH)
    assert rb == 3 and fblk == 269, (rb, fblk)          # legacy rank_blk == within-row fft_blk
    rb, fblk = _plan_blocks(NAO, RANK, NGRID, row_blk=7, fft_blk=900, free=FREE_AT_PLAN, mesh=MESH)
    assert rb == 7 and fblk == 900                       # three whole rows per batch


def test_telemetry_reports_the_planner_state():
    _need_cupy()
    from pyscf.pbc import gto
    from lib_pprpa.gpu_fft_k import get_k_lowrank, get_last_telemetry

    a0 = 3.370137329
    cell = gto.M(
        atom=[["C", [0.0, 0.0, 0.0]], ["C", [a0 / 2, a0 / 2, a0 / 2]]],
        a=np.array([[0, a0, a0], [a0, 0, a0], [a0, a0, 0]]),
        unit="bohr", basis="gth-dzvp", pseudo="gth-pade", verbose=0)
    cell.mesh = [21, 21, 21]
    cell.build()
    rng = np.random.default_rng(11)
    L = rng.standard_normal((cell.nao, 5))
    R = rng.standard_normal((cell.nao, 5))
    get_k_lowrank(cell, cell.mesh, (L, R), verbose=False)
    s = get_last_telemetry()["sets"][0]
    for key in ("rank", "row_blk", "fft_blk", "rank_blk", "rank_chunks", "row_batching",
                "free_at_plan_bytes", "transforms", "rfft"):
        assert key in s, f"telemetry missing {key}"
    assert s["rank_chunks"] == -(-s["rank"] // s["rank_blk"])
    assert s["row_batching"] == (s["row_blk"] > 1)
    assert s["transforms"] == cell.nao * 5
    assert s["free_at_plan_bytes"] > 0


if __name__ == "__main__":
    test_fft_batch_is_capped_and_shaped()
    print("fft batch cap/shape: ok")
    test_nv216_within_row_chunks_are_balanced_and_rows_batch()
    print("nv216 balanced chunks + row batching: ok")
    test_whole_row_batches_when_memory_allows()
    print("whole-row batches: ok")
    test_balancing_is_never_worse_than_ceil_division()
    print("balanced_chunk invariants: ok")
    test_forced_blocks_are_not_rebalanced()
    print("overrides: ok")
    test_telemetry_reports_the_planner_state()
    print("telemetry: ok")
