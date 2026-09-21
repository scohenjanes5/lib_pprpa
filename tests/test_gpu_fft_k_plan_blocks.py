"""``gpu_fft_k._plan_blocks``: balanced rank chunks and the row-batching gate.

Ceil-dividing the rank by a memory cap just under it leaves a runt batch --
rank 300 under a cap of 269 gives 269 + 31, which is what the NV216 relaxed
density actually ran.  The planner now spreads the same number of chunks evenly
(2 x 150).  The identity ceil(r / ceil(r/c)) <= c means this never exceeds the
memory cap and never adds a chunk, so it is free.

The module imports cupy, so these need a GPU present, but the planner itself is
pure arithmetic and is driven here with an explicit ``free``.
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


# NV216 relaxed-density numbers, measured in job 26659127
NAO, RANK, NGRID = 2795, 300, 159 ** 3
FREE_AT_PLAN = 81_600_000_000   # 81.6 GB free with ao + phiL/phiR resident
MESH = np.array([159, 159, 159])


def _n_el(free, ngrid, mesh=None, reserve_frac=0.15, bytes_per_el=64):
    """The planner's element budget: memory, then the cuFFT batched-plan cap.

    The second term matters here -- on 159^3 the relaxed cap is
    1.5 * 2^31 / ngrid = 801 elements, so it binds before memory does on any
    large GPU, and row_blk tops out at 801 // rank = 2 however much VRAM there
    is.  Leaving it out of the expectation is what made an earlier version of
    this test fail.
    """
    from lib_pprpa.gpu_mem import max_fft_batch
    reserve = max(1024 ** 3, int(free * reserve_frac))
    n_el = max(1, (max(0, free - reserve)) // (bytes_per_el * ngrid))
    return min(n_el, max_fft_batch(ngrid, mesh=mesh))


def test_nv216_rank_chunks_are_balanced():
    """The production case: 269 + 31 becomes 150 + 150, same chunk count."""
    _need_cupy()
    from lib_pprpa.gpu_fft_k import _plan_blocks

    cap = _n_el(FREE_AT_PLAN, NGRID, MESH)
    assert cap == 269, f"planner budget drifted: n_el={cap}"

    rb, kb = _plan_blocks(NAO, RANK, NGRID, free=FREE_AT_PLAN, mesh=MESH)
    assert kb == 150, f"rank_blk={kb}, expected a balanced 150"
    assert kb <= cap, "balanced chunk must still fit the memory cap"
    assert -(-RANK // kb) == -(-RANK // cap) == 2, "chunk count must not change"
    # 269 + 31 becomes 150 + 150: the runt is gone
    assert RANK - kb * (-(-RANK // kb) - 1) == 150
    assert RANK - cap * (-(-RANK // cap) - 1) == 31, "old planner left a 31-wide batch"
    assert rb == 1, "row batching still off at this budget"


def test_row_batching_path_is_untouched():
    """When the whole rank fits, rank_blk must stay == rank so row_blk can grow."""
    _need_cupy()
    from lib_pprpa.gpu_fft_k import _plan_blocks

    for free in (200_000_000_000, 400_000_000_000, 10 ** 12):
        cap = _n_el(free, NGRID, MESH)
        if cap < RANK:
            continue
        rb, kb = _plan_blocks(NAO, RANK, NGRID, free=free, mesh=MESH)
        assert kb == RANK, f"rank_blk={kb} != rank at free={free/1e9:.0f} GB"
        assert -(-RANK // kb) == 1, "one rank chunk"
        assert rb == max(1, min(NAO, cap // RANK)), "row_blk formula unchanged"
        assert rb == 2, "the cuFFT cap holds row_blk at 2 on 159^3, whatever the VRAM"


def test_balancing_is_never_worse_than_ceil_division():
    """Across ranks and budgets: same chunk count, smaller max, larger runt.

    Balancing cannot be stated as "within one element" -- ceil division leaves a
    last chunk up to nchunk-1 short (rank 31, cap 11 -> 11, 11, 9).  What it does
    guarantee is that the largest batch is the smallest it can be for that chunk
    count, and the smallest batch never shrinks.
    """
    _need_cupy()
    from lib_pprpa.gpu_fft_k import _plan_blocks

    for rank in (1, 2, 3, 7, 16, 31, 64, 150, 299, 300, 301, 600):
        for free in (2 * 1024 ** 3, 5 * 10 ** 9, 2 * 10 ** 10, 8 * 10 ** 10,
                     1.5 * 10 ** 11, 3 * 10 ** 11):
            free = int(free)
            cap = max(1, min(rank, _n_el(free, NGRID, MESH)))
            _rb, kb = _plan_blocks(NAO, rank, NGRID, free=free, mesh=MESH)
            n_old, n_new = -(-rank // cap), -(-rank // kb)
            where = (rank, free, cap, kb)
            assert 1 <= kb <= cap, where                    # memory safe
            assert n_new == n_old, where                    # no extra chunk
            assert kb == -(-rank // n_new), where           # minimal max chunk
            last_old = rank - cap * (n_old - 1)
            last_new = rank - kb * (n_new - 1)
            assert 0 < last_new <= kb, where
            assert last_new >= last_old, where              # runt never shrinks


def test_forced_blocks_are_not_rebalanced():
    """An explicit rank_blk is an override, not a hint."""
    _need_cupy()
    from lib_pprpa.gpu_fft_k import _plan_blocks

    rb, kb = _plan_blocks(NAO, RANK, NGRID, row_blk=3, rank_blk=269,
                          free=FREE_AT_PLAN, mesh=MESH)
    assert (rb, kb) == (3, 269)
    # rank_blk alone, row_blk derived
    _rb, kb = _plan_blocks(NAO, RANK, NGRID, rank_blk=269, free=FREE_AT_PLAN,
                           mesh=MESH)
    assert kb == 269, "a forced rank_blk must survive untouched"


def test_telemetry_reports_the_planner_state():
    """A production log must be enough to answer the row-batching question."""
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
    for key in ("rank", "row_blk", "rank_blk", "rank_chunks", "row_batching",
                "free_at_plan_bytes"):
        assert key in s, f"telemetry missing {key}"
    assert s["rank_chunks"] == -(-s["rank"] // s["rank_blk"])
    assert s["row_batching"] == (s["row_blk"] > 1)
    assert s["free_at_plan_bytes"] > 0


if __name__ == "__main__":
    test_nv216_rank_chunks_are_balanced()
    print("nv216 balanced: ok")
    test_row_batching_path_is_untouched()
    print("row-batching path: ok")
    test_balancing_is_never_worse_than_ceil_division()
    print("cap/chunk-count invariants: ok")
    test_forced_blocks_are_not_rebalanced()
    print("overrides: ok")
    test_telemetry_reports_the_planner_state()
    print("telemetry: ok")
