"""DeviceGroup dispatcher: opt-in device count, ordering, OOM shrink-and-retry,
error propagation, and a virtual two-slot group on one GPU.

Run on a GPU node (``python tests/test_gpu_multi.py``) or via pytest.
"""
from __future__ import annotations

import os

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
    """Two real devices when the allocation has them, else two virtual slots on GPU 0."""
    import cupy as _cp
    return [0, 1] if _cp.cuda.runtime.getDeviceCount() >= 2 else [0, 0]


def test_device_ids_opt_in(monkeypatch=None):
    _need_cupy()
    from lib_pprpa.gpu_multi import device_ids
    saved = os.environ.pop("LIB_PPRPA_GPUS", None)
    try:
        assert device_ids(visible=4) == [0]                 # default: one slot
        os.environ["LIB_PPRPA_GPUS"] = "2"
        assert device_ids(visible=4) == [0, 1]
        os.environ["LIB_PPRPA_GPUS"] = "8"
        assert device_ids(visible=2) == [0, 1]               # clipped to visible
        os.environ["LIB_PPRPA_GPUS"] = "2"
        assert device_ids(visible=0) == [0]                  # never empty
        assert device_ids(max_devices=3, visible=4) == [0, 1, 2]
    finally:
        os.environ.pop("LIB_PPRPA_GPUS", None)
        if saved is not None:
            os.environ["LIB_PPRPA_GPUS"] = saved


def _square_work(ctx, task):
    import cupy as cp
    a = cp.arange(task * 10, task * 10 + 5, dtype=cp.float64)
    return float((a * a).sum())


def test_run_ordering_inline_and_virtual_two_slots():
    _need_cupy()
    from lib_pprpa.gpu_multi import DeviceGroup
    tasks = list(range(37))
    ref = [_square_work(None, t) for t in tasks]
    g1 = DeviceGroup([0])
    assert g1.inline
    r1, s1 = g1.run(tasks, _square_work, label="sq")
    assert r1 == ref and s1["nslots"] == 1 and s1["tasks_per_slot"] == [37]
    g2 = DeviceGroup(_two_slots())
    assert not g2.inline and g2.nslots == 2
    r2, s2 = g2.run(tasks, _square_work, label="sq")
    assert r2 == ref
    assert sum(s2["tasks_per_slot"]) == 37 and s2["retries_per_slot"] == [0, 0]


def test_oom_shrink_and_retry_same_task():
    _need_cupy()
    from lib_pprpa.gpu_multi import DeviceGroup
    FakeOOM = type("OutOfMemoryError", (Exception,), {})
    calls = []

    def work(ctx, task):
        calls.append((task, ctx.state["blk"]))
        if ctx.state["blk"] > 2:
            raise FakeOOM("fake")
        return task * ctx.state["blk"]

    def shrink(ctx, task, exc):
        if ctx.state["blk"] <= 1:
            return False
        ctx.state["blk"] //= 2
        return True

    for devices in ([0], _two_slots()):
        g = DeviceGroup(devices)
        for ctx in g.ctxs:
            ctx.state["blk"] = 8
        calls.clear()
        results, stats = g.run([1, 2, 3], work, shrink=shrink, label="oom")
        # every task ends up computed with blk == 2 on whichever slot ran it
        assert results == [2, 4, 6], results
        assert sum(stats["retries_per_slot"]) >= 2
        # a task that OOMed was re-run (same task id appears more than once)
        first_task = calls[0][0]
        assert sum(1 for t, _ in calls if t == first_task) >= 2


def test_unshrinkable_oom_raises_memoryerror_and_other_errors_propagate():
    _need_cupy()
    from lib_pprpa.gpu_multi import DeviceGroup
    FakeOOM = type("OutOfMemoryError", (Exception,), {})

    def always_oom(ctx, task):
        raise FakeOOM("no")

    g = DeviceGroup([0])
    try:
        g.run([0, 1], always_oom, shrink=lambda ctx, task, exc: False, label="x")
    except MemoryError:
        pass
    else:
        raise AssertionError("expected MemoryError")

    def bad(ctx, task):
        raise ValueError("boom")

    for devices in ([0], _two_slots()):
        try:
            DeviceGroup(devices).run([0, 1, 2], bad, label="bad")
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError should propagate")


def test_broadcast_alias_vs_copy_and_free():
    _need_cupy()
    import cupy as cp
    from lib_pprpa.gpu_multi import DeviceGroup
    a = cp.arange(1000, dtype=cp.float64)
    g1 = DeviceGroup([0])
    g1.broadcast(a, "a")
    assert g1.ctxs[0].state["a"] is a                      # rank 0 aliases
    g2 = DeviceGroup(_two_slots())
    g2.broadcast(a, "a")
    b0, b1 = g2.ctxs[0].state["a"], g2.ctxs[1].state["a"]
    assert b0 is a and b1 is not a
    assert b1.data.ptr != a.data.ptr
    np.testing.assert_array_equal(cp.asnumpy(b1), cp.asnumpy(a))
    h = np.linspace(0, 1, 50)
    g2.broadcast(h, "h")
    for ctx in g2.ctxs:
        np.testing.assert_array_equal(cp.asnumpy(ctx.state["h"]), h)
    g2.free(["a"])
    assert "a" not in g2.ctxs[1].state and "h" in g2.ctxs[1].state
    g2.free()
    assert g2.ctxs[0].state == {} and g2.ctxs[1].state == {}
    assert g2.min_free_bytes() > 0


if __name__ == "__main__":
    _need_cupy()
    for fn in (test_device_ids_opt_in,
               test_run_ordering_inline_and_virtual_two_slots,
               test_oom_shrink_and_retry_same_task,
               test_unshrinkable_oom_raises_memoryerror_and_other_errors_propagate,
               test_broadcast_alias_vs_copy_and_free):
        fn()
        print(f"OK  {fn.__name__}")
