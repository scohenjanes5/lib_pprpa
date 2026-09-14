"""Opt-in multi-GPU dispatch for the lib_pprpa strip / batch kernels.

The GPU kernels in this package (``gpu_ao2mo`` pair strips, ``gpu_fft_k`` AO
row blocks, ``gpu_pairing_force`` codensity strips, the per-atom hcore
derivative) all have the same shape: a read-only set of arrays, a list of
independent tasks with *global* boundaries, and outputs that are either
written to disjoint index ranges or summed.  ``DeviceGroup`` runs such loops
on one or several GPUs from a single process:

* one Python thread per slot, each inside ``with cp.cuda.Device(id)`` (the
  idiom gpu4pyscf's own ``lib/multi_gpu.py`` uses; we do not call its
  ``run``/``map``/``reduce`` because they always take every visible device,
  cannot run two "virtual" slots on one GPU for tests, and have ``*kwargs`` /
  ``for i in num_devices`` bugs);
* a shared, lock-guarded task iterator (dynamic load balancing);
* per-slot OOM handling: the slot shrinks its own sub-block (``shrink``) and
  re-runs the *same* task, so task boundaries never depend on the slot count;
* a degenerate single-slot path that runs inline in the calling thread with no
  ``Device`` switch, so 1-GPU behaviour is unchanged.

Opt-in: the device count comes from ``LIB_PPRPA_GPUS`` (default 1), clipped to
the visible devices.  The 2-GPU Slurm scripts export ``LIB_PPRPA_GPUS=2``; the
1-GPU scripts leave it unset.  Read-only arrays are replicated to the other
slots with ``broadcast`` (chunked ``cudaMemcpyPeer`` when peer access exists,
else a pinned-host bounce); rank 0 aliases the original.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cupy as cp

from lib_pprpa.pprpa_util import tstamp


_PRINT_LOCK = threading.Lock()


def log(msg):
    """Timestamped, lock-protected print (two device threads never interleave)."""
    with _PRINT_LOCK:
        print(f"{tstamp()} {msg}", flush=True)


# --------------------------------------------------------------------------
# current-device helpers (formerly in gpu_ao2mo; re-exported from there)
# --------------------------------------------------------------------------
def _sync():
    cp.cuda.Device().synchronize()


def _free_pool():
    """Return unused CuPy blocks to the driver."""
    cp.get_default_memory_pool().free_all_blocks()
    try:
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _clear_fft_cache():
    """cuFFT plan cache retains workspace across probes and starves later allocs."""
    try:
        cp.fft.config.get_plan_cache().clear()
    except Exception:
        pass


def _reclaim_gpu():
    """Drop FFT plans + pooled blocks so memGetInfo reflects truly free VRAM."""
    _clear_fft_cache()
    _free_pool()
    try:
        cp.cuda.Device().synchronize()
    except Exception:
        pass


def _free_bytes():
    """Driver-reported free VRAM of the current device (pool holdings count as used).

    Do NOT use gpu4pyscf get_avail_mem here: it can report a stale/constant
    value that ignores live CuPy allocations.  Call ``_reclaim_gpu`` first if
    you need free after FFT probes.
    """
    free, _total = cp.cuda.runtime.memGetInfo()
    return int(free)


def _is_oom(exc):
    if isinstance(exc, cp.cuda.memory.OutOfMemoryError):
        return True
    msg = f"{type(exc).__name__}: {exc}".lower()
    return ("outofmemory" in msg
            or "memory allocation" in msg
            or "cudaerrormemoryallocation" in msg
            # cuFFT plan too large for the 32-bit Bluestein path / no workspace:
            # both are cured by a smaller batch, so retry like an OOM.
            or "cufft_invalid_size" in msg
            or "cufft_alloc_failed" in msg)


# --------------------------------------------------------------------------
# device discovery
# --------------------------------------------------------------------------
def visible_device_count():
    try:
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def device_ids(max_devices=None, visible=None):
    """Slots to use: ``LIB_PPRPA_GPUS`` (default 1) clipped to the visible devices.

    ``visible`` overrides the CUDA query (tests without a GPU).
    """
    if visible is None:
        visible = visible_device_count()
    if max_devices is None:
        raw = os.environ.get("LIB_PPRPA_GPUS", "").strip()
        max_devices = int(raw) if raw else 1
    n = max(1, min(int(max_devices), max(int(visible), 1)))
    return list(range(n))


def describe_devices(devices=None):
    """Human-readable device summary (name, VRAM, peer-access matrix)."""
    if devices is None:
        devices = device_ids()
    lines = []
    for d in devices:
        try:
            props = cp.cuda.runtime.getDeviceProperties(d)
            name = props["name"]
            name = name.decode() if isinstance(name, bytes) else str(name)
            with cp.cuda.Device(d):
                free, total = cp.cuda.runtime.memGetInfo()
            lines.append(f"  device {d}: {name}  free/total = {free/1e9:.1f}/{total/1e9:.1f} GB")
        except Exception as exc:  # pragma: no cover - diagnostics only
            lines.append(f"  device {d}: <unavailable: {exc}>")
    if len(devices) > 1:
        peers = []
        for a in devices:
            for b in devices:
                if a != b:
                    try:
                        ok = cp.cuda.runtime.deviceCanAccessPeer(a, b)
                    except Exception:
                        ok = 0
                    peers.append(f"{a}->{b}:{'yes' if ok else 'no'}")
        lines.append("  peer access: " + ", ".join(peers))
    return (f"[gpu_multi] LIB_PPRPA_GPUS={os.environ.get('LIB_PPRPA_GPUS', '<unset>')} "
            f"visible={visible_device_count()} using {len(devices)} slot(s) {devices}\n"
            + "\n".join(lines))


def configure_pool(fraction=0.9):
    """Cap the current device's CuPy pool (gpu4pyscf does this only for device 0)."""
    try:
        cp.get_default_memory_pool().set_limit(fraction=fraction)
    except Exception:
        pass


# --------------------------------------------------------------------------
# replication
# --------------------------------------------------------------------------
def copy_to_current_device(src, chunk_bytes=2 ** 31):
    """Copy ``src`` (numpy or cupy on any device) onto the current device.

    Same-device cupy input returns a *copy* (callers alias explicitly when they
    want to).  Cross-device copies use ``cudaMemcpyPeer`` in chunks when peer
    access is available, else bounce through the host in chunks.
    """
    cur = cp.cuda.Device().id
    if isinstance(src, np.ndarray):
        return cp.asarray(src)
    src_dev = src.device.id
    if src_dev == cur:
        return src.copy()
    with cp.cuda.Device(src_dev):
        src = cp.ascontiguousarray(src)
    dst = cp.empty(src.shape, dtype=src.dtype)
    try:
        peer = int(cp.cuda.runtime.deviceCanAccessPeer(cur, src_dev))
    except Exception:
        peer = 0
    nbytes = src.nbytes
    if peer and nbytes:
        off = 0
        while off < nbytes:
            n = min(int(chunk_bytes), nbytes - off)
            cp.cuda.runtime.memcpyPeer(dst.data.ptr + off, cur,
                                       src.data.ptr + off, src_dev, n)
            off += n
    elif nbytes:
        flat_src = src.ravel()
        flat_dst = dst.ravel()
        step = max(1, int(chunk_bytes) // src.dtype.itemsize)
        for i in range(0, flat_src.size, step):
            with cp.cuda.Device(src_dev):
                host = cp.asnumpy(flat_src[i:i + step])
            flat_dst[i:i + step] = cp.asarray(host)
    cp.cuda.Device().synchronize()
    return dst


# --------------------------------------------------------------------------
# device group
# --------------------------------------------------------------------------
class DeviceCtx:
    """Per-slot context handed to every callback (runs on ``device_id``)."""

    def __init__(self, rank, device_id, nslots):
        self.rank = int(rank)
        self.device_id = int(device_id)
        self.nslots = int(nslots)
        self.state = {}
        self.stats = {"tasks": 0, "busy_seconds": 0.0, "retries": 0,
                      "min_free_bytes": None}

    def log(self, msg):
        prefix = f"[dev{self.device_id} r{self.rank}] " if self.nslots > 1 else ""
        log(prefix + msg)


class DeviceGroup:
    """A fixed set of device slots; see the module docstring."""

    def __init__(self, devices=None, label="gpu_multi"):
        if devices is None:
            devices = device_ids()
        self.devices = [int(d) for d in devices]
        assert self.devices, "DeviceGroup needs at least one slot"
        self.label = label
        self.ctxs = [DeviceCtx(r, d, len(self.devices)) for r, d in enumerate(self.devices)]
        if not self.inline:
            self.each(lambda ctx: configure_pool())

    @property
    def nslots(self):
        return len(self.devices)

    @property
    def inline(self):
        return self.nslots == 1

    # -- internals ---------------------------------------------------------
    def _on_slot(self, ctx, fn):
        if self.inline:
            return fn(ctx)
        with cp.cuda.Device(ctx.device_id):
            return fn(ctx)

    # -- API ---------------------------------------------------------------
    def each(self, fn):
        """Run ``fn(ctx)`` once per slot on its device; results ordered by rank."""
        if self.inline:
            return [fn(self.ctxs[0])]
        with ThreadPoolExecutor(max_workers=self.nslots) as ex:
            futs = [ex.submit(self._on_slot, ctx, fn) for ctx in self.ctxs]
            return [f.result() for f in futs]

    def broadcast(self, arr, key):
        """``ctx.state[key]`` = ``arr`` on rank 0 (alias), a device copy elsewhere."""
        def _bc(ctx):
            if ctx.rank == 0:
                ctx.state[key] = arr if isinstance(arr, cp.ndarray) else cp.asarray(arr)
            else:
                ctx.state[key] = copy_to_current_device(arr)
        self.each(_bc)

    def min_free_bytes(self):
        return min(self.each(lambda ctx: _free_bytes()))

    def free(self, keys=None):
        """Drop ``state`` entries (all when ``keys`` is None) and reclaim VRAM per slot."""
        def _free(ctx):
            ks = list(ctx.state) if keys is None else list(keys)
            for k in ks:
                ctx.state.pop(k, None)
            _reclaim_gpu()
        self.each(_free)

    def run(self, tasks, work, shrink=None, label=""):
        """Dispatch ``tasks`` over the slots.

        ``work(ctx, task)`` runs on the slot's device and returns the task
        result (or None).  On an OOM-class exception the slot reclaims VRAM,
        calls ``shrink(ctx, task, exc)`` and re-runs the same task if that
        returned True; otherwise a ``MemoryError`` is raised.  Any other
        exception aborts the whole run.  Returns ``(results, stats)`` with
        results in task order.
        """
        tasks = list(tasks)
        results = [None] * len(tasks)
        lock = threading.Lock()
        it = iter(enumerate(tasks))
        abort = threading.Event()
        per_slot = {ctx.rank: {"tasks": 0, "busy_seconds": 0.0, "retries": 0,
                               "min_free_bytes": None} for ctx in self.ctxs}

        def _next():
            with lock:
                return next(it, None)

        def _slot(ctx):
            st = per_slot[ctx.rank]
            while not abort.is_set():
                item = _next()
                if item is None:
                    break
                idx, task = item
                t0 = time.perf_counter()
                while True:
                    try:
                        res = work(ctx, task)
                        _sync()
                        break
                    except Exception as exc:  # noqa: BLE001 - classify below
                        if abort.is_set():
                            raise
                        if shrink is None or not _is_oom(exc):
                            abort.set()
                            raise
                        _reclaim_gpu()
                        st["retries"] += 1
                        ctx.stats["retries"] += 1
                        if not shrink(ctx, task, exc):
                            abort.set()
                            raise MemoryError(
                                f"{self.label}/{label}: device {ctx.device_id} cannot shrink "
                                f"further for task {idx} ({type(exc).__name__}: {exc}); "
                                f"free~{_free_bytes()/1e9:.2f} GB") from exc
                        ctx.log(f"{label}: {type(exc).__name__} on task {idx}; "
                                f"retrying with a smaller block")
                results[idx] = res
                dt = time.perf_counter() - t0
                st["tasks"] += 1
                st["busy_seconds"] += dt
                ctx.stats["tasks"] += 1
                ctx.stats["busy_seconds"] += dt
                fb = _free_bytes()
                st["min_free_bytes"] = fb if st["min_free_bytes"] is None else min(st["min_free_bytes"], fb)
                ctx.stats["min_free_bytes"] = (fb if ctx.stats["min_free_bytes"] is None
                                               else min(ctx.stats["min_free_bytes"], fb))

        started = time.perf_counter()
        if self.inline:
            _slot(self.ctxs[0])
        else:
            with ThreadPoolExecutor(max_workers=self.nslots) as ex:
                futs = [ex.submit(self._on_slot, ctx, _slot) for ctx in self.ctxs]
                first_exc = None
                for f in futs:
                    try:
                        f.result()
                    except Exception as exc:  # noqa: BLE001
                        first_exc = first_exc or exc
                if first_exc is not None:
                    raise first_exc
        stats = {
            "label": label,
            "devices": list(self.devices),
            "nslots": self.nslots,
            "ntasks": len(tasks),
            "wall_seconds": time.perf_counter() - started,
            "tasks_per_slot": [per_slot[c.rank]["tasks"] for c in self.ctxs],
            "busy_seconds_per_slot": [per_slot[c.rank]["busy_seconds"] for c in self.ctxs],
            "retries_per_slot": [per_slot[c.rank]["retries"] for c in self.ctxs],
            "min_free_bytes_per_slot": [per_slot[c.rank]["min_free_bytes"] for c in self.ctxs],
        }
        return results, stats


_DEFAULT = [None]


def default_group():
    """Process-wide group built from ``LIB_PPRPA_GPUS`` (created on first use)."""
    if _DEFAULT[0] is None:
        _DEFAULT[0] = DeviceGroup(device_ids())
    return _DEFAULT[0]


def set_default_group(group):
    _DEFAULT[0] = group


def sum_results(results):
    """Sum a list of per-task partial arrays (numpy or cupy), skipping None."""
    total = None
    for r in results:
        if r is None:
            continue
        total = r if total is None else total + r
    return total
