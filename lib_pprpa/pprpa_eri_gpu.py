"""GPU (cupy) ERI contraction for the pp-RPA Davidson ``use_eri`` path.

Three modes, selected by ``attach_gpu_eri_contraction``:

* **resident** — keep ``vvvv`` / ``oooo`` / ``oovv`` on the GPU and GEMM all
  trial vectors at once (original path).
* **split** — with a multi-slot ``gpu_multi.DeviceGroup``, keep a row range of
  every block resident on each slot and sum the partial products on the host:
  nothing streams per MVP (194 GB at AS=300 otherwise), only the trial
  vectors and the partials (~25 MB each way).  Auto-selected when the whole
  set does not fit one slot but the per-slot shares do.
* **tiled** — keep the tensors on the host and stream ``(blk, n²)`` row strips
  for the same GEMMs.  Auto-selected when neither of the above fits.  The
  strips are *rows*: ``vvvv`` and ``oooo`` are symmetric physicist matrices
  (<ab|cd> = <cd|ab>), so ``V[:, P].T == V[P, :]`` and every upload is a
  contiguous slice of the host array -- no numpy copy of a strided column
  block in front of each transfer.  Host tensors staged in pinned memory by
  ``gpu_ao2mo`` cross the link at full speed; the per-MVP volume, seconds and
  rate are recorded in the telemetry and printed at ``release_gpu_eri``.

Algebra is identical to ``lib_pprpa.pprpa_davidson._pprpa_contraction`` (z.T
flattening, 1/sqrt(2) diagonal scaling, hh-block sign, physicist matmul
reshapes).

    prod_vv = vvvv . z_vv  + oovv^T . z_oo
    prod_oo = oooo . z_oo  + oovv   . z_vv

Validate:  ``python pprpa_eri_gpu.py``  or  ``pytest tests/test_pprpa_eri_gpu.py``
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

from lib_pprpa.gpu_mem import eri_bytes, fits_resident
from lib_pprpa.pprpa_util import tstamp as _ts

_INV_SQRT2 = 1.0 / math.sqrt(2.0)

LAST_TELEMETRY = {}


def get_last_telemetry():
    return dict(LAST_TELEMETRY)


def _require_cupy():
    if cp is None:
        raise ImportError("cupy is required for GPU ERI contraction")
    return cp


def _is_cupy(x):
    if cp is None:
        return False
    try:
        return isinstance(x, cp.ndarray)
    except Exception:
        return False


def _is_oom(exc):
    _require_cupy()
    if isinstance(exc, cp.cuda.memory.OutOfMemoryError):
        return True
    msg = f"{type(exc).__name__}: {exc}".lower()
    return ("outofmemory" in msg
            or "memory allocation" in msg
            or "cudaerrormemoryallocation" in msg)


def _is_pinned(arr):
    """Best-effort: was this numpy array allocated with ``cupyx.empty_pinned``?"""
    base = arr
    while isinstance(base, np.ndarray) and base.base is not None:
        base = base.base
    return "Pinned" in type(base).__name__


def _flatten_host(block, n0, n1):
    """``(n0, n0, n1, n1)`` or ``(n0*n0, n1*n1)`` → C-contiguous ``(n0², n1²)`` numpy.
    A reshape of a pinned array stays pinned (no copy is made)."""
    arr = np.asarray(cp.asnumpy(block) if _is_cupy(block) else block)
    if arr.ndim == 4:
        arr = arr.reshape(n0 * n0, n1 * n1)
    elif arr.ndim != 2:
        raise ValueError(f"ERI block rank {arr.ndim}, expected 2 or 4")
    if arr.shape != (n0 * n0, n1 * n1):
        raise ValueError(f"ERI block shape {arr.shape}, expected {(n0 * n0, n1 * n1)}")
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def eri_mvp_tiled(zvvT, zooT, vvvv, oovv, oooo, tile, xp=np):
    """Tiled ``prod_vv, prod_oo`` matching resident ``z @ ERI``.

    ``zvvT`` is ``(ntri, nv²)``, ``zooT`` is ``(ntri, no²)``.  ERI arguments are
    flattened physicist blocks (numpy or cupy).  ``xp`` is the array module for
    the accumulators (``numpy`` or ``cupy``).  Each inner slice is uploaded with
    ``xp.asarray`` so a numpy host ERI + cupy ``xp`` streams tiles to the GPU.

    Every block crosses the link exactly once per call, always as *row strips*
    (contiguous in the C-ordered host arrays, so no numpy copy precedes an
    upload): ``vvvv`` and ``oooo`` are symmetric physicist matrices, so
    ``V[:, P].T == V[P, :]``; ``oovv`` row strips serve both of its products
    (see below).  That is the whole transfer budget, 194.4 GB at AS=300, and
    the path is transfer-bound (intensity ntri/4 flop per byte), so a redundant
    pass costs real wall time.
    """
    tile = max(1, int(tile))
    ntri, nv2 = zvvT.shape
    no2 = zooT.shape[1]
    prod_vv = xp.zeros((ntri, nv2), dtype=zvvT.dtype)
    prod_oo = xp.zeros((ntri, no2), dtype=zooT.dtype)

    for p0 in range(0, nv2, tile):
        p1 = min(p0 + tile, nv2)
        Vt = xp.asarray(vvvv[p0:p1, :])              # == vvvv[:, p0:p1].T (symmetric)
        prod_vv = prod_vv + zvvT[:, p0:p1] @ Vt
        Vt = None

    for p0 in range(0, no2, tile):
        p1 = min(p0 + tile, no2)
        Ot = xp.asarray(oooo[p0:p1, :])
        prod_oo = prod_oo + zooT[:, p0:p1] @ Ot
        Ot = None

    # One pass over oovv serves both of its products.  A row strip
    # ov = oovv[P, :] gives prod_vv's rank-|P| update directly, and its
    # transpose gives the *disjoint column block* prod_oo[:, P] -- summing over
    # P covers every column exactly once, so no second sweep is needed:
    #
    #     prod_vv     += zooT[:, P] @ oovv[P, :]
    #     prod_oo[:, P] += zvvT      @ oovv[P, :].T
    #
    # Streaming column strips for prod_oo instead (the previous fourth loop)
    # re-read the whole tensor -- 259.2 -> 194.4 GB per MVP at AS=300 -- and,
    # being a strided slice of a C-contiguous host array, also forced numpy to
    # materialise a contiguous copy before every upload.
    for p0 in range(0, no2, tile):
        p1 = min(p0 + tile, no2)
        ov = xp.asarray(oovv[p0:p1, :])
        prod_vv = prod_vv + zooT[:, p0:p1] @ ov
        prod_oo[:, p0:p1] += zvvT @ ov.T
        ov = None

    return prod_vv, prod_oo


def estimate_eri_tile(n2, tile=None):
    """Strip length for one ``(blk, n2)`` float64 tile from free VRAM."""
    n2 = max(1, int(n2))
    if tile is not None:
        return max(1, min(int(tile), n2))
    env = os.environ.get("DAVIDSON_ERI_TILE")
    if env:
        return max(1, min(int(env), n2))
    _require_cupy()
    free, _total = cp.cuda.runtime.memGetInfo()
    cushion = max(512 * 1024 ** 2, int(0.02 * free))
    budget = max(0, int(free) - cushion)
    raw = max(1, int(budget / (8 * n2)))
    return min(raw, n2)


def _choose_mode(nocc, nvir, vvvv, oovv, oooo, mode, total_bytes):
    env = os.environ.get("PPRPA_ERI_MODE")
    if env:
        mode = env.strip().lower()
    mode = (mode or "auto").strip().lower()
    if mode not in ("auto", "resident", "split", "tiled"):
        raise ValueError(f"unknown ERI mode {mode!r}")
    if mode != "auto":
        return mode
    if all(_is_cupy(x) for x in (vvvv, oovv, oooo)):
        return "resident"
    return "resident" if fits_resident(
        nocc, nvir, extra_bytes=0, total_bytes=total_bytes) else "tiled"


def _split_fits(nocc, nvir, group, frac=0.75):
    """Do the per-slot row shares of the three blocks fit each slot's free VRAM?"""
    share = eri_bytes(nocc, nvir) / group.nslots
    try:
        free_min = group.min_free_bytes()
    except Exception:
        return False
    return share <= frac * free_min


def _split_upload(group, vvvv, oovv, oooo, nocc, nvir):
    """Put rows [r0, r1) of every block on each slot (contiguous host slices)."""
    nv2, no2 = nvir * nvir, nocc * nocc
    V = _flatten_host(vvvv, nvir, nvir)
    O = _flatten_host(oooo, nocc, nocc)
    OV = _flatten_host(oovv, nocc, nvir)
    n = group.nslots

    def _rows(total, rank):
        a = (total * rank) // n
        b = (total * (rank + 1)) // n
        return a, b

    def _up(ctx):
        st = ctx.state
        v0, v1 = _rows(nv2, ctx.rank)
        o0, o1 = _rows(no2, ctx.rank)
        st["eri_v_rows"] = (v0, v1)
        st["eri_o_rows"] = (o0, o1)
        st["eri_V"] = cp.asarray(V[v0:v1])
        st["eri_O"] = cp.asarray(O[o0:o1])
        st["eri_OV"] = cp.asarray(OV[o0:o1])
        cp.cuda.Device().synchronize()          # uploads from pinned memory are asynchronous
    group.each(_up)
    return V, O, OV


def _split_partial(ctx, zvvT_h, zooT_h):
    """This slot's contribution: rows P of vvvv/oooo and rows Q of oovv."""
    st = ctx.state
    v0, v1 = st["eri_v_rows"]
    o0, o1 = st["eri_o_rows"]
    zvvT = cp.asarray(zvvT_h)
    zooT = cp.asarray(zooT_h)
    # row strips P of the symmetric vvvv / oooo give full-width partial sums;
    # the oovv rows Q give prod_vv's partial sum and prod_oo's disjoint columns Q
    prod_vv = zvvT[:, v0:v1] @ st["eri_V"]
    prod_vv += zooT[:, o0:o1] @ st["eri_OV"]
    prod_oo = zooT[:, o0:o1] @ st["eri_O"]
    prod_oo[:, o0:o1] += zvvT @ st["eri_OV"].T
    out = (cp.asnumpy(prod_vv), cp.asnumpy(prod_oo))
    prod_vv = prod_oo = zvvT = zooT = None
    return out


def attach_gpu_eri_contraction(pprpa, vvvv, oovv, oooo, mode="auto",
                               tile=None, total_bytes=None, group=None):
    """Route Davidson MVP through cupy.  ``mode`` is auto/resident/split/tiled.

    Requires cupy.

    ``total_bytes`` overrides live VRAM for auto-select (tests).  ``tile``
    forces the tiled GEMM strip length; otherwise ``DAVIDSON_ERI_TILE`` or
    an analytic estimate from free VRAM is used.  ``group`` (a
    ``gpu_multi.DeviceGroup``; default ``LIB_PPRPA_GPUS`` slots) enables the
    split-resident mode when it has more than one slot.
    """
    _require_cupy()
    global LAST_TELEMETRY
    from lib_pprpa.gpu_multi import default_group
    group = group or default_group()
    pprpa._use_eri = True
    pprpa._eri_group = group
    nvir = pprpa.nvir
    nocc = pprpa.nocc
    chosen = _choose_mode(nocc, nvir, vvvv, oovv, oooo, mode, total_bytes)
    if chosen == "tiled" and mode in ("auto", None) and group.nslots > 1 \
            and not os.environ.get("PPRPA_ERI_MODE") and _split_fits(nocc, nvir, group):
        chosen = "split"
    if chosen == "split" and group.nslots < 2:
        chosen = "resident"
    nbytes = eri_bytes(nocc, nvir)
    pprpa._eri_stream = {"mvps": 0, "bytes": 0.0, "seconds": 0.0, "ntri": 0}

    def _finish(chosen_mode, tile_used, pinned=None):
        pprpa.vvvv = np.empty(0, dtype=np.float64)
        pprpa.oovv = None
        pprpa.oooo = None
        LAST_TELEMETRY.clear()
        LAST_TELEMETRY.update({
            "mode": chosen_mode,
            "tile": None if tile_used is None else int(tile_used),
            "eri_bytes": int(nbytes),
            "nocc": int(nocc),
            "nvir": int(nvir),
            "pinned_host": pinned,
            "gpu_slots": list(group.devices),
        })
        print(
            f"{_ts()} [pprpa_eri_gpu] mode={chosen_mode}"
            f"{'' if tile_used is None else f' tile={tile_used}'}"
            f"{'' if pinned is None else f' pinned_host={pinned}'}"
            f" eri={nbytes/1e9:.2f} GB no={nocc} nv={nvir} slots={group.nslots}",
            flush=True,
        )
        return pprpa

    if chosen == "split":
        t0 = time.perf_counter()
        V, O, OV = _split_upload(group, vvvv, oovv, oooo, nocc, nvir)
        pinned = bool(_is_pinned(V) and _is_pinned(O) and _is_pinned(OV))
        pprpa._host_vvvv = pprpa._host_oooo = pprpa._host_oovv = None
        pprpa._gpu_vvvv = pprpa._gpu_oooo = pprpa._gpu_oovv = None
        pprpa._gpu_mo_energy = cp.asarray(pprpa.mo_energy)
        pprpa._eri_tile = None
        pprpa.contraction = lambda tri_vec: _gpu_eri_contraction_split(pprpa, tri_vec)
        print(f"{_ts()} [pprpa_eri_gpu] split upload {nbytes/1e9:.1f} GB over {group.nslots} slots "
              f"in {time.perf_counter() - t0:.1f} s", flush=True)
        return _finish("split", None, pinned)

    if chosen == "resident":
        try:
            pprpa._gpu_vvvv = cp.ascontiguousarray(
                cp.asarray(vvvv).reshape(nvir * nvir, nvir * nvir))
            pprpa._gpu_oooo = cp.ascontiguousarray(
                cp.asarray(oooo).reshape(nocc * nocc, nocc * nocc))
            pprpa._gpu_oovv = cp.ascontiguousarray(
                cp.asarray(oovv).reshape(nocc * nocc, nvir * nvir))
            pprpa._gpu_mo_energy = cp.asarray(pprpa.mo_energy)
            pprpa._host_vvvv = pprpa._host_oooo = pprpa._host_oovv = None
            pprpa._eri_tile = None
            pprpa.contraction = lambda tri_vec: _gpu_eri_contraction(pprpa, tri_vec)
            return _finish("resident", None)
        except Exception as exc:
            if not _is_oom(exc):
                raise
            print(f"{_ts()} [pprpa_eri_gpu] resident ERI upload OOM — falling back to tiled",
                  flush=True)
            for attr in ("_gpu_vvvv", "_gpu_oooo", "_gpu_oovv"):
                setattr(pprpa, attr, None)
            chosen = "tiled"

    pprpa._host_vvvv = _flatten_host(vvvv, nvir, nvir)
    pprpa._host_oooo = _flatten_host(oooo, nocc, nocc)
    pprpa._host_oovv = _flatten_host(oovv, nocc, nvir)
    pinned = bool(all(_is_pinned(a) for a in (pprpa._host_vvvv, pprpa._host_oooo, pprpa._host_oovv)))
    pprpa._gpu_vvvv = pprpa._gpu_oooo = pprpa._gpu_oovv = None
    pprpa._gpu_mo_energy = cp.asarray(pprpa.mo_energy)
    n2 = max(nocc * nocc, nvir * nvir)
    pprpa._eri_tile = estimate_eri_tile(n2, tile=tile)
    pprpa.contraction = lambda tri_vec: _gpu_eri_contraction_tiled(pprpa, tri_vec)
    return _finish("tiled", pprpa._eri_tile, pinned)


def release_gpu_eri(pprpa, *extra):
    """Drop GPU and host MO-ERI tensors held by attach_gpu_eri_contraction."""
    _require_cupy()
    import gc
    st = getattr(pprpa, "_eri_stream", None)
    if st and st["mvps"]:
        rate = st["bytes"] / max(st["seconds"], 1e-9) / 1e9
        LAST_TELEMETRY["stream"] = dict(st, gb_per_s=rate)
        print(f"{_ts()} [pprpa_eri_gpu] {st['mvps']} MVPs, {st['ntri']} trial vectors: "
              f"streamed {st['bytes']/1e9:.1f} GB in {st['seconds']:.1f} s "
              f"({rate:.1f} GB/s effective)", flush=True)
    group = getattr(pprpa, "_eri_group", None)
    if group is not None and group.nslots > 1:
        try:
            group.free(["eri_V", "eri_O", "eri_OV", "eri_v_rows", "eri_o_rows"])
        except Exception:
            pass
    for attr in ("_gpu_vvvv", "_gpu_oooo", "_gpu_oovv", "_gpu_mo_energy",
                 "_host_vvvv", "_host_oooo", "_host_oovv"):
        if hasattr(pprpa, attr):
            setattr(pprpa, attr, None)
    for obj in extra:
        del obj
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    try:
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"{_ts()} [mem] after ERI release: free≈{free/1e9:.2f}/{total/1e9:.2f} GB", flush=True)


def _prepare_z(pprpa, tri_vec):
    nocc, nvir = pprpa.nocc, pprpa.nvir
    no2, nv2 = nocc * nocc, nvir * nvir
    oo_dim = pprpa.oo_dim
    k = (1 if pprpa.multi == "s" else 0) - 1
    tro, tco = cp.tril_indices(nocc, k)
    trv, tcv = cp.tril_indices(nvir, k)
    di_o = cp.arange(nocc)
    di_v = cp.arange(nvir)

    T = cp.asarray(tri_vec)
    ntri = T.shape[0]
    z_oo = cp.zeros((ntri, nocc, nocc))
    z_vv = cp.zeros((ntri, nvir, nvir))
    z_oo[:, tro, tco] = T[:, :oo_dim]
    z_oo[:, di_o, di_o] *= _INV_SQRT2
    z_vv[:, trv, tcv] = T[:, oo_dim:]
    z_vv[:, di_v, di_v] *= _INV_SQRT2
    zooT = z_oo.transpose(0, 2, 1).reshape(ntri, no2)
    zvvT = z_vv.transpose(0, 2, 1).reshape(ntri, nv2)
    return T, zooT, zvvT, tro, tco, trv, tcv, di_o, di_v


def _finish_mv(pprpa, T, prod_vv, prod_oo, tro, tco, trv, tcv, di_o, di_v):
    nocc, nvir = pprpa.nocc, pprpa.nvir
    ntri = T.shape[0]
    oo_dim = pprpa.oo_dim
    prod_vv = prod_vv.reshape(ntri, nvir, nvir)
    prod_oo = prod_oo.reshape(ntri, nocc, nocc)

    if pprpa.multi == "s":
        prod_vv = prod_vv + prod_vv.transpose(0, 2, 1)
        prod_oo = prod_oo + prod_oo.transpose(0, 2, 1)
    else:
        prod_vv = prod_vv - prod_vv.transpose(0, 2, 1)
        prod_oo = prod_oo - prod_oo.transpose(0, 2, 1)

    prod_oo = cp.ascontiguousarray(prod_oo.transpose(0, 2, 1))
    prod_oo[:, di_o, di_o] *= _INV_SQRT2
    prod_vv = cp.ascontiguousarray(prod_vv.transpose(0, 2, 1))
    prod_vv[:, di_v, di_v] *= _INV_SQRT2

    mv = cp.empty((ntri, pprpa.full_dim))
    mv[:, :oo_dim] = prod_oo[:, tro, tco]
    mv[:, oo_dim:] = prod_vv[:, trv, tcv]

    me = pprpa._gpu_mo_energy
    orb_oo = (me[None, :nocc] + me[:nocc, None])[tro, tco]
    orb_vv = (me[None, nocc:] + me[nocc:, None])[trv, tcv]
    orb = cp.concatenate((orb_oo, orb_vv)) - 2.0 * pprpa.mu
    orb[:oo_dim] *= -1.0
    mv += orb[None, :] * T
    return cp.asnumpy(mv)


def _gpu_eri_contraction(pprpa, tri_vec):
    T, zooT, zvvT, tro, tco, trv, tcv, di_o, di_v = _prepare_z(pprpa, tri_vec)
    prod_vv = zvvT @ pprpa._gpu_vvvv.T
    prod_oo = zooT @ pprpa._gpu_oooo.T
    prod_vv = prod_vv + zooT @ pprpa._gpu_oovv
    prod_oo = prod_oo + zvvT @ pprpa._gpu_oovv.T
    return _finish_mv(pprpa, T, prod_vv, prod_oo, tro, tco, trv, tcv, di_o, di_v)


def _gpu_eri_contraction_split(pprpa, tri_vec):
    T, zooT, zvvT, tro, tco, trv, tcv, di_o, di_v = _prepare_z(pprpa, tri_vec)
    t0 = time.perf_counter()
    zvvT_h = cp.asnumpy(zvvT)
    zooT_h = cp.asnumpy(zooT)
    parts = pprpa._eri_group.each(lambda ctx: _split_partial(ctx, zvvT_h, zooT_h))
    prod_vv = cp.asarray(sum(p[0] for p in parts))
    prod_oo = cp.asarray(sum(p[1] for p in parts))
    st = pprpa._eri_stream
    st["mvps"] += 1
    st["ntri"] += int(T.shape[0])
    st["bytes"] += 2.0 * (zvvT_h.nbytes + zooT_h.nbytes) * pprpa._eri_group.nslots
    st["seconds"] += time.perf_counter() - t0
    return _finish_mv(pprpa, T, prod_vv, prod_oo, tro, tco, trv, tcv, di_o, di_v)


def _gpu_eri_contraction_tiled(pprpa, tri_vec):
    T, zooT, zvvT, tro, tco, trv, tcv, di_o, di_v = _prepare_z(pprpa, tri_vec)
    n2 = max(pprpa.nocc * pprpa.nocc, pprpa.nvir * pprpa.nvir)
    tile = max(1, int(getattr(pprpa, "_eri_tile", 0) or estimate_eri_tile(n2)))
    t0 = time.perf_counter()
    while True:
        try:
            prod_vv, prod_oo = eri_mvp_tiled(
                zvvT, zooT,
                pprpa._host_vvvv, pprpa._host_oovv, pprpa._host_oooo,
                tile, xp=cp,
            )
            cp.cuda.Device().synchronize()
            st = pprpa._eri_stream
            st["mvps"] += 1
            st["ntri"] += int(T.shape[0])
            st["bytes"] += float(pprpa._host_vvvv.nbytes + pprpa._host_oooo.nbytes
                                 + pprpa._host_oovv.nbytes)
            st["seconds"] += time.perf_counter() - t0
            break
        except Exception as exc:
            if not _is_oom(exc) or tile <= 1:
                raise
            tile = max(1, tile // 2)
            pprpa._eri_tile = tile
            print(f"{_ts()} [pprpa_eri_gpu] OOM — retrying tiled MVP with tile={tile}",
                  flush=True)
            cp.get_default_memory_pool().free_all_blocks()
    return _finish_mv(pprpa, T, prod_vv, prod_oo, tro, tco, trv, tcv, di_o, di_v)


if __name__ == "__main__":
    from lib_pprpa.pprpa_davidson import ppRPA_Davidson, _pprpa_contraction

    rng = np.random.default_rng(0)
    nocc, nvir = 6, 8
    nmo = nocc + nvir
    moe = rng.standard_normal(nmo)
    vvvv = rng.standard_normal((nvir, nvir, nvir, nvir))
    oooo = rng.standard_normal((nocc, nocc, nocc, nocc))
    oovv = rng.standard_normal((nocc, nocc, nvir, nvir))

    for multi in ("s", "t"):
        cpu = ppRPA_Davidson(nocc, moe, Lpq=None, channel="hh",
                             nroot=2, residue_thresh=1e-10, trial="identity")
        cpu.mu = 0.0
        cpu.use_eri(vvvv, oovv, oooo)
        cpu.multi = multi
        cpu.check_parameter()
        tv = rng.standard_normal((7, cpu.full_dim))
        mv_cpu = _pprpa_contraction(cpu, tv)

        gpu = ppRPA_Davidson(nocc, moe, Lpq=None, channel="hh",
                             nroot=2, residue_thresh=1e-10, trial="identity")
        gpu.mu = 0.0
        gpu.multi = multi
        gpu.check_parameter()
        attach_gpu_eri_contraction(gpu, vvvv, oovv, oooo, mode="resident")
        mv_res = gpu.contraction(tv)
        attach_gpu_eri_contraction(gpu, vvvv, oovv, oooo, mode="tiled", tile=5)
        mv_tile = gpu.contraction(tv)
        print(
            f"multi={multi}  full_dim={cpu.full_dim}  "
            f"max|res-cpu|={np.abs(mv_res - mv_cpu).max():.3e}  "
            f"max|tile-res|={np.abs(mv_tile - mv_res).max():.3e}"
        )
