"""GPU (cupy) ERI contraction for the pp-RPA Davidson ``use_eri`` path.

Two modes, selected by ``attach_gpu_eri_contraction``:

* **resident** — keep ``vvvv`` / ``oooo`` / ``oovv`` on the GPU and GEMM all
  trial vectors at once (original path).
* **tiled** — keep the tensors on the host and stream ``(blk, n²)`` slices for
  the same GEMMs.  Auto-selected when the three tensors would not fit in 75% of
  VRAM (``gpu_mem.fits_resident``, same rule as ao2mo).

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

import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

from lib_pprpa.gpu_mem import eri_bytes, fits_resident

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


def _flatten_host(block, n0, n1):
    """``(n0, n0, n1, n1)`` or ``(n0*n0, n1*n1)`` → C-contiguous ``(n0², n1²)`` numpy."""
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
    """
    tile = max(1, int(tile))
    ntri, nv2 = zvvT.shape
    no2 = zooT.shape[1]
    prod_vv = xp.zeros((ntri, nv2), dtype=zvvT.dtype)
    prod_oo = xp.zeros((ntri, no2), dtype=zooT.dtype)

    for p0 in range(0, nv2, tile):
        p1 = min(p0 + tile, nv2)
        Vt = xp.asarray(vvvv[:, p0:p1])
        prod_vv = prod_vv + zvvT[:, p0:p1] @ Vt.T
        Vt = None

    for p0 in range(0, no2, tile):
        p1 = min(p0 + tile, no2)
        Ot = xp.asarray(oooo[:, p0:p1])
        prod_oo = prod_oo + zooT[:, p0:p1] @ Ot.T
        Ot = None

    for p0 in range(0, no2, tile):
        p1 = min(p0 + tile, no2)
        ov = xp.asarray(oovv[p0:p1, :])
        prod_vv = prod_vv + zooT[:, p0:p1] @ ov
        ov = None

    for q0 in range(0, nv2, tile):
        q1 = min(q0 + tile, nv2)
        ov = xp.asarray(oovv[:, q0:q1])
        prod_oo = prod_oo + zvvT[:, q0:q1] @ ov.T
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
    if mode not in ("auto", "resident", "tiled"):
        raise ValueError(f"unknown ERI mode {mode!r}")
    if mode != "auto":
        return mode
    if all(_is_cupy(x) for x in (vvvv, oovv, oooo)):
        return "resident"
    return "resident" if fits_resident(
        nocc, nvir, extra_bytes=0, total_bytes=total_bytes) else "tiled"


def attach_gpu_eri_contraction(pprpa, vvvv, oovv, oooo, mode="auto",
                               tile=None, total_bytes=None):
    """Route Davidson MVP through cupy.  ``mode`` is auto/resident/tiled.

    Requires cupy.

    ``total_bytes`` overrides live VRAM for auto-select (tests).  ``tile``
    forces the tiled GEMM strip length; otherwise ``DAVIDSON_ERI_TILE`` or
    an analytic estimate from free VRAM is used.
    """
    _require_cupy()
    global LAST_TELEMETRY
    pprpa._use_eri = True
    nvir = pprpa.nvir
    nocc = pprpa.nocc
    chosen = _choose_mode(nocc, nvir, vvvv, oovv, oooo, mode, total_bytes)
    nbytes = eri_bytes(nocc, nvir)

    def _finish(chosen_mode, tile_used):
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
        })
        print(
            f"[pprpa_eri_gpu] mode={chosen_mode}"
            f"{'' if tile_used is None else f' tile={tile_used}'}"
            f" eri={nbytes/1e9:.2f} GB no={nocc} nv={nvir}",
            flush=True,
        )
        return pprpa

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
            print("[pprpa_eri_gpu] resident ERI upload OOM — falling back to tiled",
                  flush=True)
            for attr in ("_gpu_vvvv", "_gpu_oooo", "_gpu_oovv"):
                setattr(pprpa, attr, None)
            chosen = "tiled"

    pprpa._host_vvvv = _flatten_host(vvvv, nvir, nvir)
    pprpa._host_oooo = _flatten_host(oooo, nocc, nocc)
    pprpa._host_oovv = _flatten_host(oovv, nocc, nvir)
    pprpa._gpu_vvvv = pprpa._gpu_oooo = pprpa._gpu_oovv = None
    pprpa._gpu_mo_energy = cp.asarray(pprpa.mo_energy)
    n2 = max(nocc * nocc, nvir * nvir)
    pprpa._eri_tile = estimate_eri_tile(n2, tile=tile)
    pprpa.contraction = lambda tri_vec: _gpu_eri_contraction_tiled(pprpa, tri_vec)
    return _finish("tiled", pprpa._eri_tile)


def release_gpu_eri(pprpa, *extra):
    """Drop GPU and host MO-ERI tensors held by attach_gpu_eri_contraction."""
    _require_cupy()
    import gc
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
    print(f"[mem] after ERI release: free≈{free/1e9:.2f}/{total/1e9:.2f} GB", flush=True)


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


def _gpu_eri_contraction_tiled(pprpa, tri_vec):
    T, zooT, zvvT, tro, tco, trv, tcv, di_o, di_v = _prepare_z(pprpa, tri_vec)
    n2 = max(pprpa.nocc * pprpa.nocc, pprpa.nvir * pprpa.nvir)
    tile = max(1, int(getattr(pprpa, "_eri_tile", 0) or estimate_eri_tile(n2)))
    while True:
        try:
            prod_vv, prod_oo = eri_mvp_tiled(
                zvvT, zooT,
                pprpa._host_vvvv, pprpa._host_oovv, pprpa._host_oooo,
                tile, xp=cp,
            )
            break
        except Exception as exc:
            if not _is_oom(exc) or tile <= 1:
                raise
            tile = max(1, tile // 2)
            pprpa._eri_tile = tile
            print(f"[pprpa_eri_gpu] OOM — retrying tiled MVP with tile={tile}",
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
