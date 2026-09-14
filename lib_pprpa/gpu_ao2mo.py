"""GPU (cupy) Gamma-point FFT ao2mo for pp-RPA: builds the active-space
vvvv / oovv / oooo blocks directly on the GPU (no CPU ao2mo, no GDF).

Same FFT MO-integral transform as pyscf.pbc.df.fft_ao2mo, in cupy:
  rho_pq(r) = mo_p(r) mo_q(r)  ->  fft  ->  eri_chem[(pq),(rs)] = sum_G rho_pq(G)* wG rho_rs(G)
then permuted to the physicist convention pp-RPA expects (matches pprpaobj):
  vvvv[a,b,c,d] = <ab|cd>,  oovv[i,j,a,b] = <ij|ab>,  oooo[i,j,k,l] = <ij|kl>.

Memory strategy
---------------
* Pair strips are streamed so the full (npair, ngrid) codensity is never built.
* pair_blk is chosen first with chem on host (full VRAM for the FFT strip), via
  cold cuFFT probes + binary search.
* Chem ERI is moved to the GPU only if that same pair_blk still fits beside it —
  never shrink the strip just to keep chem on device.
* Chemist->physicist reorder host-bounces to avoid a 2x GPU peak.
* Outer pair strips are dispatched over ``gpu_multi.DeviceGroup`` slots (one
  by default; LIB_PPRPA_GPUS=2 opts in).  MO grids are replicated per slot; an
  OOM shrinks only that slot's sub-strip and redoes the failing strip instead
  of restarting the whole tensor.  With >1 slot the finals are host-staged.
* Strips are also capped at ``gpu_mem.max_fft_batch(ngrid)`` rows: cuFFT
  returns CUFFT_INVALID_SIZE (not an OOM) for batched plans above 2^31
  elements on Bluestein-sized meshes (e.g. 151^3), which a roomy B200 would
  otherwise trigger with pair_blk=1200.  That error is treated as retryable.
* After assembly, the three finals are uploaded only if they still fit in
  75% of VRAM (``gpu_mem.fits_resident``). Otherwise they stay on the host
  for tiled Davidson contraction.

Validated element-wise vs CPU pprpaobj(mo_eri=True) (run this file as __main__).
"""
from __future__ import annotations

import math
import os
import threading
import time

import numpy as np
import cupy as cp
from gpu4pyscf.pbc import tools as gtools
from gpu4pyscf.pbc.dft import numint as gnumint

from lib_pprpa.gpu_mem import eri_bytes, fits_resident, max_fft_batch
from lib_pprpa.pprpa_util import tstamp as _ts


LAST_TELEMETRY = {}


def get_last_telemetry():
    """Return metrics from the most recent gpu_ao2mo_blocks call."""
    return dict(LAST_TELEMETRY)


# current-device helpers live in gpu_multi (shared with gpu_fft_k / gpu_pairing_force)
from lib_pprpa.gpu_multi import (  # noqa: E402,F401  (re-exported for callers/tests)
    _sync, _free_pool, _clear_fft_cache, _reclaim_gpu, _free_bytes, _is_oom)


def _mo_on_grid(cell, mo, mesh):
    """MO values on the uniform grid: (nmo, ngrid) cupy real (Gamma)."""
    coords = cell.gen_uniform_grids(mesh)
    ao = gnumint.eval_ao_kpts(cell, coords, kpts=np.zeros((1, 3)), deriv=0)[0]
    ao = cp.asarray(ao)  # (ngrid, nao), real at Gamma
    return cp.asarray(mo).T @ ao.T  # (nmo, ngrid)


def _codensity_pairs(moA, moB, p0, p1):
    """Codensity strip for pair indices [p0:p1], pair = a*nB + b -> (blk, ngrid)."""
    nB = moB.shape[0]
    idx = cp.arange(p0, p1)
    return moA[idx // nB] * moB[idx % nB]


def _cushion_bytes():
    """VRAM to hold back during probes so assembly is not razor-tight.

    At least 512 MiB, or 2% of currently free memory (whichever is larger).
    """
    return max(512 * 1024 ** 2, int(0.02 * _free_bytes()))



def _estimate_pair_blk(npair, ngrid, nB, pair_blk=None, free=None, mesh=None):
    """Estimate a fast strip size from live driver memory.

    The final ERI tensor has already been allocated when this is called.  The
    estimate budgets the remaining memory for rho, FFT output/workspace, vR,
    rho_q and the GEMM tile.  Default strips are aligned to complete second-MO
    rows, which permits direct contiguous writes in physicist layout.  ``free``
    (bytes) overrides the current-device query (multi-GPU: plan from the
    slot with the least free memory).
    """
    if pair_blk is not None:
        raw = max(1, min(int(pair_blk), npair))
    else:
        if free is None:
            free = _free_bytes()
        reserve = max(_cushion_bytes(), int(0.06 * free))
        budget = max(0, free - reserve)
        # rho_p (8) + complex FFT copy (16) + vG (16) + vR (16) + rho_q (8) bytes
        # per pair-gridpoint; 40 was optimistic (216-atom vvvv OOMed at 900 and
        # fell back to 300 while 600 ran fine).
        linear = max(ngrid * 64, 1)
        # Solve 8*b^2 + linear*b <= budget (tile plus grid temporaries).
        raw = int((-linear + math.sqrt(linear * linear + 32 * budget)) / 16)
        raw = max(1, min(raw, npair))
        env_cap = os.environ.get("GPU_AO2MO_MAX_PAIR_BLK")
        if env_cap:
            raw = min(raw, max(1, int(env_cap)))
    # One strip is one batched cuFFT plan: keep it under the cuFFT element limit
    # (strict 2^31 on Bluestein meshes, relaxed on direct-path meshes).
    raw = min(raw, max_fft_batch(ngrid, mesh=mesh))
    raw = max(nB, raw)
    raw = max(nB, (raw // nB) * nB)
    return min(raw, npair)

def _strip_task(ctx, task, keyA, keyB, mesh, out, on_gpu, nB, npair):
    """Outer pair strip [p0, p1) of a physicist-layout ERI, in sub-strips of the
    slot's ``sub_blk`` (whole nB-row multiples).  Every (outer, inner) tile is a
    plain assignment into a disjoint ``out`` rectangle, so a retry is idempotent
    and slots never write the same elements."""
    st = ctx.state
    moA, moB, wcoulG = st[keyA], st[keyB], st["wcoulG"]
    p0, p1 = task
    s0 = p0
    while s0 < p1:
        blk = int(st["sub_blk"])
        s1 = min(p1, s0 + blk)
        a0, a1 = s0 // nB, s1 // nB
        rho_p = _codensity_pairs(moA, moB, s0, s1)
        vR = gtools.ifft(gtools.fft(rho_p, mesh) * wcoulG, mesh).real
        rho_p = None
        for q0 in range(0, npair, blk):
            q1 = min(q0 + blk, npair)
            b0, b1 = q0 // nB, q1 // nB
            rho_q = _codensity_pairs(moA, moB, q0, q1)
            tile = vR.dot(rho_q.T)
            phys = tile.reshape(a1 - a0, nB, b1 - b0, nB).transpose(0, 2, 1, 3)
            if on_gpu:
                out[a0:a1, b0:b1] = phys
            else:
                out[a0:a1, b0:b1] = cp.asnumpy(phys)
            rho_q = tile = phys = None
        vR = None
        s0 = s1


def _make_strip_shrink(nB):
    def _shrink(ctx, task, exc):
        blk = int(ctx.state["sub_blk"])
        if blk <= nB:
            return False
        ctx.state["sub_blk"] = max(nB, ((blk // 2) // nB) * nB)
        return True
    return _shrink


def _block_direct(name, keyA, keyB, mesh, group, pair_blk=None, force_host=False):
    """Fill a final physicist ERI on GPU or directly on host when VRAM is tight.

    ``keyA``/``keyB`` name the MO grids in every slot's ``state`` (see
    ``DeviceGroup.broadcast``).  ``force_host`` stages a final tensor until all
    MO grids are released; with more than one GPU slot the tensor is always
    host-staged because only device 0 could write a cupy ``out``.  Outer pair
    strips are the dispatched tasks; an OOM shrinks only the failing slot's
    sub-strip and redoes that task (no whole-tensor restart).
    """
    moA0, moB0 = group.ctxs[0].state[keyA], group.ctxs[0].state[keyB]
    nA, nB, ngrid = moA0.shape[0], moB0.shape[0], moA0.shape[1]
    npair = nA * nB
    out_bytes = npair * npair * 8
    free_before = group.min_free_bytes()
    on_gpu = not force_host and group.inline
    if on_gpu:
        try:
            out = cp.empty((nA, nA, nB, nB), dtype=cp.float64)
        except Exception as exc:
            if not _is_oom(exc):
                raise
            raise MemoryError(
                f"gpu_ao2mo: final {name} tensor ({out_bytes/1e9:.2f} GB) "
                f"does not fit (free≈{free_before/1e9:.2f} GB)") from exc
        _reclaim_gpu()
        blk = _estimate_pair_blk(npair, ngrid, nB, pair_blk=pair_blk, mesh=mesh)
        # A retained result must leave enough space for useful FFT strips.
        if pair_blk is None and blk <= min(npair, 4 * nB):
            del out
            _reclaim_gpu()
            on_gpu = False
    if not on_gpu:
        out = np.empty((nA, nA, nB, nB), dtype=np.float64)
        blk = _estimate_pair_blk(npair, ngrid, nB, pair_blk=pair_blk,
                                 free=group.min_free_bytes(), mesh=mesh)
    stats = {
        "name": name,
        "npair": int(npair),
        "output_bytes": int(out_bytes),
        "requested_pair_blk": None if pair_blk is None else int(pair_blk),
        "pair_blk": int(blk),
        "retries": 0,
        "output_location": ("gpu" if on_gpu else
                            ("host_staged" if force_host else "host_direct")),
        "free_before_bytes": int(free_before),
        "free_after_output_bytes": int(group.min_free_bytes()),
        "min_free_bytes": int(group.min_free_bytes()),
    }
    location = "GPU" if on_gpu else ("host-staged" if force_host else "host-direct")
    tasks = [(p0, min(p0 + blk, npair)) for p0 in range(0, npair, blk)]
    nblk = len(tasks)
    print(
        f"{_ts()} [gpu_ao2mo] {name} final tensor {out_bytes/1e9:.2f} GB "
        f"on {location}; pair_blk={blk} (analytic); "
        f"free≈{free_before/1e9:.2f}→{stats['free_after_output_bytes']/1e9:.2f} GB", flush=True)
    print(
        f"{_ts()} [gpu_ao2mo]   direct strips={nblk}x{nblk} GEMMs={nblk * nblk}; "
        f"pair_blk={blk} on {group.nslots} slot(s)", flush=True)
    stats["nstrips"] = nblk
    stats["gemms"] = nblk * nblk

    def _init(ctx):
        ctx.state["sub_blk"] = blk
    group.each(_init)
    done = [0]
    lock = threading.Lock()
    report_every = max(1, nblk // 4)

    def _work(ctx, task):
        _strip_task(ctx, task, keyA, keyB, mesh, out, on_gpu, nB, npair)
        with lock:
            done[0] += 1
            k = done[0]
        if nblk > 1 and (k % report_every == 0 or k == nblk):
            print(f"{_ts()} [gpu_ao2mo]   strip {k}/{nblk} done", flush=True)

    started = time.perf_counter()
    try:
        _results, run_stats = group.run(tasks, _work, shrink=_make_strip_shrink(nB),
                                        label=f"ao2mo {name}")
    except Exception:
        del out
        raise
    if not on_gpu and not force_host:
        uploaded = cp.asarray(out)
        del out
        out = uploaded
    _sync()
    stats["seconds"] = time.perf_counter() - started
    stats["retries"] = int(sum(run_stats["retries_per_slot"]))
    stats["final_pair_blk_per_slot"] = [int(c.state["sub_blk"]) for c in group.ctxs]
    stats["min_free_bytes"] = int(min(
        [b for b in run_stats["min_free_bytes_per_slot"] if b is not None] + [stats["min_free_bytes"]]))
    stats["multi_gpu"] = run_stats
    return out, stats


def gpu_ao2mo_blocks(cell, cocc, cvir, mesh, pair_blk=None, return_gpu=False, group=None):
    """Return direct-assembled vvvv, oovv, oooo tensors in physicist layout.

    ``group`` (``lib_pprpa.gpu_multi.DeviceGroup``, default LIB_PPRPA_GPUS slots)
    dispatches the pair strips over GPUs; the MO grids are built once on the
    current device and replicated to the other slots.
    """
    global LAST_TELEMETRY
    from lib_pprpa.gpu_multi import default_group
    group = group or default_group()
    _free_pool()
    _sync()
    total_started = time.perf_counter()
    free_started, total_bytes = cp.cuda.runtime.memGetInfo()

    grid_started = time.perf_counter()
    moO = _mo_on_grid(cell, cocc, mesh)
    moV = _mo_on_grid(cell, cvir, mesh)
    no, nv, ng = moO.shape[0], moV.shape[0], moO.shape[1]
    coulG = cp.asarray(gtools.get_coulG(cell, mesh=mesh))
    wcoulG = coulG * (cell.vol / ng)
    _sync()
    group.broadcast(moO, "moO")
    group.broadcast(moV, "moV")
    group.broadcast(wcoulG, "wcoulG")
    grid_seconds = time.perf_counter() - grid_started

    final_bytes = eri_bytes(no, nv)
    vvvv_b = nv ** 4 * 8
    oooo_b = no ** 4 * 8
    oovv_b = (no ** 2) * (nv ** 2) * 8
    print(
        f"{_ts()} [gpu_ao2mo] no={no} nv={nv} ngrid={ng} | "
        f"ERI sizes vvvv/oovv/oooo = "
        f"{vvvv_b/1e9:.2f}/{oovv_b/1e9:.2f}/{oooo_b/1e9:.2f} GB | "
        f"free≈{_free_bytes()/1e9:.2f} GB", flush=True)

    grid_bytes = moO.nbytes + moV.nbytes + coulG.nbytes + wcoulG.nbytes
    stage_host = not fits_resident(
        no, nv, extra_bytes=grid_bytes, total_bytes=total_bytes)
    if stage_host:
        print(f"{_ts()} [gpu_ao2mo] staging all final ERIs on host until MO grids are released",
              flush=True)
    vvvv, stat_v = _block_direct(
        "vvvv", "moV", "moV", mesh, group, pair_blk=pair_blk, force_host=stage_host)
    oooo, stat_o = _block_direct(
        "oooo", "moO", "moO", mesh, group, pair_blk=pair_blk, force_host=stage_host)
    oovv, stat_ov = _block_direct(
        "oovv", "moO", "moV", mesh, group, pair_blk=pair_blk, force_host=stage_host)

    group.free(["moO", "moV", "wcoulG"])
    del moO, moV, coulG, wcoulG
    _reclaim_gpu()
    uploaded = False
    if stage_host:
        can_resident = fits_resident(no, nv, extra_bytes=0, total_bytes=total_bytes)
        if can_resident:
            print(f"{_ts()} [gpu_ao2mo] uploading host ERIs to GPU for resident Davidson",
                  flush=True)
            vvvv_g = cp.asarray(vvvv); del vvvv; vvvv = vvvv_g
            oovv_g = cp.asarray(oovv); del oovv; oovv = oovv_g
            oooo_g = cp.asarray(oooo); del oooo; oooo = oooo_g
            uploaded = True
        else:
            print(
                f"{_ts()} [gpu_ao2mo] leaving ERIs on host for tiled Davidson "
                f"(eri={final_bytes/1e9:.2f} GB, "
                f"0.75*VRAM={0.75 * total_bytes/1e9:.2f} GB)",
                flush=True,
            )
    _free_pool()
    _sync()
    LAST_TELEMETRY.clear()
    LAST_TELEMETRY.update({
        "implementation": "direct_phys_v2_staged",
        "no": int(no),
        "nv": int(nv),
        "ngrid": int(ng),
        "mo_grid_seconds": float(grid_seconds),
        "blocks": [stat_v, stat_o, stat_ov],
        "min_free_bytes": int(min(s["min_free_bytes"] for s in (stat_v, stat_o, stat_ov))),
        "free_started_bytes": int(free_started),
        "total_vram_bytes": int(total_bytes),
        "host_staged": bool(stage_host),
        "uploaded_to_gpu": bool(uploaded),
        "gpu_slots": list(group.devices),
        "seconds": float(time.perf_counter() - total_started),
    })

    on_gpu = uploaded or not stage_host
    if return_gpu:
        return vvvv, oovv, oooo
    if on_gpu:
        return cp.asnumpy(vvvv), cp.asnumpy(oovv), cp.asnumpy(oooo)
    return vvvv, oovv, oooo


if __name__ == "__main__":
    from pyscf.pbc import gto, dft as cdft

    a0 = 3.370137329
    cell = gto.M(
        atom=[["C", [0.0, 0.0, 0.0]], ["C", [a0 / 2, a0 / 2, a0 / 2]]],
        a=np.array([[0, a0, a0], [a0, 0, a0], [a0, a0, 0]]),
        unit="bohr",
        basis="gth-szv",
        pseudo="gth-pade",
        verbose=0,
    )
    cell.mesh = [20, 20, 20]
    cell.build()
    mf = cdft.RKS(cell, xc="pbe")
    mf.exxdiv = None
    mf.conv_tol = 1e-10
    mf.kernel()
    nocc = cell.nelectron // 2
    nvir = cell.nao - nocc
    nmo = cell.nao
    cocc = mf.mo_coeff[:, :nocc]
    cvir = mf.mo_coeff[:, nocc:]
    eri = mf.with_df.get_mo_eri(mf.mo_coeff, compact=False)
    eri = eri.reshape(nmo, nmo, nmo, nmo).transpose(0, 2, 1, 3)
    vvvv_c = eri[nocc:, nocc:, nocc:, nocc:]
    oovv_c = eri[:nocc, :nocc, nocc:, nocc:]
    oooo_c = eri[:nocc, :nocc, :nocc, :nocc]
    for pb in (None, 8):
        vvvv_g, oovv_g, oooo_g = gpu_ao2mo_blocks(
            cell, cocc, cvir, cell.mesh, pair_blk=pb
        )
        print(f"pair_blk={pb}:")
        for name, g, c in [
            ("vvvv", vvvv_g, vvvv_c),
            ("oovv", oovv_g, oovv_c),
            ("oooo", oooo_g, oooo_c),
        ]:
            print(
                f"  {name}: shape {g.shape}  "
                f"max|gpu-cpu| = {np.abs(g - np.asarray(c)).max():.3e}"
            )
