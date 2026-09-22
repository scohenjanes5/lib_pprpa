"""GPU (cupy) Gamma-point FFT ao2mo for pp-RPA: builds the active-space
vvvv / oovv / oooo blocks directly on the GPU (no CPU ao2mo, no GDF).

Same FFT MO-integral transform as pyscf.pbc.df.fft_ao2mo, in cupy:
  rho_pq(r) = mo_p(r) mo_q(r)  ->  fft  ->  eri_chem[(pq),(rs)] = sum_G rho_pq(G)* wG rho_rs(G)
then permuted to the physicist convention pp-RPA expects (matches pprpaobj):
  vvvv[a,b,c,d] = <ab|cd>,  oovv[i,j,a,b] = <ij|ab>,  oooo[i,j,k,l] = <ij|kl>.

Memory strategy
---------------
* The MO grids are built in grid chunks (``_mos_on_grid``), so the (ngrid, nao)
  AO array is never materialised (90 GB at nao=2795, 159^3 -- the only term in
  ao2mo that is quadratic in system size; measured peak 109 -> 54 GB).  One AO
  block feeds every coefficient set, so the occ and vir grids share a single
  evaluation.  Each slot builds its own copy concurrently, which replaces the
  old build-on-device-0-then-broadcast; splitting the chunks across slots
  instead is opt-in (``GPU_AO2MO_GRID_DISPATCH=1``) because the host round trip
  costs more than the evaluation it saves at NV216 scale.
* The chemist ERI is a Gram matrix in the Coulomb metric, E = rho W rho^T with
  W = (vol/ngrid) F^-1 diag(coulG) F, so it block-factorises exactly:
  E[I,J] = (rho_I W) rho_J^T.  Pair strips are streamed accordingly and the
  full (npair, ngrid) codensity is never built (2.9 TB at nv=300, 159^3).
* pair_blk is chosen analytically from live free memory (``_estimate_pair_blk``:
  64 B per pair-gridpoint of transients plus the b x b tile), not by probing.
* The Gram matrix is symmetric, so only its lower-triangle tiles are formed
  (``pair_layout``); each tile is scattered together with its transpose.  For
  vvvv / oooo the same MO set sits on both sides of the pair, so the pair
  index runs over p >= q only: 1/8 of the all-pairs GEMM flops for those
  blocks, 1/2 for oovv.  Strips are whole rows of the pair layout, so the
  chemist->physicist write stays a set of slice assignments (rectangles for
  oovv, (p+1) x (r+1) sub-blocks for the compact blocks) with no reorder pass.
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
from lib_pprpa.pair_layout import PairLayout, write_tile
from lib_pprpa.pprpa_util import tstamp as _ts


LAST_TELEMETRY = {}


def get_last_telemetry():
    """Return metrics from the most recent gpu_ao2mo_blocks call."""
    return dict(LAST_TELEMETRY)


# current-device helpers live in gpu_multi (shared with gpu_fft_k / gpu_pairing_force)
from lib_pprpa.gpu_multi import (  # noqa: E402,F401  (re-exported for callers/tests)
    _sync, _free_pool, _clear_fft_cache, _reclaim_gpu, _free_bytes, _is_oom)


_KPTS0 = np.zeros((1, 3))  # one object: eval_ao_kpts asserts `kpts is opt.kpts`

# Floor on the grid chunk: below this the per-call overhead of eval_ao_kpts
# dominates and nothing is gained by shrinking further.
_AO_CHUNK_FLOOR = 4096


def _estimate_gchunk(nao, nmo_total, ngrid, gchunk=None, free=None, frac=0.4):
    """Grid points per AO chunk, so one chunk uses about ``frac`` of free VRAM.

    ``nao`` enters ao2mo in exactly one place — the AO values on the grid — and
    it enters quadratically in the system size (8*nao*ngrid bytes, 90 GB at
    nao=2795 on 159^3, while every other term is 8*nmo*ngrid or 64*pair_blk*
    ngrid).  Chunking the grid axis caps it at ``8*nao*gchunk``.  ``free``
    (bytes) overrides the current-device query (multi-GPU: plan from the slot
    with the least free memory).
    """
    if gchunk is not None:
        return max(1, min(int(gchunk), ngrid))
    env = os.environ.get("GPU_AO2MO_GRID_CHUNK")
    if env:
        return max(1, min(int(env), ngrid))
    if free is None:
        free = _free_bytes()
    # (g, nao) AO block, a like-sized allowance for what eval_ao_kpts holds
    # while building it, and one output column per MO of every set.
    per_point = max(1, 16 * int(nao) + 8 * int(nmo_total))
    budget = int(max(0, free) * frac)
    return int(max(min(_AO_CHUNK_FLOOR, ngrid),
                   min(ngrid, budget // per_point)))


def _grid_shrink(ctx, task, exc):
    """Halve this slot's AO chunk after an OOM and redo the same grid task."""
    cur = int(ctx.state["gchunk"])
    if cur <= 1:
        return False
    ctx.state["gchunk"] = max(1, cur // 2)
    return True


def _mos_on_grid(cell, mos, keys, mesh, group=None, gchunk=None, dispatch=None):
    """Build MO grids on every slot, leaving them in ``ctx.state[key]``.

    One (nmo, ngrid) array per entry of ``mos``, real at Gamma.  The
    (ngrid, nao) AO array is never materialised: the AOs are evaluated in grid
    chunks, and one AO block feeds every coefficient set, so the occupied and
    virtual grids share a single evaluation.  Chunking is exact -- an output
    element sums over ``nao`` only, so no reduction crosses a chunk boundary.

    Multi-slot strategy (measured at NV216 scale, nao=2795, 159^3, two B200s):

    * **replicated** (default): every slot chunks the whole grid into its own
      device arrays, concurrently.  Same wall time as one slot (1.0 s), no host
      staging and no peer traffic, and every slot ends holding what the strips
      need -- strictly less work than building on device 0 and broadcasting.
    * **dispatched** (``GPU_AO2MO_GRID_DISPATCH=1``): chunks are split across
      slots into a host array that is then broadcast back.  It halves the AO
      evaluation but pays a host round trip, which loses at this scale (2.5 s
      vs 1.0 s).  It is kept for systems where the evaluation dominates -- the
      AO term is the one part of ao2mo that grows as natoms^2.
    """
    from lib_pprpa.gpu_multi import default_group
    group = group or default_group()
    assert len(keys) == len(mos), "one state key per coefficient set"
    coords = cell.gen_uniform_grids(mesh)
    ngrid = len(coords)
    nao = int(cell.nao)
    mos = [np.asarray(m) for m in mos]
    nmos = [int(m.shape[1]) for m in mos]
    if dispatch is None:
        dispatch = os.environ.get("GPU_AO2MO_GRID_DISPATCH", "").strip() in ("1", "true", "yes")
    dispatch = bool(dispatch) and not group.inline

    started = time.perf_counter()

    def _chunk_to(ctx, out_list, s0, s1, to_host):
        """One AO block -> a column slice of every output. Returns bytes touched."""
        ao = gnumint.eval_ao_kpts(cell, coords[s0:s1], kpts=_KPTS0,
                                  deriv=0, opt=ctx.state["ao_opt"])[0]
        aoT = ao.T  # (nao, g), real at Gamma
        for out, C in zip(out_list, ctx.state["_mo_coeff"]):
            sub = C.T @ aoT  # (nmo, g)
            out[:, s0:s1] = cp.asnumpy(sub) if to_host else sub
            sub = None

    def _init(ctx, blk):
        ctx.state["gchunk"] = int(blk)
        ctx.state["ao_opt"] = gnumint._GTOvalOpt(cell, _KPTS0, deriv=0)
        ctx.state["_mo_coeff"] = [cp.asarray(m) for m in mos]

    if dispatch:
        outs = [np.empty((n, ngrid), dtype=np.float64) for n in nmos]
        blk = _estimate_gchunk(nao, sum(nmos), ngrid, gchunk=gchunk,
                               free=group.min_free_bytes())
        tasks = [(g0, min(g0 + blk, ngrid)) for g0 in range(0, ngrid, blk)]
        group.each(lambda ctx: _init(ctx, blk))

        def _work(ctx, task):
            g0, g1 = task
            s0 = g0
            while s0 < g1:
                s1 = min(g1, s0 + int(ctx.state["gchunk"]))
                _chunk_to(ctx, outs, s0, s1, True)
                s0 = s1

        _results, run_stats = group.run(tasks, _work, shrink=_grid_shrink,
                                        label="mo_on_grid")
        final_gchunk = [int(c.state.get("gchunk", blk)) for c in group.ctxs]
        nchunks = [len(tasks)]
        for key, arr in zip(keys, outs):
            group.broadcast(arr, key)
        outs = None
    else:
        blk = _estimate_gchunk(nao, sum(nmos), ngrid, gchunk=gchunk,
                               free=group.min_free_bytes())

        def _alloc(ctx):
            _init(ctx, blk)
            for key, n in zip(keys, nmos):
                ctx.state[key] = cp.empty((n, ngrid), dtype=np.float64)
            _reclaim_gpu()
            # re-plan per slot now that the outputs are resident (slots may be
            # different devices, or differently loaded)
            ctx.state["gchunk"] = _estimate_gchunk(nao, sum(nmos), ngrid,
                                                   gchunk=gchunk)

        def _build(ctx):
            out_list = [ctx.state[k] for k in keys]
            s0, done = 0, 0
            while s0 < ngrid:
                step = int(ctx.state["gchunk"])
                s1 = min(ngrid, s0 + step)
                try:
                    _chunk_to(ctx, out_list, s0, s1, False)
                except Exception as exc:  # noqa: BLE001 - classified below
                    if not _is_oom(exc) or step <= 1:
                        raise
                    _reclaim_gpu()
                    ctx.state["gchunk"] = max(1, step // 2)
                    ctx.log(f"mo_on_grid: {type(exc).__name__}; retrying with "
                            f"gchunk={ctx.state['gchunk']}")
                    continue
                s0 = s1
                done += 1
            return done

        group.each(_alloc)
        nchunks = group.each(_build)
        final_gchunk = [int(c.state.get("gchunk", blk)) for c in group.ctxs]
        run_stats = {"nslots": group.nslots, "devices": list(group.devices),
                     "mode": "replicated"}

    group.free(["ao_opt", "_mo_coeff", "gchunk"])
    _sync()
    stats = {
        "mode": "dispatched" if dispatch else "replicated",
        "nao": nao,
        "nmo_per_set": nmos,
        "ngrid": int(ngrid),
        "gchunk": int(blk),
        "nchunks": nchunks if isinstance(nchunks, list) else [nchunks],
        "final_gchunk_per_slot": final_gchunk,
        "ao_block_bytes": int(blk) * nao * 8,
        "ao_untiled_bytes": int(ngrid) * nao * 8,
        "seconds": time.perf_counter() - started,
        "multi_gpu": run_stats,
    }
    print(
        f"{_ts()} [gpu_ao2mo] MO grids nao={nao} "
        f"nmo={'+'.join(map(str, nmos))} ngrid={ngrid} | {stats['mode']} on "
        f"{group.nslots} slot(s); gchunk={blk} -> AO block "
        f"{stats['ao_block_bytes']/1e9:.2f} GB vs untiled "
        f"{stats['ao_untiled_bytes']/1e9:.2f} GB; {stats['seconds']:.1f} s",
        flush=True)
    return stats


def _mo_on_grid(cell, mo, mesh, group=None):
    """MO values on the uniform grid: (nmo, ngrid), real at Gamma.

    Single-set wrapper over ``_mos_on_grid``; returns the rank-0 array and drops
    the other slots' copies.
    """
    from lib_pprpa.gpu_multi import default_group
    group = group or default_group()
    _mos_on_grid(cell, [mo], ["_mo_grid_tmp"], mesh, group=group)
    out = group.ctxs[0].state["_mo_grid_tmp"]
    group.free(["_mo_grid_tmp"])
    return out


def _codensity_pairs(moA, moB, p0, p1):
    """Codensity strip for full pair indices [p0:p1], pair = a*nB + b -> (blk, ngrid)."""
    nB = moB.shape[0]
    idx = cp.arange(p0, p1)
    return moA[idx // nB] * moB[idx % nB]


def _codensity(moA, moB, pidx, qidx, P0, P1):
    """Codensity strip for ``PairLayout`` pairs [P0:P1] -> (P1-P0, ngrid)."""
    return moA[pidx[P0:P1]] * moB[qidx[P0:P1]]


def _cushion_bytes():
    """VRAM to hold back from the strip estimate so assembly is not razor-tight.

    At least 512 MiB, or 2% of currently free memory (whichever is larger).
    """
    return max(512 * 1024 ** 2, int(0.02 * _free_bytes()))



def _estimate_pair_blk(npair, ngrid, nB, pair_blk=None, free=None, mesh=None,
                       compact=False):
    """Estimate a fast strip size from live driver memory.

    The final ERI tensor has already been allocated when this is called.  The
    estimate budgets the remaining memory for rho, FFT output/workspace, vR,
    rho_q and the GEMM tile.  Strips are whole rows of the pair layout: with
    the full pair index that means multiples of ``nB`` pairs; with the
    compact ``p >= q`` index (``compact=True``) rows have variable length, so
    only the floor of one longest row (``nB`` pairs) applies and
    ``PairLayout.split`` packs whole rows into the budget.  ``free`` (bytes)
    overrides the current-device query (multi-GPU: plan from the slot with
    the least free memory).
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
    if not compact:
        raw = max(nB, (raw // nB) * nB)
    return min(raw, npair)


def _strip_task(ctx, task, keyA, keyB, mesh, out, on_gpu, layout):
    """Outer row strip [p0, p1) of one block, in sub-strips of the slot's
    ``sub_blk`` pairs.

    For each outer sub-strip the Coulomb potential of its codensities is
    formed once (FFT) and contracted against every inner strip of rows below
    the sub-strip's end -- the lower triangle of the symmetric Gram matrix.
    ``write_tile`` scatters each tile and its symmetry images into disjoint
    elements of ``out``, so a retry is idempotent and slots never write the
    same element.  Returns ``(gemm_flop, scatter_seconds)``.
    """
    st = ctx.state
    moA, moB, wcoulG = st[keyA], st[keyB], st["wcoulG"]
    pidx, qidx = st["pidx"], st["qidx"]
    ngrid = moA.shape[1]
    blk = int(st["sub_blk"])
    p0, p1 = task
    flop = 0.0
    t_scatter = 0.0
    for pa, pb in layout.split(p0, p1, blk):
        P0, P1 = layout.pairs(pa, pb)
        rho_p = _codensity(moA, moB, pidx, qidx, P0, P1)
        vR = gtools.ifft(gtools.fft(rho_p, mesh) * wcoulG, mesh).real
        rho_p = None
        for ra, rb in layout.split(0, pb, blk):
            Q0, Q1 = layout.pairs(ra, rb)
            rho_q = _codensity(moA, moB, pidx, qidx, Q0, Q1)
            tile = vR.dot(rho_q.T)
            flop += 2.0 * (P1 - P0) * (Q1 - Q0) * ngrid
            rho_q = None
            t0 = time.perf_counter()
            if not on_gpu:
                tile = cp.asnumpy(tile)
            write_tile(out, tile, layout, pa, pb, ra, rb)
            t_scatter += time.perf_counter() - t0
            tile = None
        vR = None
    return flop, t_scatter


def _make_strip_shrink(min_blk):
    def _shrink(ctx, task, exc):
        blk = int(ctx.state["sub_blk"])
        if blk <= min_blk:
            return False
        ctx.state["sub_blk"] = max(min_blk, ((blk // 2) // min_blk) * min_blk)
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
    compact = keyA == keyB          # same MO set both sides: (p,q) ~ (q,p)
    layout = PairLayout(nA, nB, compact)
    npair = layout.npair
    out_bytes = (nA * nB) ** 2 * 8
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
        blk = _estimate_pair_blk(npair, ngrid, layout.min_blk, pair_blk=pair_blk,
                                 mesh=mesh, compact=compact)
        # A retained result must leave enough space for useful FFT strips.
        if pair_blk is None and blk <= min(npair, 4 * layout.min_blk):
            del out
            _reclaim_gpu()
            on_gpu = False
    if not on_gpu:
        out = np.empty((nA, nA, nB, nB), dtype=np.float64)
        blk = _estimate_pair_blk(npair, ngrid, layout.min_blk, pair_blk=pair_blk,
                                 free=group.min_free_bytes(), mesh=mesh,
                                 compact=compact)
    stats = {
        "name": name,
        "compact": bool(compact),
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
    # Outer strips are whole rows of the pair layout.  High rows own more
    # lower-triangle tiles, so hand them out first for load balance.
    tasks = layout.split(0, nA, blk)[::-1]
    nblk = len(tasks)
    ntiles = layout.tiles(tasks, blk)
    planned_flop = layout.gemm_flop(ngrid, blk)
    print(
        f"{_ts()} [gpu_ao2mo] {name} final tensor {out_bytes/1e9:.2f} GB "
        f"on {location}; pair_blk={blk} (analytic); "
        f"free≈{free_before/1e9:.2f}→{stats['free_after_output_bytes']/1e9:.2f} GB", flush=True)
    print(
        f"{_ts()} [gpu_ao2mo]   {'compact p>=q pairs, ' if compact else ''}lower-triangle "
        f"tiles: strips={nblk} tiles={ntiles} GEMM={planned_flop/1e15:.2f} PFLOP; "
        f"pair_blk={blk} on {group.nslots} slot(s)", flush=True)
    stats["nstrips"] = nblk
    stats["gemms"] = ntiles
    stats["gemm_flop_planned"] = float(planned_flop)

    def _init(ctx):
        ctx.state["sub_blk"] = blk
        pi, qi = layout.index_arrays()
        ctx.state["pidx"] = cp.asarray(pi)
        ctx.state["qidx"] = cp.asarray(qi)
    group.each(_init)
    done = [0]
    lock = threading.Lock()
    report_every = max(1, nblk // 4)

    def _work(ctx, task):
        res = _strip_task(ctx, task, keyA, keyB, mesh, out, on_gpu, layout)
        with lock:
            done[0] += 1
            k = done[0]
        if nblk > 1 and (k % report_every == 0 or k == nblk):
            print(f"{_ts()} [gpu_ao2mo]   strip {k}/{nblk} done", flush=True)
        return res

    started = time.perf_counter()
    try:
        results, run_stats = group.run(tasks, _work, shrink=_make_strip_shrink(layout.min_blk),
                                       label=f"ao2mo {name}")
    finally:
        group.free(["pidx", "qidx"])
    if not on_gpu and not force_host:
        uploaded = cp.asarray(out)
        del out
        out = uploaded
    _sync()
    stats["seconds"] = time.perf_counter() - started
    stats["gemm_flop"] = float(sum(r[0] for r in results))
    stats["scatter_seconds"] = float(sum(r[1] for r in results))
    stats["tflops"] = stats["gemm_flop"] / max(stats["seconds"], 1e-9) / 1e12
    stats["retries"] = int(sum(run_stats["retries_per_slot"]))
    stats["final_pair_blk_per_slot"] = [int(c.state["sub_blk"]) for c in group.ctxs]
    stats["min_free_bytes"] = int(min(
        [b for b in run_stats["min_free_bytes_per_slot"] if b is not None] + [stats["min_free_bytes"]]))
    stats["multi_gpu"] = run_stats
    print(
        f"{_ts()} [gpu_ao2mo]   {name} done in {stats['seconds']:.1f} s: "
        f"{stats['gemm_flop']/1e15:.3f} PFLOP at {stats['tflops']:.1f} TFLOP/s "
        f"(tile scatter {stats['scatter_seconds']:.1f} s{'' if on_gpu else ' host'})",
        flush=True)
    return out, stats


def gpu_ao2mo_blocks(cell, cocc, cvir, mesh, pair_blk=None, return_gpu=False, group=None):
    """Return direct-assembled vvvv, oovv, oooo tensors in physicist layout.

    ``group`` (``lib_pprpa.gpu_multi.DeviceGroup``, default LIB_PPRPA_GPUS slots)
    dispatches both the MO-grid chunks and the pair strips over GPUs; the MO
    grids are then replicated to every slot.
    """
    global LAST_TELEMETRY
    from lib_pprpa.gpu_multi import default_group
    group = group or default_group()
    _free_pool()
    _sync()
    total_started = time.perf_counter()
    free_started, total_bytes = cp.cuda.runtime.memGetInfo()

    grid_started = time.perf_counter()
    # The grids land straight in every slot's state -- no host copy to broadcast.
    grid_stats = _mos_on_grid(cell, [cocc, cvir], ["moO", "moV"], mesh, group=group)
    no, nv = int(np.asarray(cocc).shape[1]), int(np.asarray(cvir).shape[1])
    ng = int(grid_stats["ngrid"])
    coulG = cp.asarray(gtools.get_coulG(cell, mesh=mesh))
    wcoulG = coulG * (cell.vol / ng)
    _sync()
    group.broadcast(wcoulG, "wcoulG")
    grid_bytes = (no + nv) * ng * 8 + coulG.nbytes + wcoulG.nbytes
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

    stage_host = not fits_resident(
        no, nv, extra_bytes=grid_bytes, total_bytes=total_bytes)
    if os.environ.get("GPU_AO2MO_FORCE_HOST", "").strip() in ("1", "true", "yes"):
        stage_host = True       # benchmark the host-staged (NV216) write path on a small cell
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
    del coulG, wcoulG
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
        "mo_grid": grid_stats,
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
