"""GPU (cupy) Gamma-point FFT exchange for low-rank densities  D = L @ R.T.

Why this exists
---------------
The pp-RPA relaxed-density gradient needs K[X_ao] and K[Y_ao] for the amplitude
densities  X_ao = C_a x C_a^T  and  Y_ao = C_i y C_i^T, whose rank is bounded by
the active space (e.g. 300) rather than by nao.  The dense gpu4pyscf
``fft_jk.get_k`` throws that structure away: it holds ``ao`` (nao x ngrid),
``ao_dms`` and ``vR_dm`` (nset x nao x ngrid each) and FFTs nao x nao
codensities per AO row.  At the 216-atom NV cell (nao=2795, mesh 159^3) each of
those is 90-180 GB and the kernel cannot fit on any single GPU.

Here the exchange is built from low-rank codensities instead.  With
D = L R^T, phiL = ao . L and phiR = ao . R (r x ngrid each),

    K_pq = sum_jk (pj|kq) D_jk                       (fft_jk.get_k, hermi=0 convention)
         = sum_m  int int ao_p(r) phiL_m(r) v(r-r') phiR_m(r') ao_q(r')
         = sum_g  ao_p(g) U_q(g),      U_q = sum_m phiL_m * W[phiR_m ao_q]

so the Coulomb operator W is applied to the (m, q) codensities and the result
is contracted with phiL and, last, with the AO grid.  Two entry points:

* ``ket=None`` -- the full nao x nao K: nao x r transforms, the AO grid
  resident.
* ``ket=C`` (nao x nk) -- K @ C directly: the codensities are phiR_m phi_k with
  phi = ao . C, so only r x nk transforms, and the AO grid is never held (it
  is evaluated in grid chunks for the final contraction).  The relaxed
  density only ever uses ``mo_coeff.T @ K @ orbp`` with orbp the active
  orbitals, so this is the production path: 300 x 600 instead of 300 x 2795
  transforms at NV216, without the 90 GB AO grid.

Planner: the FFT batch (pairs per transform batch, ``fft_blk``) and the row
block of the accumulator U (``row_blk``) are sized separately from live VRAM
with the per-point costs from ``gpu_coulomb`` -- the FFT chain is the cost
here, so it gets half the budget; the row block only has to keep the final
GEMM / grid pass efficient and the task count balanced.  Both halve on OOM.

Validated against ``gpu4pyscf.pbc.df.fft_jk.get_k(hermi=0)`` for symmetric,
antisymmetric and general low-rank densities, full and ket-projected
(tests/test_gpu_fft_k_lowrank.py).
"""
from __future__ import annotations

import threading
import time

import numpy as np
import cupy as cp
from gpu4pyscf.pbc import tools as gtools
from gpu4pyscf.pbc.dft import numint as gnumint
from gpu4pyscf.lib.cupy_helper import contract

from lib_pprpa.gpu_coulomb import (balanced_chunk, codensity, coulomb_potential,
                                   fft_batch_bytes, half_kernel, plan_fft_batch, use_rfft)
from lib_pprpa.gpu_multi import _free_bytes, _reclaim_gpu, default_group
from lib_pprpa.pprpa_util import tstamp


LAST_TELEMETRY = {}

_KPTS0 = np.zeros((1, 3))
# Share of the usable VRAM handed to the FFT batch (the dominant cost here).
_FFT_FRAC = 0.5
# Rows of U per task: enough for an efficient final contraction, few enough
# that a device group has tasks to balance.
_ROW_BLK_CAP = 256
_MIN_TASKS = 8


def get_last_telemetry():
    return dict(LAST_TELEMETRY)


def _log(msg):
    print(f"{tstamp()} [gpu_fft_k] {msg}", flush=True)


def ao_on_grid(cell, mesh):
    """AO values on the uniform grid as a C-contiguous (nao, ngrid) cupy array."""
    coords = cell.gen_uniform_grids(mesh)
    ao = gnumint.eval_ao_kpts(cell, coords, kpts=_KPTS0, deriv=0)[0]
    return cp.ascontiguousarray(cp.asarray(ao).T)


def _plan_blocks(nrows, rank, ngrid, row_blk=None, fft_blk=None, reserve_frac=0.15,
                 free=None, mesh=None, rfft=True, rank_blk=None):
    """Choose ``(row_blk, fft_blk)``.

    ``fft_blk`` is the number of (row, m) codensities per FFT batch: whole
    rows (a multiple of ``rank``) when the budget allows, otherwise balanced
    chunks of one row's ``rank`` (300 under a cap of 269 -> 150 + 150, never
    269 + 31).  ``row_blk`` is the number of rows of the accumulator U per
    task, from what the FFT batch leaves, capped so a device group keeps at
    least ``_MIN_TASKS`` tasks.  ``rank_blk`` is the legacy name of a forced
    within-row chunk (``fft_blk`` below ``rank``).  ``free`` overrides the
    current device's free bytes.
    """
    nrows, rank, ngrid = int(nrows), int(rank), int(ngrid)
    if fft_blk is None and rank_blk is not None:
        fft_blk = int(rank_blk)
    if free is None:
        free = _free_bytes()
    reserve = max(1024 ** 3, int(free * reserve_frac))
    usable = max(0, free - reserve)
    if fft_blk is not None:
        fblk = max(1, min(int(fft_blk), nrows * rank))   # an override is not rebalanced
    else:
        fblk = plan_fft_batch(usable * _FFT_FRAC, ngrid, mesh, rfft, nrows * rank)
        if fblk < rank:
            fblk = balanced_chunk(rank, fblk)         # within one row, balanced
    if fblk >= rank:
        fblk = (fblk // rank) * rank                  # whole rows
    if row_blk is not None:
        rb = max(1, min(nrows, int(row_blk)))
    else:
        remaining = max(0, usable - fft_batch_bytes(fblk, ngrid, rfft))
        rb = remaining // (8 * ngrid)                 # the U block, 8 B per row-point
        rb = max(1, min(nrows, int(rb), _ROW_BLK_CAP, -(-nrows // _MIN_TASKS)))
        if fblk >= rank:
            rb = min(nrows, max(rb, fblk // rank))    # never split one FFT batch across tasks
    return int(rb), int(fblk)


def _fft_chunks(a0, a1, rank, fblk):
    """(i0, i1, m0, m1) FFT batches covering rows [a0, a1) x all m: whole rows
    when ``fblk >= rank``, else balanced m-chunks of each row."""
    if fblk >= rank:
        per = max(1, fblk // rank)
        for i0 in range(a0, a1, per):
            yield i0, min(a1, i0 + per), 0, rank
    else:
        for i in range(a0, a1):
            for m0 in range(0, rank, fblk):
                yield i, i + 1, m0, min(rank, m0 + fblk)


def _work_buffers(st, rb, fblk, ngrid):
    bufs = st.get("bufs")
    if bufs is None or bufs[0].shape[0] != rb or bufs[1].shape[0] != fblk:
        st["bufs"] = None
        _reclaim_gpu()
        bufs = (cp.empty((rb, ngrid), dtype=cp.float64), cp.empty((fblk, ngrid), dtype=cp.float64))
        st["bufs"] = bufs
    return bufs


def _row_task(ctx, task, mesh, consume):
    """Rows [r0, r1) of U in blocks of the slot's ``row_blk``:

        U_k(g) = sum_m phiL_m(g) * W[phiR_m phiK_k](g)

    with the (k, m) codensities transformed ``fft_blk`` at a time, then
    ``consume(ctx, a0, a1, U_blk)`` for each finished block (plain
    assignments, so a retry after an OOM is idempotent)."""
    st = ctx.state
    phiK, phiL, phiR = st["phiK"], st["phiL"], st["phiR"]
    rfft = bool(st["rfft"])
    w = st["w_half"] if rfft else st["coulG"]
    kidx, midx = st["kidx"], st["midx"]
    rank, ngrid = phiL.shape
    rb, fblk = int(st["row_blk"]), int(st["fft_blk"])
    U_buf, rho_buf = _work_buffers(st, rb, fblk, ngrid)
    r0, r1 = task
    for a0 in range(r0, r1, rb):
        a1 = min(r1, a0 + rb)
        U = U_buf[:a1 - a0]
        U[...] = 0.0
        for i0, i1, m0, m1 in _fft_chunks(a0, a1, rank, fblk):
            P0 = i0 * rank + m0
            P1 = (i1 - 1) * rank + m1
            n = P1 - P0
            rho = codensity(phiK, phiR, kidx, midx, P0, P1, out=rho_buf[:n])
            V = coulomb_potential(rho, mesh, w, rfft).reshape(i1 - i0, m1 - m0, ngrid)
            U[i0 - a0:i1 - a0] += contract("img,mg->ig", V, phiL[m0:m1])
            V = None
        consume(ctx, a0, a1, U)


def _row_shrink(ctx, task, exc):
    st = ctx.state
    rank = int(st["phiL"].shape[0])
    rb, fblk = int(st["row_blk"]), int(st["fft_blk"])
    if rb > 1:
        st["row_blk"] = max(1, rb // 2)
        if fblk > rank:
            st["fft_blk"] = max(rank, (min(fblk, st["row_blk"] * rank) // rank) * rank)
        return True
    if fblk > 1:
        st["fft_blk"] = balanced_chunk(rank, fblk // 2) if fblk <= rank else rank
        return True
    return False


def _grid_chunk(nao, ncol, ngrid, frac=0.4, floor=4096):
    free = _free_bytes()
    per_point = 16 * int(nao) + 8 * int(ncol)
    return int(max(min(floor, ngrid), min(ngrid, int(free * frac) // max(1, per_point))))


def get_k_lowrank(cell, mesh, factors, hermi=0, exxdiv=None, ao=None,
                  row_blk=None, rank_blk=None, fft_blk=None, group=None, verbose=True,
                  ket=None, rfft=None):
    """Exchange matrices K[D] for low-rank AO densities D = L @ R.T (Gamma point).

    Args:
        cell: pyscf.pbc Cell.
        mesh: FFT mesh (3 ints).
        factors: one ``(L, R)`` pair or a sequence of pairs; L, R are
            (nao, r) numpy/cupy arrays (r may differ between pairs).
        hermi: accepted for signature compatibility with ``mf.get_k``; the
            result never assumes symmetry (equivalent to hermi=0).
        exxdiv: passed to ``get_coulG`` (None = bare 4pi/G^2, G=0 dropped).
        ao: optional precomputed (nao, ngrid) AO grid (see ``ao_on_grid``);
            required by the full-K path (built here if missing), optional
            for the ket path (which otherwise never holds it).
        row_blk, fft_blk: override the planner (rows of U per task, pairs
            per FFT batch).  ``rank_blk`` is the legacy name for a forced
            within-row chunk.
        group: ``lib_pprpa.gpu_multi.DeviceGroup`` (default: LIB_PPRPA_GPUS
            slots).  Grids are replicated per slot; row blocks of U are the
            dispatched tasks.
        ket: optional (nao, nk) coefficients; return ``K @ ket`` (nao, nk)
            with r x nk transforms instead of the full K with r x nao.
        rfft: real-to-complex Coulomb chain (default on, ``GPU_FFT_RFFT=0``
            for the complex chain).

    Returns:
        numpy (nao, nao) -- or (nao, nk) with ``ket`` -- for a single pair,
        else with a leading set axis, with
        K_pq = sum_jk (pj|kq) D_jk  -- the same convention as
        ``gpu4pyscf.pbc.df.fft_jk.get_k(..., hermi=0)``.
    """
    single = (isinstance(factors, (tuple, list)) and len(factors) == 2
              and getattr(factors[0], "ndim", 0) == 2)
    pairs = [factors] if single else list(factors)
    mesh = np.asarray(mesh, dtype=int)
    ngrid = int(np.prod(mesh))
    nao = cell.nao
    rfft = use_rfft(rfft)
    started = time.perf_counter()
    group = group or default_group()
    weight = cell.vol / ngrid

    def _host(a):
        return np.asarray(cp.asnumpy(a) if isinstance(a, cp.ndarray) else a, dtype=np.float64)

    ket_h = None if ket is None else _host(ket)
    if ket_h is not None:
        assert ket_h.shape[0] == nao, ket_h.shape
    nk = nao if ket_h is None else ket_h.shape[1]

    own_ao = ao is None and ket_h is None
    if own_ao:
        _reclaim_gpu()
        ao = ao_on_grid(cell, mesh)
    if ao is not None:
        assert ao.shape == (nao, ngrid), ao.shape
        group.broadcast(ao, "ao")
    coulG = cp.asarray(gtools.get_coulG(cell, k=np.zeros(3), exx=exxdiv, mesh=mesh))
    group.broadcast(coulG, "coulG")
    group.broadcast(half_kernel(coulG, mesh), "w_half")
    coords = None if ao is not None else cell.gen_uniform_grids(mesh)

    out = np.empty((len(pairs), nao, nk))
    stats = []
    for iset, (L, R) in enumerate(pairs):
        Lh, Rh = _host(L), _host(R)
        assert Lh.shape == Rh.shape and Lh.shape[0] == nao, (Lh.shape, Rh.shape)
        rank = Lh.shape[1]
        t_set = time.perf_counter()

        # ---- residents: phiL, phiR and the ket grid phiK ----------------------
        if ao is not None:
            def _setup(ctx):
                a = ctx.state["ao"]
                ctx.state["phiL"] = cp.asarray(Lh).T @ a
                ctx.state["phiR"] = cp.asarray(Rh).T @ a
                ctx.state["phiK"] = a if ket_h is None else cp.asarray(ket_h).T @ a
                _reclaim_gpu()
            group.each(_setup)
        else:
            from lib_pprpa.gpu_ao2mo import _mos_on_grid
            _mos_on_grid(cell, [Lh, Rh, ket_h], ["phiL", "phiR", "phiK"], mesh, group=group)
        kidx = np.repeat(np.arange(nk, dtype=np.int32), rank)
        midx = np.tile(np.arange(rank, dtype=np.int32), nk)

        def _plan(ctx):
            ctx.state["kidx"] = cp.asarray(kidx)
            ctx.state["midx"] = cp.asarray(midx)
            ctx.state["rfft"] = rfft
            if ket_h is not None:
                ctx.state["U"] = cp.empty((nk, ngrid), dtype=cp.float64)
                ctx.state["rows_done"] = []
            _reclaim_gpu()
            return _free_bytes()
        free_min = min(group.each(_plan))
        rb, fblk = _plan_blocks(nk, rank, ngrid, row_blk, fft_blk, free=free_min, mesh=mesh,
                                rfft=rfft, rank_blk=rank_blk)

        def _init(ctx):
            ctx.state["row_blk"] = rb
            ctx.state["fft_blk"] = fblk
        group.each(_init)
        tasks = [(p0, min(nk, p0 + rb)) for p0 in range(0, nk, rb)]
        if verbose:
            _log(f"set {iset + 1}/{len(pairs)}: nao={nao} rank={rank} nk={nk} ngrid={ngrid} "
                 f"row_blk={rb} fft_blk={fblk} {'r2c' if rfft else 'c2c'} "
                 f"free~{free_min/1e9:.2f} GB ({len(tasks)} row blocks x "
                 f"{max(1, -(-rank // min(rank, fblk)))} FFT chunks/row on {group.nslots} slot(s))")
        K_host = out[iset]
        done = [0]
        lock = threading.Lock()
        report_every = max(1, len(tasks) // 10)

        if ket_h is None:
            def _consume(ctx, a0, a1, U):
                K_host[:, a0:a1] = cp.asnumpy(weight * (ctx.state["ao"] @ U.T))
        else:
            def _consume(ctx, a0, a1, U):
                ctx.state["U"][a0:a1] = U
                ctx.state["rows_done"].append((a0, a1))

        def _work(ctx, task):
            _row_task(ctx, task, mesh, _consume)
            if verbose and len(tasks) > 4:
                with lock:
                    done[0] += 1
                    k = done[0]
                if k % report_every == 0:
                    _log(f"  set {iset + 1}: row blocks {k}/{len(tasks)} done "
                         f"({time.perf_counter() - started:.0f} s elapsed)")
        _results, run_stats = group.run(tasks, _work, shrink=_row_shrink,
                                        label=f"fft_k set{iset + 1}")
        t_strips = time.perf_counter() - t_set

        # ---- ket path: K[:, rows] = weight * ao . U[rows]^T, per slot ------------
        if ket_h is not None:
            def _finish(ctx):
                st = ctx.state
                st["bufs"] = None
                _reclaim_gpu()
                rows = sorted(set(st["rows_done"]))
                U = st["U"]
                Kp = cp.zeros((nao, nk), dtype=cp.float64)
                if "ao" in st:
                    for a0, a1 in rows:
                        Kp[:, a0:a1] = st["ao"] @ U[a0:a1].T
                else:
                    opt0 = gnumint._GTOvalOpt(cell, _KPTS0, deriv=0)
                    gch = _grid_chunk(nao, nk, ngrid)
                    for g0 in range(0, ngrid, gch):
                        g1 = min(ngrid, g0 + gch)
                        aog = gnumint.eval_ao_kpts(cell, coords[g0:g1], kpts=_KPTS0, deriv=0,
                                                   opt=opt0)[0]                  # (g, nao)
                        for a0, a1 in rows:
                            Kp[:, a0:a1] += aog.T @ U[a0:a1, g0:g1].T
                        aog = None
                Kp *= weight
                res = cp.asnumpy(Kp)
                Kp = None
                return res
            K_host[...] = np.sum(group.each(_finish), axis=0)
        t_total = time.perf_counter() - t_set
        final = [(int(c.state["row_blk"]), int(c.state["fft_blk"])) for c in group.ctxs]
        stats.append({"rank": int(rank), "nk": int(nk), "row_blk": int(rb), "fft_blk": int(fblk),
                      "rank_blk": int(min(rank, fblk)), "rfft": bool(rfft),
                      "free_at_plan_bytes": int(free_min),
                      "rank_chunks": int(-(-rank // min(rank, fblk))),
                      "row_batching": bool(rb > 1), "ntasks": len(tasks),
                      "transforms": int(nk * rank),
                      "strip_seconds": float(t_strips), "seconds": float(t_total),
                      "final_blocks_per_slot": final,
                      "retries": int(sum(run_stats["retries_per_slot"])),
                      "multi_gpu": run_stats})
        group.free(["phiL", "phiR", "phiK", "kidx", "midx", "U", "rows_done", "bufs"])

    group.free(["ao", "coulG", "w_half"])
    if own_ao:
        del ao
        _reclaim_gpu()
    elapsed = time.perf_counter() - started
    LAST_TELEMETRY.clear()
    LAST_TELEMETRY.update({"nao": int(nao), "nk": int(nk), "ngrid": ngrid, "nset": len(pairs),
                           "ket": ket_h is not None, "rfft": bool(rfft),
                           "sets": stats, "seconds": float(elapsed),
                           "gpu_slots": list(group.devices)})
    if verbose:
        _log(f"done: {len(pairs)} set(s) in {elapsed:.1f} s")
    return out[0] if single else out


def attach_lowrank_getk(mf, cell, mesh=None, group=None):
    """Give a mean-field object a ``get_k_lowrank(factors, **kw)`` method so that
    ``lib_pprpa.grad.pprpa.make_rdm1_relaxed_rhf_pprpa`` never forms the dense
    nao x nao amplitude densities.  ``group`` selects the GPU slots; keyword
    arguments (``ket``, block overrides) pass through."""
    if mesh is None:
        mesh = cell.mesh

    def _gk_lowrank(factors, hermi=0, **kw):
        kw.setdefault("group", group)
        return get_k_lowrank(cell, mesh, factors, hermi=hermi, exxdiv=None, **kw)
    mf.get_k_lowrank = _gk_lowrank
    return mf
