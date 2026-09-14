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

Here the exchange is built from (AO, active-MO) codensities instead:

    K_pq = sum_jk (pj|kq) D_jk                      (fft_jk.get_k, hermi=0 convention)
         = sum_m  int int ao_p(r) phiL_m(r) v(r-r') phiR_m(r') ao_q(r')
    phiL = ao . L,  phiR = ao . R                    (r x ngrid each)

so per AO-row block only  blk x r x ngrid  codensities are FFT'd, and the only
nao x ngrid array is the AO grid itself.  Blocks over AO rows and over the rank
index are sized from live free VRAM and halved on OOM.

Validated against ``gpu4pyscf.pbc.df.fft_jk.get_k(hermi=0)`` for symmetric and
antisymmetric low-rank densities (tests/test_gpu_fft_k_lowrank.py).
"""
from __future__ import annotations

import time

import numpy as np
import cupy as cp
from gpu4pyscf.pbc import tools as gtools
from gpu4pyscf.pbc.dft import numint as gnumint
from gpu4pyscf.lib.cupy_helper import contract

import threading

from lib_pprpa.gpu_multi import _free_bytes, _reclaim_gpu, default_group
from lib_pprpa.gpu_mem import max_fft_batch
from lib_pprpa.pprpa_util import tstamp


LAST_TELEMETRY = {}

# bytes per (row, rank, grid) element held transiently in one FFT batch:
# rho (8) + complex copy for cuFFT (16) + vG (16) + vR (16) + contract scratch (8)
_BYTES_PER_ELEMENT = 64


def get_last_telemetry():
    return dict(LAST_TELEMETRY)


def _log(msg):
    print(f"{tstamp()} [gpu_fft_k] {msg}", flush=True)


def ao_on_grid(cell, mesh):
    """AO values on the uniform grid as a C-contiguous (nao, ngrid) cupy array."""
    coords = cell.gen_uniform_grids(mesh)
    ao = gnumint.eval_ao_kpts(cell, coords, kpts=np.zeros((1, 3)), deriv=0)[0]
    return cp.ascontiguousarray(cp.asarray(ao).T)


def _plan_blocks(nao, rank, ngrid, row_blk=None, rank_blk=None, reserve_frac=0.15,
                 free=None, mesh=None):
    """Choose (row_blk, rank_blk) so one FFT batch fits in the free VRAM
    (``free``: bytes to plan from, default the current device's)."""
    if row_blk is not None and rank_blk is not None:
        return max(1, min(nao, int(row_blk))), max(1, min(rank, int(rank_blk)))
    if free is None:
        free = _free_bytes()
    reserve = max(1024 ** 3, int(free * reserve_frac))
    usable = max(0, free - reserve)
    n_el = max(1, usable // (_BYTES_PER_ELEMENT * ngrid))
    # row_blk * rank_blk transforms form one batched cuFFT plan (cuFFT element cap).
    n_el = min(n_el, max_fft_batch(ngrid, mesh=mesh))
    if rank_blk is None:
        rank_blk = min(rank, n_el)
    rank_blk = max(1, min(rank, int(rank_blk)))
    if row_blk is None:
        row_blk = max(1, n_el // rank_blk) if rank_blk == rank else 1
    row_blk = max(1, min(nao, int(row_blk)))
    return row_blk, rank_blk


def _row_task(ctx, task, K_host, weight, mesh):
    """K rows [p0, p1) -> K_host (numpy), in sub-blocks of the slot's row_blk /
    rank_blk.  Rows are plain assignments, so a retry after an OOM is idempotent."""
    st = ctx.state
    ao, phiL, phiR, coulG = st["ao"], st["phiL"], st["phiR"], st["coulG"]
    rank, ngrid = phiL.shape
    p0, p1 = task
    r0 = p0
    while r0 < p1:
        rb, kb = int(st["row_blk"]), int(st["rank_blk"])
        r1 = min(p1, r0 + rb)
        vR_dm = cp.zeros((r1 - r0, ngrid))
        for m0 in range(0, rank, kb):
            m1 = min(rank, m0 + kb)
            rho = ao[r0:r1, None, :] * phiL[None, m0:m1, :]      # (nb, kc, ngrid)
            vG = gtools.fft(rho.reshape(-1, ngrid), mesh)
            rho = None
            vG *= coulG
            vR = gtools.ifft(vG, mesh)
            vG = None
            vR = vR.real.reshape(r1 - r0, m1 - m0, ngrid)
            vR_dm += contract("img,mg->ig", vR, phiR[m0:m1])
            vR = None
        K_host[r0:r1] = cp.asnumpy(weight * (vR_dm @ ao.T))
        vR_dm = None
        r0 = r1


def _row_shrink(ctx, task, exc):
    st = ctx.state
    if st["row_blk"] > 1:
        st["row_blk"] = max(1, int(st["row_blk"]) // 2)
    elif st["rank_blk"] > 1:
        st["rank_blk"] = max(1, int(st["rank_blk"]) // 2)
    else:
        return False
    return True


def get_k_lowrank(cell, mesh, factors, hermi=0, exxdiv=None, ao=None,
                  row_blk=None, rank_blk=None, group=None, verbose=True):
    """Exchange matrices K[D] for low-rank AO densities D = L @ R.T (Gamma point).

    Args:
        cell: pyscf.pbc Cell.
        mesh: FFT mesh (3 ints).
        factors: one ``(L, R)`` pair or a sequence of pairs; L, R are
            (nao, r) numpy/cupy arrays (r may differ between pairs).
        hermi: accepted for signature compatibility with ``mf.get_k``; the
            result never assumes symmetry (equivalent to hermi=0).
        exxdiv: passed to ``get_coulG`` (None = bare 4pi/G^2, G=0 dropped).
        ao: optional precomputed (nao, ngrid) AO grid (see ``ao_on_grid``).
        row_blk, rank_blk: override the automatic block sizes.
        group: ``lib_pprpa.gpu_multi.DeviceGroup`` (default: LIB_PPRPA_GPUS
            slots).  The AO grid and coulG are replicated per slot; AO row
            blocks are the dispatched tasks.

    Returns:
        numpy (nao, nao) for a single pair, else (nset, nao, nao), with
        K_pq = sum_jk (pj|kq) D_jk  -- the same convention as
        ``gpu4pyscf.pbc.df.fft_jk.get_k(..., hermi=0)``.
    """
    single = (isinstance(factors, (tuple, list)) and len(factors) == 2
              and getattr(factors[0], "ndim", 0) == 2)
    pairs = [factors] if single else list(factors)
    mesh = np.asarray(mesh, dtype=int)
    ngrid = int(np.prod(mesh))
    nao = cell.nao
    started = time.perf_counter()
    group = group or default_group()

    own_ao = ao is None
    if own_ao:
        _reclaim_gpu()
        ao = ao_on_grid(cell, mesh)
    assert ao.shape == (nao, ngrid), ao.shape
    coulG = cp.asarray(gtools.get_coulG(cell, k=np.zeros(3), exx=exxdiv, mesh=mesh))
    weight = cell.vol / ngrid
    group.broadcast(ao, "ao")
    group.broadcast(coulG, "coulG")

    out = np.empty((len(pairs), nao, nao))
    stats = []
    for iset, (L, R) in enumerate(pairs):
        Lh = np.asarray(cp.asnumpy(L) if isinstance(L, cp.ndarray) else L, dtype=np.float64)
        Rh = np.asarray(cp.asnumpy(R) if isinstance(R, cp.ndarray) else R, dtype=np.float64)
        assert Lh.shape == Rh.shape and Lh.shape[0] == nao, (Lh.shape, Rh.shape)
        rank = Lh.shape[1]

        def _setup(ctx):
            a = ctx.state["ao"]
            ctx.state["phiL"] = cp.asarray(Lh).T @ a       # (r, ngrid)
            ctx.state["phiR"] = cp.asarray(Rh).T @ a
            _reclaim_gpu()
            return _free_bytes()
        free_min = min(group.each(_setup))
        rb, kb = _plan_blocks(nao, rank, ngrid, row_blk, rank_blk, free=free_min, mesh=mesh)

        def _init(ctx):
            ctx.state["row_blk"] = rb
            ctx.state["rank_blk"] = kb
        group.each(_init)
        tasks = [(p0, min(nao, p0 + rb)) for p0 in range(0, nao, rb)]
        if verbose:
            _log(f"set {iset + 1}/{len(pairs)}: nao={nao} rank={rank} ngrid={ngrid} "
                 f"row_blk={rb} rank_blk={kb} free~{free_min/1e9:.2f} GB "
                 f"({len(tasks)} row batches x {(rank + kb - 1)//kb} rank chunks on "
                 f"{group.nslots} slot(s))")
        K_host = out[iset]
        done = [0]
        lock = threading.Lock()
        report_every = max(1, len(tasks) // 10)

        def _work(ctx, task):
            _row_task(ctx, task, K_host, weight, mesh)
            if verbose and len(tasks) > 4:
                with lock:
                    done[0] += 1
                    k = done[0]
                if k % report_every == 0:
                    _log(f"  set {iset + 1}: row batches {k}/{len(tasks)} done "
                         f"({time.perf_counter() - started:.0f} s elapsed)")
        _results, run_stats = group.run(tasks, _work, shrink=_row_shrink,
                                        label=f"fft_k set{iset + 1}")
        final = [(int(c.state["row_blk"]), int(c.state["rank_blk"])) for c in group.ctxs]
        stats.append({"rank": int(rank), "row_blk": int(rb), "rank_blk": int(kb),
                      "final_blocks_per_slot": final,
                      "retries": int(sum(run_stats["retries_per_slot"])),
                      "multi_gpu": run_stats})
        group.free(["phiL", "phiR"])

    group.free(["ao", "coulG"])
    if own_ao:
        del ao
        _reclaim_gpu()
    elapsed = time.perf_counter() - started
    LAST_TELEMETRY.clear()
    LAST_TELEMETRY.update({"nao": int(nao), "ngrid": ngrid, "nset": len(pairs),
                           "sets": stats, "seconds": float(elapsed),
                           "gpu_slots": list(group.devices)})
    if verbose:
        _log(f"done: {len(pairs)} set(s) in {elapsed:.1f} s")
    return out[0] if single else out


def attach_lowrank_getk(mf, cell, mesh=None, group=None):
    """Give a mean-field object a ``get_k_lowrank(factors, **kw)`` method so that
    ``lib_pprpa.grad.pprpa.make_rdm1_relaxed_rhf_pprpa`` never forms the dense
    nao x nao amplitude densities.  ``group`` selects the GPU slots."""
    if mesh is None:
        mesh = cell.mesh

    def _gk_lowrank(factors, hermi=0, **kw):
        return get_k_lowrank(cell, mesh, factors, hermi=hermi, exxdiv=None, group=group)
    mf.get_k_lowrank = _gk_lowrank
    return mf
