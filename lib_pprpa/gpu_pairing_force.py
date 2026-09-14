"""Low-rank FFT pairing-exchange force for the Gamma-point pp-RPA gradient.

The pp-RPA gradient contains the "pairing" exchange term (CPU reference,
``lib_pprpa/grad/pprpa_gamma.py``)::

    vk[x]_{il} = -sum_{jk} (∇_x i j | k l) X_jk          (kmf_grad.get_k, hermi=0)
    de[A, x]  += 2 * sum_{i in A} sum_l vk[x]_{il} X_il

with the amplitude density X = C_vir x C_vir^T + C_occ y C_occ^T.  Its rank is
bounded by the active space (<= nocc + nvir), so X = L R^T with
L = [C_vir x, C_occ y] and R = [C_vir, C_occ].  Inserting the factors and
moving to the FFT grid gives

    de[A, x] = -2 * sum_{i in A} sum_m L_im G[x, i, m]
    G[x,i,m] = (vol/ngrid) * sum_g ∂_x i(g) W_m(g)
    W_m(g)   = sum_n phiL_n(g) V_nm(g)
    V_nm     = ifft(coulG * fft(phiR_n phiR_m))       (exxdiv=None: G=0 term zero)
    phiR = ao . R  (active MOs on the grid),  phiL = ao . L

so the expensive part is one FFT pass over the r(r+1)/2 active-MO codensity
pairs (V is symmetric in n, m) -- the same kernel as the ao2mo strips -- plus
one chunked pass over the gradient AOs.  At the 216-atom NV cell (nao=2795,
mesh 159^3, r=600) this replaces 35,890 nao^2-sized AFT G-blocks (16 h) by
180,300 codensity FFTs (~1-3 h).  The formula is exact for a general X, so
singlet (symmetric) and triplet (antisymmetric) amplitudes need no special
bookkeeping.  The sign: ``eval_ao(deriv=1)`` is the electron-coordinate
gradient, the nuclear derivative flips it, hence the -2.

Multi-GPU: strips are global tasks dispatched by ``gpu_multi.DeviceGroup``;
every slot holds its own phiR, phiL, coulG and a private W accumulator, runs
the gradient-AO pass on its partial W (linear) and returns G; the host sums G.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import cupy as cp
from gpu4pyscf.pbc import tools as gtools
from gpu4pyscf.pbc.dft import numint as gnumint

from lib_pprpa.gpu_mem import max_fft_batch
from lib_pprpa.gpu_multi import DeviceGroup, _free_bytes, _reclaim_gpu, default_group, log


LAST_TELEMETRY = {}

_KPTS0 = np.zeros((1, 3))           # one object: eval_ao_kpts asserts `kpts is opt.kpts`
# transient bytes per (pair, grid point) in one strip: gathers (2x8), complex
# fft copy (16), vG (16), V (16), scratch (8) -- with a reserve on top
_BYTES_PER_PAIR_POINT = 64


def get_last_telemetry():
    return dict(LAST_TELEMETRY)


def _log(msg):
    log(f"[pairing_k] {msg}")


# --------------------------------------------------------------------------
# grid factors
# --------------------------------------------------------------------------
def _grid_chunk(bytes_per_point, ngrid, frac=0.4, floor=4096):
    """Grid points per chunk so a chunk uses ~``frac`` of the free VRAM."""
    budget = int(_free_bytes() * frac)
    return int(max(min(floor, ngrid), min(ngrid, budget // max(int(bytes_per_point), 1))))


def lr_on_grid(cell, mesh, coords, L, R, gchunk=None):
    """phiL = ao . L and phiR = ao . R as (r, ngrid) cupy arrays, built in grid
    chunks so the full (nao, ngrid) AO array is never materialised."""
    nao, r = L.shape
    ngrid = len(coords)
    Lg = cp.asarray(L, dtype=np.float64)
    Rg = cp.asarray(R, dtype=np.float64)
    phiL = cp.empty((r, ngrid))
    phiR = cp.empty((r, ngrid))
    if gchunk is None:
        gchunk = _grid_chunk(nao * 8 * 2 + 2 * r * 8, ngrid)
    opt0 = gnumint._GTOvalOpt(cell, _KPTS0, deriv=0)
    for g0 in range(0, ngrid, int(gchunk)):
        g1 = min(ngrid, g0 + int(gchunk))
        ao = gnumint.eval_ao_kpts(cell, coords[g0:g1], kpts=_KPTS0, deriv=0, opt=opt0)[0]
        aoT = ao.T                              # (nao, g), C-contiguous
        phiR[:, g0:g1] = Rg.T @ aoT
        phiL[:, g0:g1] = Lg.T @ aoT
        ao = aoT = None
    return phiL, phiR


def grad_ao_pass(cell, mesh, coords, W, gchunk=None):
    """G[x, i, m] = (vol/ngrid) sum_g ∂_x i(g) W_m(g), in grid chunks (cupy (3, nao, r))."""
    r, ngrid = W.shape
    nao = cell.nao
    G = cp.zeros((3, nao, r))
    opt1 = gnumint._GTOvalOpt(cell, _KPTS0, deriv=1)
    if gchunk is None:
        gchunk = _grid_chunk(4 * nao * 8 + 2 * r * 8, ngrid)
    for g0 in range(0, ngrid, int(gchunk)):
        g1 = min(ngrid, g0 + int(gchunk))
        ao4 = gnumint.eval_ao_kpts(cell, coords[g0:g1], kpts=_KPTS0, deriv=1, opt=opt1)[0]
        Wc = cp.ascontiguousarray(W[:, g0:g1])  # (r, g)
        for x in range(3):
            G[x] += ao4[1 + x].T @ Wc.T          # (nao, g) @ (g, r)
        ao4 = Wc = None
    G *= cell.vol / ngrid
    return G


def assemble_force(G, L, aoslices):
    """de[A, x] = -2 sum_{i in A} sum_m L_im G[x, i, m]   (numpy (natm, 3))."""
    G = np.asarray(G)
    L = np.asarray(L)
    t = np.einsum('xim,im->xi', G, L)
    de = np.zeros((len(aoslices), 3))
    for ia, sl in enumerate(aoslices):
        p0, p1 = int(sl[2]), int(sl[3])
        de[ia] = -2.0 * t[:, p0:p1].sum(axis=1)
    return de


# --------------------------------------------------------------------------
# pair enumeration and strips
# --------------------------------------------------------------------------
def pair_index(r, symmetric=True):
    """Flat pair list (n_idx, m_idx) ordered by n, plus row_start[n] (first flat
    index with that n; row_start[r] = npair)."""
    if symmetric:
        n_idx, m_idx = np.triu_indices(r)
    else:
        n_idx, m_idx = np.indices((r, r)).reshape(2, -1)
    n_idx = np.ascontiguousarray(n_idx, dtype=np.int64)
    m_idx = np.ascontiguousarray(m_idx, dtype=np.int64)
    row_start = np.searchsorted(n_idx, np.arange(r + 1)).astype(np.int64)
    return n_idx, m_idx, row_start


def _segments(p0, p1, n_idx, m_idx, row_start):
    """Yield (n, local_slice, ma, mb) for the fixed-n runs inside flat [p0, p1)."""
    q = p0
    while q < p1:
        n = int(n_idx[q])
        q1 = min(p1, int(row_start[n + 1]))
        ma = int(m_idx[q])
        mb = int(m_idx[q1 - 1]) + 1
        yield n, slice(q - p0, q1 - p0), ma, mb
        q = q1


def plan_strip(ngrid, npair, free=None, reserve_frac=0.15, mesh=None):
    """Pairs per FFT strip from free VRAM, the cuFFT plan cap and an env override."""
    env = os.environ.get("PPRPA_PAIRING_BLK")
    if env:
        return max(1, min(npair, int(env)))
    if free is None:
        free = _free_bytes()
    reserve = max(1024 ** 3, int(free * reserve_frac))
    usable = max(0, free - reserve)
    blk = usable // (_BYTES_PER_PAIR_POINT * ngrid)
    return int(max(1, min(npair, blk, max_fft_batch(ngrid, mesh=mesh))))


def pairing_strip(ctx, task, n_idx, m_idx, row_start, mesh, symmetric=True):
    """Accumulate W for flat pairs [p0, p1) in sub-strips of ``ctx.state['blk']``.

    Two phases per sub-strip: the FFT phase may OOM but touches nothing
    persistent; the accumulation phase only writes into preallocated buffers.
    Progress is recorded per task so a retry after an OOM resumes at the first
    unfinished sub-strip and never double counts.
    """
    st = ctx.state
    phiR, phiL, coulG, W = st["phiR"], st["phiL"], st["coulG"], st["W"]
    scratch, rowbuf = st["scratch"], st["rowbuf"]
    progress = st.setdefault("progress", {})
    p0, p1 = task
    q0 = progress.get(task, p0)
    while q0 < p1:
        q1 = min(p1, q0 + int(st["blk"]))
        # ---- phase 1: codensity potentials (may OOM; W untouched) ----------
        ni = cp.asarray(n_idx[q0:q1])
        mi = cp.asarray(m_idx[q0:q1])
        rho = phiR[ni]
        rho *= phiR[mi]
        vG = gtools.fft(rho, mesh)
        rho = None
        vG *= coulG
        V = gtools.ifft(vG, mesh)
        vG = None
        Vr = V.real                              # strided view, no copy
        # ---- phase 2: W updates into preallocated scratch --------------------
        for n, s, ma, mb in _segments(q0, q1, n_idx, m_idx, row_start):
            k = mb - ma
            cp.multiply(phiL[n][None, :], Vr[s], out=scratch[:k])
            W[ma:mb] += scratch[:k]
            if symmetric:
                off = 1 if ma == n else 0        # diagonal pair counted once
                if k - off > 0:
                    cp.multiply(phiL[ma + off:mb], Vr[s][off:], out=scratch[:k - off])
                    cp.sum(scratch[:k - off], axis=0, out=rowbuf)
                    W[n] += rowbuf
        V = Vr = None
        q0 = q1
        progress[task] = q0
    progress.pop(task, None)


def _shrink_strip(ctx, task, exc):
    blk = int(ctx.state["blk"])
    if blk <= 1:
        return False
    ctx.state["blk"] = max(1, blk // 2)
    return True


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def pairing_k_force_lowrank(cell, mesh, L, R, aoslices=None, exxdiv=None, blk=None,
                            gchunk=None, group=None, symmetric=True, verbose=True):
    """Pairing-exchange contribution to the pp-RPA gradient for X = L @ R.T.

    Returns numpy (natm, 3) in the convention of ``grad_elec`` (dE/dR; the
    caller adds it to ``de``).  ``exxdiv`` must be None (bare 4pi/G^2, G=0
    dropped), matching the AFT path it replaces.  ``blk`` overrides the pairs
    per FFT strip; ``symmetric=False`` enumerates all r^2 ordered pairs (2x the
    FFTs; debugging aid).  ``group`` is a ``gpu_multi.DeviceGroup`` (default:
    ``LIB_PPRPA_GPUS`` slots).
    """
    assert exxdiv is None, "pairing_k_force_lowrank: only exxdiv=None is supported"
    started = time.perf_counter()
    group = group or default_group()
    L = np.asarray(L, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    assert L.shape == R.shape and L.shape[0] == cell.nao, (L.shape, R.shape)
    keep = np.abs(L).max(axis=0) > 0
    if not keep.all():
        L = np.ascontiguousarray(L[:, keep])
        R = np.ascontiguousarray(R[:, keep])
    nao, r = L.shape
    mesh = np.asarray(mesh, dtype=int)
    ngrid = int(np.prod(mesh))
    coords = cell.gen_uniform_grids(mesh)
    if aoslices is None:
        aoslices = cell.aoslice_by_atom()
    natm = len(aoslices)
    if r == 0:
        return np.zeros((natm, 3))
    n_idx, m_idx, row_start = pair_index(r, symmetric)
    npair = len(n_idx)

    # ---- per-slot residents ----------------------------------------------
    def _setup(ctx):
        st = ctx.state
        _reclaim_gpu()
        st["phiL"], st["phiR"] = lr_on_grid(cell, mesh, coords, L, R, gchunk)
        st["coulG"] = cp.asarray(gtools.get_coulG(cell, k=np.zeros(3), exx=None, mesh=mesh))
        st["W"] = cp.zeros((r, ngrid))
        st["rowbuf"] = cp.empty(ngrid)
        _reclaim_gpu()
        return (plan_strip(ngrid, npair, mesh=mesh) if blk is None
                else max(1, min(npair, int(blk))))

    t0 = time.perf_counter()
    plans = group.each(_setup)
    blk_global = int(min(plans))
    setup_seconds = time.perf_counter() - t0

    def _alloc_scratch(ctx):
        ctx.state["blk"] = blk_global
        ctx.state["scratch"] = cp.empty((blk_global, ngrid))
    group.each(_alloc_scratch)

    tasks = [(p0, min(npair, p0 + blk_global)) for p0 in range(0, npair, blk_global)]
    if verbose:
        _log(f"rank={r} pairs={npair} ({'n<=m' if symmetric else 'all'}) ngrid={ngrid} "
             f"nao={nao} natm={natm} | strip={blk_global} pairs -> {len(tasks)} strips on "
             f"{group.nslots} slot(s); residents/slot={3*r*ngrid*8/1e9:.1f} GB; "
             f"setup {setup_seconds:.0f} s")

    done = [0]
    lock = threading.Lock()
    report_every = max(1, len(tasks) // 20)
    t_strips = time.perf_counter()

    def _work(ctx, task):
        pairing_strip(ctx, task, n_idx, m_idx, row_start, mesh, symmetric)
        if verbose:
            with lock:
                done[0] += 1
                k = done[0]
            if k % report_every == 0 or k == len(tasks):
                el = time.perf_counter() - t_strips
                eta = el / k * (len(tasks) - k)
                _log(f"  strips {k}/{len(tasks)} done ({el:.0f} s elapsed, ETA {eta:.0f} s)")
        return None

    _results, run_stats = group.run(tasks, _work, shrink=_shrink_strip, label="pairing_k")
    strip_seconds = time.perf_counter() - t_strips

    # ---- gradient-AO pass on each partial W, reduce G on the host ----------
    t1 = time.perf_counter()

    def _grad(ctx):
        st = ctx.state
        for key in ("phiL", "phiR", "scratch", "coulG", "rowbuf", "progress"):
            st.pop(key, None)
        _reclaim_gpu()
        G = grad_ao_pass(cell, mesh, coords, st["W"], gchunk)
        st.pop("W", None)
        out = cp.asnumpy(G)
        G = None
        _reclaim_gpu()
        return out

    G = np.sum(group.each(_grad), axis=0)
    grad_seconds = time.perf_counter() - t1
    de = assemble_force(G, L, aoslices)
    total = time.perf_counter() - started
    LAST_TELEMETRY.clear()
    LAST_TELEMETRY.update({
        "rank": int(r), "npair": int(npair), "symmetric": bool(symmetric),
        "ngrid": ngrid, "nao": int(nao), "strip": blk_global, "nstrips": len(tasks),
        "setup_seconds": setup_seconds, "strip_seconds": strip_seconds,
        "grad_seconds": grad_seconds, "total_seconds": total,
        "multi_gpu": run_stats,
    })
    if verbose:
        _log(f"done in {total:.0f} s (setup {setup_seconds:.0f}, strips {strip_seconds:.0f}, "
             f"grad-AO {grad_seconds:.0f}); retries={sum(run_stats['retries_per_slot'])} "
             f"tasks/slot={run_stats['tasks_per_slot']}")
    return de
