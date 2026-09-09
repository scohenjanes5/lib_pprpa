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
* After assembly, the three finals are uploaded only if they still fit in
  75% of VRAM (``gpu_mem.fits_resident``). Otherwise they stay on the host
  for tiled Davidson contraction.

Validated element-wise vs CPU pprpaobj(mo_eri=True) (run this file as __main__).
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import cupy as cp
from gpu4pyscf.pbc import tools as gtools
from gpu4pyscf.pbc.dft import numint as gnumint

from lib_pprpa.gpu_mem import eri_bytes, fits_resident


LAST_TELEMETRY = {}


def get_last_telemetry():
    """Return metrics from the most recent gpu_ao2mo_blocks call."""
    return dict(LAST_TELEMETRY)


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
    """Driver-reported free VRAM (CuPy pool holdings count as used).

    Do NOT use gpu4pyscf get_avail_mem here: it can report a stale/constant
    value that ignores live CuPy allocations.  Call ``_reclaim_gpu`` first if
    you need free after FFT probes.
    """
    free, _total = cp.cuda.runtime.memGetInfo()
    return int(free)


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


def _is_oom(exc):
    if isinstance(exc, cp.cuda.memory.OutOfMemoryError):
        return True
    msg = f"{type(exc).__name__}: {exc}".lower()
    return ("outofmemory" in msg
            or "memory allocation" in msg
            or "cudaerrormemoryallocation" in msg)


def _cushion_bytes():
    """VRAM to hold back during probes so assembly is not razor-tight.

    At least 512 MiB, or 2% of currently free memory (whichever is larger).
    """
    return max(512 * 1024 ** 2, int(0.02 * _free_bytes()))



def _estimate_pair_blk(npair, ngrid, nB, pair_blk=None):
    """Estimate a fast strip size from live driver memory.

    The final ERI tensor has already been allocated when this is called.  The
    estimate budgets the remaining memory for rho, FFT output/workspace, vR,
    rho_q and the GEMM tile.  Default strips are aligned to complete second-MO
    rows, which permits direct contiguous writes in physicist layout.
    """
    if pair_blk is not None:
        raw = max(1, min(int(pair_blk), npair))
    else:
        free = _free_bytes()
        reserve = max(_cushion_bytes(), int(0.06 * free))
        budget = max(0, free - reserve)
        # FFT output/workspace is partly reused; geometric OOM rollback covers
        # residual driver variation without expensive cold cuFFT probes.
        linear = max(ngrid * 40, 1)
        # Solve 8*b^2 + linear*b <= budget (tile plus grid temporaries).
        raw = int((-linear + math.sqrt(linear * linear + 32 * budget)) / 16)
        raw = max(1, min(raw, npair))
        env_cap = os.environ.get("GPU_AO2MO_MAX_PAIR_BLK")
        if env_cap:
            raw = min(raw, max(1, int(env_cap)))
    raw = max(nB, raw)
    raw = max(nB, (raw // nB) * nB)
    return min(raw, npair)

def _assemble_direct(moA, moB, wcoulG, mesh, out, blk, stats, on_gpu):
    """Fill a physicist-layout ERI directly in baseline arithmetic order."""
    nA, nB = moA.shape[0], moB.shape[0]
    npair = nA * nB
    nblk = (npair + blk - 1) // blk
    ngemms = nblk * nblk
    print(
        f"[gpu_ao2mo]   direct strips={nblk}x{nblk} GEMMs={ngemms}; "
        f"pair_blk={blk}", flush=True)
    stats["nstrips"] = nblk
    stats["gemms"] = ngemms

    for ip, p0 in enumerate(range(0, npair, blk)):
        p1 = min(p0 + blk, npair)
        a0, a1 = p0 // nB, p1 // nB
        rho_p = _codensity_pairs(moA, moB, p0, p1)
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
        stats["min_free_bytes"] = min(stats["min_free_bytes"], _free_bytes())
        if nblk > 1 and ((ip + 1) % max(1, nblk // 4) == 0 or ip + 1 == nblk):
            print(f"[gpu_ao2mo]   strip {ip + 1}/{nblk} done", flush=True)
    return out


def _block_direct(name, moA, moB, wcoulG, mesh, pair_blk=None, force_host=False):
    """Fill a final physicist ERI on GPU or directly on host when VRAM is tight.

    ``force_host`` stages a final tensor until all MO grids are released.  This
    is required when the three final ERIs fit, but they do not fit together with
    the MO grids and one FFT strip on smaller high-memory GPUs.
    """
    nA, nB = moA.shape[0], moB.shape[0]
    npair = nA * nB
    out_bytes = npair * npair * 8
    free_before = _free_bytes()
    on_gpu = not force_host
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
        blk = _estimate_pair_blk(npair, moA.shape[1], nB, pair_blk=pair_blk)
        # A retained result must leave enough space for useful FFT strips.
        if pair_blk is None and blk <= min(npair, 4 * nB):
            del out
            _reclaim_gpu()
            on_gpu = False
    if not on_gpu:
        out = np.empty((nA, nA, nB, nB), dtype=np.float64)
        blk = _estimate_pair_blk(npair, moA.shape[1], nB, pair_blk=pair_blk)
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
        "free_after_output_bytes": int(_free_bytes()),
        "min_free_bytes": int(_free_bytes()),
    }
    location = "GPU" if on_gpu else ("host-staged" if force_host else "host-direct")
    print(
        f"[gpu_ao2mo] {name} final tensor {out_bytes/1e9:.2f} GB "
        f"on {location}; pair_blk={blk} (analytic); "
        f"free≈{free_before/1e9:.2f}→{_free_bytes()/1e9:.2f} GB", flush=True)

    while True:
        started = time.perf_counter()
        try:
            _assemble_direct(moA, moB, wcoulG, mesh, out, blk, stats, on_gpu)
            if not on_gpu and not force_host:
                uploaded = cp.asarray(out)
                del out
                out = uploaded
            _sync()
            stats["seconds"] = time.perf_counter() - started
            stats["pair_blk"] = int(blk)
            return out, stats
        except Exception as exc:
            if not _is_oom(exc):
                del out
                raise
            stats["retries"] += 1
            _reclaim_gpu()
            if blk <= nB:
                del out
                raise MemoryError(
                    f"gpu_ao2mo: OOM at one complete MO-row strip for {name} "
                    f"(free≈{_free_bytes()/1e9:.2f} GB)") from exc
            blk = max(nB, ((blk // 2) // nB) * nB)
            print(
                f"[gpu_ao2mo] OOM — retrying {name} from start with "
                f"pair_blk={blk}; free≈{_free_bytes()/1e9:.2f} GB", flush=True)


def gpu_ao2mo_blocks(cell, cocc, cvir, mesh, pair_blk=None, return_gpu=False):
    """Return direct-assembled vvvv, oovv, oooo tensors in physicist layout."""
    global LAST_TELEMETRY
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
    grid_seconds = time.perf_counter() - grid_started

    final_bytes = eri_bytes(no, nv)
    vvvv_b = nv ** 4 * 8
    oooo_b = no ** 4 * 8
    oovv_b = (no ** 2) * (nv ** 2) * 8
    print(
        f"[gpu_ao2mo] no={no} nv={nv} ngrid={ng} | "
        f"ERI sizes vvvv/oovv/oooo = "
        f"{vvvv_b/1e9:.2f}/{oovv_b/1e9:.2f}/{oooo_b/1e9:.2f} GB | "
        f"free≈{_free_bytes()/1e9:.2f} GB", flush=True)

    grid_bytes = moO.nbytes + moV.nbytes + coulG.nbytes + wcoulG.nbytes
    stage_host = not fits_resident(
        no, nv, extra_bytes=grid_bytes, total_bytes=total_bytes)
    if stage_host:
        print("[gpu_ao2mo] staging all final ERIs on host until MO grids are released",
              flush=True)
    vvvv, stat_v = _block_direct(
        "vvvv", moV, moV, wcoulG, mesh, pair_blk=pair_blk, force_host=stage_host)
    oooo, stat_o = _block_direct(
        "oooo", moO, moO, wcoulG, mesh, pair_blk=pair_blk, force_host=stage_host)
    oovv, stat_ov = _block_direct(
        "oovv", moO, moV, wcoulG, mesh, pair_blk=pair_blk, force_host=stage_host)

    del moO, moV, coulG, wcoulG
    _reclaim_gpu()
    uploaded = False
    if stage_host:
        can_resident = fits_resident(no, nv, extra_bytes=0, total_bytes=total_bytes)
        if can_resident:
            print("[gpu_ao2mo] uploading host ERIs to GPU for resident Davidson",
                  flush=True)
            vvvv_g = cp.asarray(vvvv); del vvvv; vvvv = vvvv_g
            oovv_g = cp.asarray(oovv); del oovv; oovv = oovv_g
            oooo_g = cp.asarray(oooo); del oooo; oooo = oooo_g
            uploaded = True
        else:
            print(
                f"[gpu_ao2mo] leaving ERIs on host for tiled Davidson "
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
