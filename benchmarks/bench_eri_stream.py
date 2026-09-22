#!/usr/bin/env python
"""Davidson ERI matrix-vector product: resident vs split vs tiled (pageable / pinned).

Uses the saved reference tensors (``--ref``: vvvv/oovv/oooo.npy written by
bench_gpu_ao2mo.py --save) instead of recomputing an ao2mo, and times the
contraction on random trial vectors in every mode the driver can pick.  The
products of all modes are compared with the resident one.

    python bench_eri_stream.py --ref /nfs/roberts/scratch/.../nv63_ke300_as300_stage2 --ntri 32
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="directory with vvvv.npy oovv.npy oooo.npy")
    p.add_argument("--ntri", type=int, default=32, help="trial vectors per MVP")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--modes", default="resident,split,tiled_pageable,tiled_pinned")
    p.add_argument("--mult", default="t")
    p.add_argument("--results", default="bench_results.jsonl")
    p.add_argument("--tag", default="")
    return p.parse_args(argv)


def _load(ref, name, pinned):
    import cupyx
    src = np.load(os.path.join(ref, f"{name}.npy"), mmap_mode="r")
    t0 = time.perf_counter()
    if pinned:
        out = cupyx.empty_pinned(src.shape, dtype=np.float64)
    else:
        out = np.empty(src.shape, dtype=np.float64)
    step = max(1, src.shape[0] // 16)
    for i in range(0, src.shape[0], step):
        out[i:i + step] = src[i:i + step]
    print(f"[eri] loaded {name} {out.nbytes/1e9:.1f} GB ({'pinned' if pinned else 'pageable'}) "
          f"in {time.perf_counter() - t0:.0f} s", flush=True)
    return out


def main(argv=None):
    args = parse(argv)
    import cupy as cp
    from lib_pprpa.gpu_multi import DeviceGroup
    from lib_pprpa.pprpa_davidson import ppRPA_Davidson
    from lib_pprpa.pprpa_eri_gpu import attach_gpu_eri_contraction, get_last_telemetry, release_gpu_eri

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    need_pinned = any(m in ("tiled_pinned", "split", "resident") for m in modes)
    need_pageable = "tiled_pageable" in modes
    host = {}
    if need_pinned:
        host["pinned"] = {n: _load(args.ref, n, True) for n in ("vvvv", "oovv", "oooo")}
    if need_pageable:
        host["pageable"] = {n: _load(args.ref, n, False) for n in ("vvvv", "oovv", "oooo")}
    any_set = next(iter(host.values()))
    nvir = any_set["vvvv"].shape[0]
    nocc = any_set["oooo"].shape[0]
    rng = np.random.default_rng(0)
    moe = np.sort(rng.standard_normal(nocc + nvir))
    mp = ppRPA_Davidson(nocc, moe, Lpq=None, channel="hh", nroot=5, residue_thresh=1e-9,
                        trial="identity")
    mp.mu = 0.0
    mp.multi = args.mult
    mp.check_parameter()
    tv = rng.standard_normal((args.ntri, mp.full_dim))
    ndev = cp.cuda.runtime.getDeviceCount()
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"[eri] {props['name'].decode()} x{ndev}  nocc={nocc} nvir={nvir} full_dim={mp.full_dim} "
          f"ntri={args.ntri}  eri={(any_set['vvvv'].nbytes + any_set['oovv'].nbytes + any_set['oooo'].nbytes)/1e9:.1f} GB",
          flush=True)

    results = {}
    for mode in modes:
        cp.get_default_memory_pool().free_all_blocks()
        if mode == "resident":
            src, kw = host["pinned"], {"mode": "resident", "group": DeviceGroup([0])}
        elif mode == "split":
            devs = [0, 1] if ndev >= 2 else [0, 0]
            src, kw = host["pinned"], {"mode": "split", "group": DeviceGroup(devs)}
        elif mode == "tiled_pinned":
            src, kw = host["pinned"], {"mode": "tiled", "group": DeviceGroup([0])}
        elif mode == "tiled_pageable":
            src, kw = host["pageable"], {"mode": "tiled", "group": DeviceGroup([0])}
        else:
            raise ValueError(mode)
        print(f"\n[eri] ===== {mode} =====", flush=True)
        t0 = time.perf_counter()
        attach_gpu_eri_contraction(mp, src["vvvv"], src["oovv"], src["oooo"], **kw)
        t_attach = time.perf_counter() - t0
        tel = get_last_telemetry()
        mv = None
        times = []
        for r in range(args.repeat):
            cp.cuda.Device().synchronize()
            t1 = time.perf_counter()
            mv = mp.contraction(tv)
            cp.cuda.Device().synchronize()
            times.append(time.perf_counter() - t1)
        release_gpu_eri(mp)
        stream = get_last_telemetry().get("stream", {})
        rec = {"tag": args.tag, "kind": "eri_mvp", "mode": mode, "attach_seconds": t_attach,
               "mvp_seconds": times, "mvp_best": min(times), "ntri": args.ntri,
               "telemetry_mode": tel.get("mode"), "pinned_host": tel.get("pinned_host"),
               "tile": tel.get("tile"), "gpu_slots": tel.get("gpu_slots"),
               "gb_per_s": stream.get("gb_per_s"), "streamed_gb_per_mvp": (stream.get("bytes", 0) / max(stream.get("mvps", 1), 1)) / 1e9}
        print(f"[eri] {mode}: attach {t_attach:.1f} s, MVP {min(times):.2f} s best of {times} "
              f"(mode={tel.get('mode')} tile={tel.get('tile')} pinned={tel.get('pinned_host')} "
              f"{rec['streamed_gb_per_mvp']:.1f} GB/MVP at {stream.get('gb_per_s', 0) or 0:.1f} GB/s)",
              flush=True)
        results[mode] = (rec, mv)
        with open(args.results, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    ref_mode = "resident" if "resident" in results else modes[0]
    mv_ref = results[ref_mode][1]
    for mode, (rec, mv) in results.items():
        if mode == ref_mode:
            continue
        d = float(np.abs(mv - mv_ref).max())
        sc = float(np.abs(mv_ref).max())
        print(f"[eri] {mode} vs {ref_mode}: max|diff| = {d:.3e}  max|ref| = {sc:.3e}  rel = {d/sc:.3e}  "
              f"MVP speedup {results[ref_mode][0]['mvp_best'] / rec['mvp_best']:.2f}x", flush=True)
        with open(args.results, "a") as fh:
            fh.write(json.dumps({"tag": args.tag, "kind": "eri_mvp_compare", "mode": mode, "ref": ref_mode,
                                 "rel": d / sc}) + "\n")


if __name__ == "__main__":
    main()
