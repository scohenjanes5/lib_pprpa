#!/usr/bin/env python
"""Benchmark and consistency check for the low-rank exchange and the pairing force.

Same NV63 setup and cached SCF as ``bench_gpu_ao2mo.py``.  The amplitude
factors are synthetic but production-shaped: X = C_a x C_a^T with a random
antisymmetric x (triplet) on the active virtuals, Y = C_i y C_i^T on the
active occupieds; the pairing force uses L = [C_a x, C_i y], R = [C_a, C_i].

    python bench_gpu_fft_kernels.py --geom grace_opt.xyz --ke 300 --as 300 --what k,pairing

Reference = the frozen kernels in ``gpu_fft_k_base.py`` /
``gpu_pairing_force_base.py`` (commit 3f56e5f); every result is compared
element-wise against them.  For the exchange the quantity compared is what the
gradient consumes, ``mo_coeff.T @ K @ orbp`` (nmo x nact).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geom", required=True)
    p.add_argument("--ke", type=float, default=300.0)
    p.add_argument("--as", dest="as_size", type=int, default=300)
    p.add_argument("--what", default="k,pairing", help="comma list from {k,pairing}")
    p.add_argument("--k-impl", default="base,new_ao,new_ket",
                   help="exchange variants: base (frozen, full K), new_ao (full K), new_ket (K @ orbp)")
    p.add_argument("--pairing-impl", default="base,new")
    p.add_argument("--basis", default="gth-dzvp")
    p.add_argument("--pseudo", default="gth-pbe")
    p.add_argument("--xc", default="pbe")
    p.add_argument("--charge", type=int, default=-3)
    p.add_argument("--cache", default="bench_mo_cache.npz")
    p.add_argument("--results", default="bench_results.jsonl")
    p.add_argument("--tag", default="")
    return p.parse_args(argv)


def load_module(path):
    modname = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    args = parse(argv)
    import cupy as cp
    import lib_pprpa
    from lib_pprpa.gpu_multi import DeviceGroup
    from bench_gpu_ao2mo import build_cell, scf_orbitals   # same cell / cache
    print(f"[bench] lib_pprpa from {os.path.dirname(lib_pprpa.__file__)}", flush=True)
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    cell = build_cell(args)
    ng = int(np.prod(cell.mesh))
    mo, moe, _e = scf_orbitals(cell, args)
    nocc_all = cell.nelectron // 2
    nocc = min(args.as_size, nocc_all)
    nvir = min(args.as_size, cell.nao - nocc_all)
    nfo = nocc_all - nocc
    orbi = mo[:, nfo:nfo + nocc]
    orba = mo[:, nfo + nocc:nfo + nocc + nvir]
    orbp = mo[:, nfo:nfo + nocc + nvir]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((nvir, nvir)); x = (x - x.T) / np.sqrt(nvir)     # triplet-like
    y = rng.standard_normal((nocc, nocc)); y = (y - y.T) / np.sqrt(nocc)
    print(f"[bench] GPU {props['name'].decode()} natm={cell.natm} nao={cell.nao} "
          f"mesh={list(map(int, cell.mesh))} ngrid={ng} nocc={nocc} nvir={nvir} nact={nocc + nvir}",
          flush=True)
    what = [w.strip() for w in args.what.split(",") if w.strip()]

    def record(rec):
        with open(args.results, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    if "k" in what:
        factors = ((orba @ x, orba), (orbi @ y, orbi))
        results = {}
        for name in [s.strip() for s in args.k_impl.split(",") if s.strip()]:
            cp.get_default_memory_pool().free_all_blocks()
            group = DeviceGroup([0])
            if name == "base":
                mod = load_module(os.path.join(HERE, "gpu_fft_k_base.py"))
                kw = {}
            else:
                import lib_pprpa.gpu_fft_k as mod
                kw = {"ket": orbp} if name == "new_ket" else {}
            print(f"\n[bench] ===== exchange {name} =====", flush=True)
            t0 = time.perf_counter()
            K = mod.get_k_lowrank(cell, cell.mesh, factors, hermi=2, exxdiv=None, group=group,
                                  verbose=True, **kw)
            dt = time.perf_counter() - t0
            if K.shape[-1] == cell.nao:
                proj = np.stack([mo.T @ Ki @ orbp for Ki in K])
            else:
                proj = np.stack([mo.T @ Ki for Ki in K])
            tel = mod.get_last_telemetry()
            sets = tel.get("sets", [])
            desc = "  ".join(f"set{i}: row_blk={s.get('row_blk')} "
                             f"{'fft_blk=' + str(s.get('fft_blk')) if 'fft_blk' in s else 'rank_blk=' + str(s.get('rank_blk'))}"
                             for i, s in enumerate(sets))
            print(f"[bench] exchange {name}: {dt:8.1f} s  {desc}", flush=True)
            results[name] = (dt, proj)
            record({"tag": args.tag, "kind": "exchange", "impl": name, "seconds": dt, "as": args.as_size,
                    "ngrid": ng, "sets": [{k: v for k, v in s.items() if k != "multi_gpu"} for s in sets]})
            del K
            cp.get_default_memory_pool().free_all_blocks()
        names = list(results)
        ref = names[0]
        for name in names[1:]:
            d = np.abs(results[name][1] - results[ref][1]).max(axis=(1, 2))
            sc = np.abs(results[ref][1]).max(axis=(1, 2))
            print(f"[bench] exchange {name} vs {ref}: max|diff| = {d}  max|ref| = {sc}  "
                  f"rel = {d / sc}  speedup {results[ref][0] / results[name][0]:.2f}x", flush=True)
            record({"tag": args.tag, "kind": "exchange_compare", "impl": name, "ref": ref,
                    "rel": (d / sc).tolist(), "speedup": results[ref][0] / results[name][0]})

    if "pairing" in what:
        L = np.hstack([orba @ x, orbi @ y])
        R = np.hstack([orba, orbi])
        results = {}
        for name in [s.strip() for s in args.pairing_impl.split(",") if s.strip()]:
            cp.get_default_memory_pool().free_all_blocks()
            group = DeviceGroup([0])
            if name == "base":
                mod = load_module(os.path.join(HERE, "gpu_pairing_force_base.py"))
            else:
                import lib_pprpa.gpu_pairing_force as mod
            print(f"\n[bench] ===== pairing force {name} =====", flush=True)
            t0 = time.perf_counter()
            de = mod.pairing_k_force_lowrank(cell, cell.mesh, L, R, exxdiv=None, group=group, verbose=True)
            dt = time.perf_counter() - t0
            tel = mod.get_last_telemetry()
            print(f"[bench] pairing {name}: {dt:8.1f} s  strip={tel.get('strip')} nstrips={tel.get('nstrips')} "
                  f"strips {tel.get('strip_seconds', 0):.1f} s  grad-AO {tel.get('grad_seconds', 0):.1f} s",
                  flush=True)
            results[name] = (dt, de)
            record({"tag": args.tag, "kind": "pairing", "impl": name, "seconds": dt, "as": args.as_size,
                    "ngrid": ng, "tel": {k: v for k, v in tel.items() if k != "multi_gpu"}})
        names = list(results)
        ref = names[0]
        for name in names[1:]:
            d = float(np.abs(results[name][1] - results[ref][1]).max())
            sc = float(np.abs(results[ref][1]).max())
            print(f"[bench] pairing {name} vs {ref}: max|diff| = {d:.3e}  max|ref| = {sc:.3e}  "
                  f"rel = {d / sc:.3e}  speedup {results[ref][0] / results[name][0]:.2f}x", flush=True)
            record({"tag": args.tag, "kind": "pairing_compare", "impl": name, "ref": ref,
                    "rel": d / sc, "speedup": results[ref][0] / results[name][0]})


if __name__ == "__main__":
    import sys
    sys.path.insert(0, HERE)
    main()
