#!/usr/bin/env python
"""Benchmark and consistency check for the GPU FFT ao2mo kernel.

Runs the frozen baseline kernel (``benchmarks/gpu_ao2mo_base.py``, the
all-tiles full-pair-index version from commit 7475b2f) and the current
``lib_pprpa.gpu_ao2mo`` on the same SCF orbitals, reports per-block wall
time / GEMM rate, and compares the two ERI tensors element-wise.

    python bench_gpu_ao2mo.py --geom grace_opt.xyz --ke 300 --as 300 --pair-blk 600 --impl base,new

The SCF (GPU KRKS, same settings as the production drivers) is cached in
``--cache`` so repeated runs on one geometry pay it once.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geom", required=True, help="ASE-readable geometry (lattice taken from it)")
    p.add_argument("--ke", type=float, default=300.0)
    p.add_argument("--as", dest="as_size", type=int, default=300)
    p.add_argument("--pair-blk", type=int, default=None, help="force the strip size (default: planner)")
    p.add_argument("--impl", default="base,new", help="comma list from {base,new}; both -> compared")
    p.add_argument("--force-host", action="store_true",
                   help="stage the final tensors on the host (production NV216 path)")
    p.add_argument("--basis", default="gth-dzvp")
    p.add_argument("--pseudo", default="gth-pbe")
    p.add_argument("--xc", default="pbe")
    p.add_argument("--charge", type=int, default=-3)
    p.add_argument("--cache", default="bench_mo_cache.npz")
    p.add_argument("--results", default="bench_results.jsonl")
    p.add_argument("--tag", default="")
    p.add_argument("--save", default=None, metavar="DIR",
                   help="write the (single) impl's tensors as .npy into DIR")
    p.add_argument("--compare", default=None, metavar="DIR",
                   help="compare the (single) impl's tensors with the .npy files in DIR")
    return p.parse_args(argv)


def build_cell(args):
    from ase.io import read
    from pyscf.data.nist import BOHR
    from pyscf.pbc import gto
    at = read(args.geom)
    cell = gto.Cell()
    cell.atom = [(s, p) for s, p in zip(at.get_chemical_symbols(), at.get_positions() / BOHR)]
    cell.a = np.asarray(at.cell) / BOHR
    cell.unit = "Bohr"
    cell.basis = args.basis
    cell.pseudo = args.pseudo
    cell.charge = args.charge
    cell.spin = 0
    cell.ke_cutoff = args.ke
    cell.verbose = 0
    cell.build()
    return cell


def scf_orbitals(cell, args):
    import cupy as cp
    key = f"{os.path.abspath(args.geom)}|{args.ke}|{args.basis}|{args.pseudo}|{args.xc}|{args.charge}"
    if os.path.isfile(args.cache):
        d = np.load(args.cache, allow_pickle=True)
        if str(d["key"]) == key:
            print(f"[bench] SCF orbitals from cache {args.cache}", flush=True)
            return d["mo_coeff"], d["mo_energy"], float(d["e_tot"])
        print(f"[bench] cache key mismatch, redoing SCF", flush=True)
    from gpu4pyscf.pbc import dft as gdft
    t0 = time.perf_counter()
    kg = gdft.KRKS(cell, kpts=np.zeros((1, 3)), xc=args.xc)
    kg.exxdiv = None
    kg.conv_tol = 1e-9
    kg.kernel()
    mo = cp.asnumpy(kg.mo_coeff[0])
    moe = cp.asnumpy(kg.mo_energy[0])
    e_tot = float(kg.e_tot)
    print(f"[bench] SCF done in {time.perf_counter() - t0:.0f} s, E={e_tot:.8f}, "
          f"converged={kg.converged}", flush=True)
    kg = None
    cp.get_default_memory_pool().free_all_blocks()
    np.savez(args.cache, key=key, mo_coeff=mo, mo_energy=moe, e_tot=e_tot)
    return mo, moe, e_tot


def load_impl(name):
    """'new' = lib_pprpa.gpu_ao2mo; 'base' = benchmarks/gpu_ao2mo_base.py;
    anything else = a snapshot module by path (or name under benchmarks/)."""
    if name == "new":
        import lib_pprpa.gpu_ao2mo as mod
        return mod
    path = os.path.join(HERE, "gpu_ao2mo_base.py") if name == "base" else name
    if not os.path.isfile(path):
        path = os.path.join(HERE, name if name.endswith(".py") else f"gpu_ao2mo_{name}.py")
    modname = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def max_abs_diff(a, b, chunk=8):
    """max|a-b| and max|a| over the first axis in chunks (tensors may be ~65 GB)."""
    worst = 0.0
    scale = 0.0
    for i in range(0, a.shape[0], chunk):
        da = np.asarray(a[i:i + chunk])
        db = np.asarray(b[i:i + chunk])
        worst = max(worst, float(np.abs(da - db).max()))
        scale = max(scale, float(np.abs(da).max()))
    return worst, scale


def main(argv=None):
    args = parse(argv)
    import cupy as cp
    import lib_pprpa
    from lib_pprpa.gpu_multi import DeviceGroup
    print(f"[bench] lib_pprpa from {os.path.dirname(lib_pprpa.__file__)}", flush=True)
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    print(f"[bench] GPU {props['name'].decode()} | cupy {cp.__version__} | "
          f"cuBLAS {cp.cuda.cublas.getVersion(cp.cuda.device.get_cublas_handle())} | "
          f"CUBLAS env: " + " ".join(f"{k}={v}" for k, v in os.environ.items() if k.startswith("CUBLAS")),
          flush=True)

    cell = build_cell(args)
    ng = int(np.prod(cell.mesh))
    print(f"[bench] natm={cell.natm} nao={cell.nao} mesh={list(map(int, cell.mesh))} ngrid={ng} "
          f"ke={args.ke}", flush=True)
    mo, moe, e_tot = scf_orbitals(cell, args)
    nocc_all = cell.nelectron // 2
    nocc = min(args.as_size, nocc_all)
    nvir = min(args.as_size, cell.nao - nocc_all)
    nfo = nocc_all - nocc
    cocc = mo[:, nfo:nfo + nocc]
    cvir = mo[:, nfo + nocc:nfo + nocc + nvir]
    print(f"[bench] active space nocc={nocc} nvir={nvir} (frozen {nfo})", flush=True)

    if args.force_host:
        os.environ["GPU_AO2MO_FORCE_HOST"] = "1"
    else:
        os.environ.pop("GPU_AO2MO_FORCE_HOST", None)

    impls = [s.strip() for s in args.impl.split(",") if s.strip()]
    results = {}
    for name in impls:
        mod = load_impl(name)
        cp.get_default_memory_pool().free_all_blocks()
        group = DeviceGroup([0])
        print(f"\n[bench] ===== {name}: pair_blk={args.pair_blk} force_host={args.force_host} =====",
              flush=True)
        t0 = time.perf_counter()
        vvvv, oovv, oooo = mod.gpu_ao2mo_blocks(cell, cocc, cvir, cell.mesh,
                                                pair_blk=args.pair_blk, return_gpu=False,
                                                group=group)
        wall = time.perf_counter() - t0
        tel = mod.get_last_telemetry()
        blocks = {b["name"]: b for b in tel["blocks"]}
        rec = {"tag": args.tag, "impl": name, "as": args.as_size, "nocc": nocc, "nvir": nvir,
               "ngrid": ng, "pair_blk_req": args.pair_blk, "force_host": bool(args.force_host),
               "wall": wall, "mo_grid_seconds": tel["mo_grid_seconds"],
               "gpu": props["name"].decode(), "blocks": {}}
        for bn in ("vvvv", "oovv", "oooo"):
            b = blocks[bn]
            rec["blocks"][bn] = {k: b.get(k) for k in
                                 ("seconds", "pair_blk", "final_pair_blk_per_slot", "output_location",
                                  "compact", "gemms", "gemm_flop", "tflops", "scatter_seconds",
                                  "retries")}
            print(f"[bench] {name} {bn}: {b['seconds']:8.1f} s  pair_blk={b['pair_blk']} "
                  f"loc={b['output_location']}"
                  + (f"  {b['gemm_flop']/1e15:.3f} PFLOP  {b['tflops']:.1f} TFLOP/s  "
                     f"scatter {b['scatter_seconds']:.1f} s" if "tflops" in b else ""),
                  flush=True)
        print(f"[bench] {name} total ao2mo wall {wall:.1f} s "
              f"(mo grids {tel['mo_grid_seconds']:.1f} s)", flush=True)
        results[name] = (rec, {"vvvv": vvvv, "oovv": oovv, "oooo": oooo})
        with open(args.results, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if args.save:
            os.makedirs(args.save, exist_ok=True)
            for bn, arr in results[name][1].items():
                np.save(os.path.join(args.save, f"{bn}.npy"), np.asarray(arr))
            print(f"[bench] saved {name} tensors to {args.save}", flush=True)
        if args.compare:
            print(f"\n[bench] ===== consistency {name} vs {args.compare} =====", flush=True)
            cmp = {}
            for bn, arr in results[name][1].items():
                ref = np.load(os.path.join(args.compare, f"{bn}.npy"), mmap_mode="r")
                worst, scale = max_abs_diff(ref, np.asarray(arr))
                cmp[bn] = {"max_abs_diff": worst, "max_abs": scale, "rel": worst / max(scale, 1e-300)}
                print(f"[bench] {bn}: max|ref-{name}| = {worst:.3e}   max|ref| = {scale:.3e}   "
                      f"rel = {worst / max(scale, 1e-300):.3e}", flush=True)
            with open(args.results, "a") as fh:
                fh.write(json.dumps({"tag": args.tag, "compare_dir": args.compare, "impl": name,
                                     "as": args.as_size, "blocks": cmp}) + "\n")
        del vvvv, oovv, oooo
        cp.get_default_memory_pool().free_all_blocks()

    if len(impls) == 2:
        a, b = impls
        print(f"\n[bench] ===== consistency {a} vs {b} =====", flush=True)
        cmp = {}
        for bn in ("vvvv", "oovv", "oooo"):
            worst, scale = max_abs_diff(results[a][1][bn], results[b][1][bn])
            cmp[bn] = {"max_abs_diff": worst, "max_abs": scale, "rel": worst / max(scale, 1e-300)}
            print(f"[bench] {bn}: max|{a}-{b}| = {worst:.3e}   max|{a}| = {scale:.3e}   "
                  f"rel = {worst / max(scale, 1e-300):.3e}", flush=True)
        wa = results[a][0]["wall"]
        wb = results[b][0]["wall"]
        print(f"[bench] speedup {a}/{b} = {wa / wb:.2f}x  ({wa:.0f} s -> {wb:.0f} s)", flush=True)
        with open(args.results, "a") as fh:
            fh.write(json.dumps({"tag": args.tag, "compare": [a, b], "as": args.as_size,
                                 "pair_blk_req": args.pair_blk, "force_host": bool(args.force_host),
                                 "speedup": wa / wb, "blocks": cmp}) + "\n")


if __name__ == "__main__":
    main()
