#!/usr/bin/env python
"""CPHF response function: dense GPU implementation vs the grouped one.

NV63 setup and cached SCF as bench_gpu_ao2mo.py.  The perturbing density has
the CPHF's structure, dm = C_A x C_I^T + h.c. with random x.  Times ``ncalls``
calls of each and compares the response matrices.

    python bench_gpu_response.py --geom grace_opt.xyz --ke 300 --ncalls 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geom", required=True)
    p.add_argument("--ke", type=float, default=300.0)
    p.add_argument("--as", dest="as_size", type=int, default=300)
    p.add_argument("--ncalls", type=int, default=3)
    p.add_argument("--impl", default="dense,grouped1,grouped2")
    p.add_argument("--basis", default="gth-dzvp")
    p.add_argument("--pseudo", default="gth-pbe")
    p.add_argument("--xc", default="pbe")
    p.add_argument("--charge", type=int, default=-3)
    p.add_argument("--cache", default="bench_mo_cache.npz")
    p.add_argument("--results", default="bench_results.jsonl")
    p.add_argument("--tag", default="")
    return p.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    import cupy as cp
    from pyscf.pbc import dft as cdft
    from lib_pprpa.gpu_multi import DeviceGroup
    from lib_pprpa.grad.pprpa_gamma_gpu import make_gpu_vresp
    from lib_pprpa.grad.gpu_response import GammaResponse
    from bench_gpu_ao2mo import build_cell, scf_orbitals
    cell = build_cell(args)
    mo, moe, e_tot = scf_orbitals(cell, args)
    nocc_all = cell.nelectron // 2
    mo_occ = np.zeros(cell.nao)
    mo_occ[:nocc_all] = 2.0
    mf = cdft.RKS(cell, xc=args.xc)
    mf.exxdiv = None
    mf.mo_coeff, mf.mo_energy, mf.mo_occ, mf.e_tot, mf.converged = mo, moe, mo_occ, e_tot, True
    orbI = mo[:, :nocc_all]
    orbA = mo[:, nocc_all:]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((orbA.shape[1], nocc_all)) * 1e-2
    dm = orbA @ x @ orbI.T
    dm = dm + dm.T
    ndev = cp.cuda.runtime.getDeviceCount()
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"[resp] {props['name'].decode()} x{ndev} nao={cell.nao} mesh={list(map(int, cell.mesh))} "
          f"ngrid={int(np.prod(cell.mesh))} xc={args.xc}", flush=True)

    results = {}
    for name in [s.strip() for s in args.impl.split(",") if s.strip()]:
        cp.get_default_memory_pool().free_all_blocks()
        t0 = time.perf_counter()
        if name == "dense":
            fn = make_gpu_vresp(cell, mf)
            resp = None
        else:
            slots = [0] if name.endswith("1") else ([0, 1] if ndev >= 2 else [0, 0])
            resp = GammaResponse(cell, mf, group=DeviceGroup(slots))
            fn = resp
        t_setup = time.perf_counter() - t0
        times = []
        v = None
        for _ in range(args.ncalls):
            cp.cuda.Device().synchronize()
            t1 = time.perf_counter()
            v = fn(dm)
            cp.cuda.Device().synchronize()
            times.append(time.perf_counter() - t1)
        summary = resp.summary() if resp is not None else ""
        print(f"[resp] {name}: setup {t_setup:.1f} s, call {min(times):.2f} s best of "
              f"{[round(t, 2) for t in times]}  {summary}", flush=True)
        results[name] = (min(times), v)
        with open(args.results, "a") as fh:
            fh.write(json.dumps({"tag": args.tag, "kind": "response", "impl": name,
                                 "setup_seconds": t_setup, "call_seconds": times}) + "\n")
        if resp is not None:
            resp.release()
    ref = "dense" if "dense" in results else next(iter(results))
    for name, (t, v) in results.items():
        if name == ref:
            continue
        d = float(np.abs(v - results[ref][1]).max())
        sc = float(np.abs(results[ref][1]).max())
        print(f"[resp] {name} vs {ref}: max|diff| = {d:.3e}  max|ref| = {sc:.3e}  rel = {d/sc:.3e}  "
              f"speedup {results[ref][0] / t:.2f}x", flush=True)
        with open(args.results, "a") as fh:
            fh.write(json.dumps({"tag": args.tag, "kind": "response_compare", "impl": name,
                                 "ref": ref, "rel": d / sc, "speedup": results[ref][0] / t}) + "\n")


if __name__ == "__main__":
    main()
