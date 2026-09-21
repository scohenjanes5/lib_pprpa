#!/usr/bin/env python
"""Micro-benchmark of the ao2mo tile GEMM  C[M,N] = A[M,K] @ B[N,K]^T  in cupy fp64.

Sweeps the tile width (M = N = b) at the production K (grid points) to show
how the sustained rate depends on the strip width, optionally chunking K.
Also reports the error of each shape against a float64 reference computed by
numpy on a small sub-block (so fp64-emulation modes can be checked).

    python bench_gemm_shapes.py --K 1225043 --widths 300,600,1200,2400 --kchunk 0
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=107 ** 3)
    p.add_argument("--widths", default="300,600,1200,2400")
    p.add_argument("--kchunk", type=int, default=0, help="0 = single GEMM, else accumulate over K chunks")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--check", type=int, default=64, help="rows/cols of C checked against numpy")
    return p.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    import cupy as cp
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    print(f"[gemm] {props['name'].decode()} cupy {cp.__version__} "
          f"cuBLAS {cp.cuda.cublas.getVersion(cp.cuda.device.get_cublas_handle())} "
          f"env: " + " ".join(f"{k}={v}" for k, v in os.environ.items() if k.startswith("CUBLAS")),
          flush=True)
    K = args.K
    rng = cp.random.default_rng(0)
    for b in [int(x) for x in args.widths.split(",")]:
        need = 2 * b * K * 8 + b * b * 8
        free = cp.cuda.runtime.memGetInfo()[0]
        if need > 0.85 * free:
            print(f"[gemm] b={b}: needs {need/1e9:.0f} GB, free {free/1e9:.0f} GB -> skipped", flush=True)
            continue
        A = rng.standard_normal((b, K), dtype=cp.float64)
        B = rng.standard_normal((b, K), dtype=cp.float64)
        # reference on a corner, in numpy fp64 (chunked so the host copy stays small)
        nc = min(args.check, b)
        ref = np.zeros((nc, nc))
        for g0 in range(0, K, 1 << 20):
            g1 = min(K, g0 + (1 << 20))
            ref += cp.asnumpy(A[:nc, g0:g1]) @ cp.asnumpy(B[:nc, g0:g1]).T
        times = []
        for _ in range(args.repeat):
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            if args.kchunk <= 0:
                C = A.dot(B.T)
            else:
                C = cp.zeros((b, b), dtype=cp.float64)
                for g0 in range(0, K, args.kchunk):
                    g1 = min(K, g0 + args.kchunk)
                    C += A[:, g0:g1].dot(B[:, g0:g1].T)
            cp.cuda.Device().synchronize()
            times.append(time.perf_counter() - t0)
        dt = min(times)
        flop = 2.0 * b * b * K
        err = float(np.abs(cp.asnumpy(C[:nc, :nc]) - ref).max())
        scale = float(np.abs(ref).max())
        print(f"[gemm] b={b:5d} K={K} kchunk={args.kchunk}: {dt:8.3f} s  "
              f"{flop/dt/1e12:6.1f} TFLOP/s  max|C-ref|={err:.2e} (max|ref|={scale:.2e}, "
              f"rel {err/scale:.1e})", flush=True)
        del A, B, C
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
