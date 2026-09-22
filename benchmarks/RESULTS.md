# gpu_ao2mo tile-GEMM work: measurements

System for every run: the 63-atom NV cell (`work/pprpa/NV63/grace_opt.xyz`),
gth-dzvp / gth-pbe, PBE, charge -3, ke = 300 Ha -> nao = 819, mesh 107^3,
ngrid = 1,225,043.  One NVIDIA B200 (gpu_devel), cupy 14.1.1, cuBLAS 12.8.
Active space AS = 300 means nocc = 128 (all valence), nvir = 300; AS = 100 is
100 + 100.  "base" is the kernel at commit 7475b2f (all tiles over the full
pair index); the driver is `bench_gpu_ao2mo.py`, the job `bench_gemm.sbatch`.

Consistency is `max|base - new|` over every element of each tensor, divided
by `max|base|`.  Everything below is exact bookkeeping, so the target is
round-off (1e-14).

## Stage 1: symmetric tiles (commit 590f9bf) -- job 27009007

Only the lower-triangle tiles of the Gram matrix, over p >= q pairs for
vvvv / oooo.

AS = 300, pair_blk = 600 (the production strip width):

| block | base | new | speedup | new GEMM | rate | rel. diff |
|---|---|---|---|---|---|---|
| vvvv | 1540.8 s | 232.7 s | 6.6x | 2.53 PFLOP | 10.9 TFLOP/s | 2.0e-14 |
| oovv | 285.2 s | 152.5 s | 1.9x | 1.84 PFLOP | 12.0 TFLOP/s | 2.0e-14 |
| oooo | 59.4 s | 10.2 s | 5.8x | 0.09 PFLOP | 8.7 TFLOP/s | 8.6e-15 |
| total | 2280.6 s | 874.5 s | **2.61x** | | | |

AS = 300, planner-chosen strips (1459 / 1200 / 1433 pairs): 271.6 s total,
**8.4x** over base, with the vvvv GEMM at 17.4 TFLOP/s -- the strip width
matters, which is what stage 2 is about.

AS = 300, pair_blk = 600, finals host-staged (the NV216 write path):
436.3 s total; the host tile scatter costs 14.0 s of the 243.9 s vvvv block.

AS = 100, pair_blk = 600: 67.7 s -> 24.0 s (2.83x), rel. diff <= 2.5e-14.

Unit tests in the same job: `test_pair_layout` (host-only, every strip
width), `test_gpu_ao2mo_multi` (1 vs 2 virtual slots vs pyscf), the planner
caps, and the module's own diamond-cell check (1.5e-13 vs pyscf).

## Stage 2: GEMM strip decoupled from the FFT batch (commit f138fc3) -- job 27089157

The potential of an outer strip is formed in FFT sub-batches into a real
buffer; the GEMM strip is sized from 24 B per pair-gridpoint instead of 64
and is no longer capped by the cuFFT plan limit.  "stage1" = commit 590f9bf.

AS = 300, planner-chosen strips:

| block | stage1 (strip) | new (strip) | rate stage1 -> new | rel. diff |
|---|---|---|---|---|
| vvvv | 149.4 s (1459) | 129.0 s (3305) | 17.2 -> 20.7 TFLOP/s | 2.2e-14 |
| oovv | 110.2 s (1200) | 80.6 s (2700) | 16.9 -> 24.0 TFLOP/s | 2.3e-14 |
| oooo | 8.3 s (1433) | 8.0 s (3248) | 11.8 -> 14.1 TFLOP/s | 2.3e-14 |
| total | 274.0 s | 231.1 s | **1.19x** (base 2280.6 s: **9.9x**) | |

Rate vs forced strip width, new kernel, AS = 300 (block "rate" = GEMM flops
over the whole block time, so it includes codensity rebuilds, FFTs and the
scatter):

| pair_blk | vvvv | oovv | total |
|---|---|---|---|
| 600 | 158.5 s, 15.9 TFLOP/s | 115.1 s, 15.9 | 288.0 s |
| 1200 | 134.6 s, 19.0 | 99.3 s, 18.8 | 248.1 s |
| 2400 | 124.1 s, 21.1 | 90.7 s, 21.2 | 228.7 s |
| planner, host-staged (5062 / 4800) | 119.5 s, 23.2 | 79.7 s, 25.5 | 226.7 s |

Host-staged finals (NV216 path) cost nothing extra now: 226.7 s vs 231.1 s
resident, with the host scatter at 13.2 s of the vvvv block.

Isolated tile GEMM `C[b,b] = A[b,K] B[b,K]^T`, fp64, warm (bench_gemm_shapes.py):

| b | K = 1.2e6 (NV63) | K = 4.0e6 (NV216) |
|---|---|---|
| 300 | 26.5 TFLOP/s | 25.0 |
| 600 | 27.0 | 24.3 |
| 1200 | 27.3 | 26.0 |
| 2400 | 28.5 | 33.3 |
| 4800 | 34.5 | (does not fit) |

So the bare GEMM is at 65-85% of the B200's fp64 peak (~40 TFLOP/s) from
b = 2400 up; the remaining gap between the in-kernel rate (21-25) and the
bare GEMM (28-34) is the codensity regeneration, the FFTs and the scatter,
not the GEMM shape.  At the production K the width still buys 33%.

## Stage 3: cuBLAS fp64 emulation -- job 27090186 (tile GEMM only)

cuBLAS >= 13.0 update 2 emulates DGEMM in fixed point on the integer tensor
cores (`CUBLAS_EMULATE_DOUBLE_PRECISION=1`, strategy `performant` lets the
library decide per call, `eager` forces it).  Bouchet's newest CUDA module
is 12.9.1 and the project venv runs cuBLAS 12.8, so this was measured from a
scratch venv (`/nfs/roberts/project/pi_tz324/sc3352/venv_cu13`: cupy-cuda13x
14.2.0 + pip CUDA 13.4 wheels, cuBLAS 13.8) on the same B200.

Tile GEMM, TFLOP/s (max rel. error vs numpy fp64 on a 64x64 corner):

| b | K | native | performant | eager |
|---|---|---|---|---|
| 600 | 1.2e6 | 27.3 (2.1e-14) | 27.2 (same call) | 23.6 (5.3e-15) |
| 1200 | 1.2e6 | 27.2 (3.6e-14) | 27.0 (same call) | 49.0 (2.5e-15) |
| 2400 | 1.2e6 | 28.7 (2.2e-14) | 71.7 (3.1e-15) | 72.6 (3.1e-15) |
| 600 | 4.0e6 | 25.4 (3.9e-14) | 25.4 (same call) | 23.6 (2.3e-15) |
| 1200 | 4.0e6 | 27.0 (5.6e-14) | 27.0 (same call) | 48.9 (2.7e-15) |
| 2400 | 4.0e6 | 33.1 (4.0e-14) | 70.9 (2.9e-15) | 71.4 (2.9e-15) |

Emulation is 2.2-2.6x native at b = 2400 and *more* accurate (the fixed-point
accumulation over K = 4e6 terms loses less than fp64's rounding); it only
pays from b ~ 1200 up, which is exactly the regime stage 2 puts the kernel
in.  The `performant` heuristic declines the b <= 1200 calls.  Nothing in
lib_pprpa has to change: the switch is the environment variable, read when
the cuBLAS handle is created.  What does have to change is the environment:
CUDA 13 runtime, cupy-cuda13x and a gpu4pyscf built for CUDA 13.
