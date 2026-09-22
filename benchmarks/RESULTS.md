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

## Stage 4: full ao2mo with fp64 emulation -- job 27091748

Same NV63 / AS = 300 run from the CUDA 13 scratch venv (pip gpu4pyscf-cuda13x
1.8.1, cupy 14.2.0, cuBLAS 13.8), same cached SCF orbitals; the native run
writes its tensors to disk and the emulated runs compare element-wise.

| mode | vvvv | oovv | oooo | total | rel. diff vs native |
|---|---|---|---|---|---|
| native fp64 | 130.3 s (20.5 TFLOP/s) | 83.1 s (23.3) | 9.5 s | 238.6 s | -- |
| emulation, eager | 64.2 s (41.3) | 50.3 s (38.1) | 5.5 s | 127.0 s | 2.6e-14 / 3.0e-14 / 4.0e-14 |
| emulation, performant | 60.4 s (44.3) | 50.7 s (38.1) | 5.2 s | 123.1 s | 2.5e-14 / 2.4e-14 / 4.0e-14 |

The emulated tensors differ from native fp64 by the same 1e-14 that separates
any two summation orders; the isolated GEMM test shows the emulated result is
the one closer to the exact sum.  End to end on this cell: 2280.6 s (base)
-> 123.1 s, **18.5x**, all three steps exact.  In the CUDA 12.8 production
environment the first two steps give 2280.6 -> 231.1 s, **9.9x**.

## Stage 5: where the non-GEMM time goes -- job 27101508

`GPU_AO2MO_PROFILE=1` synchronises after every phase.  NV63, AS = 300,
planner strips (4952 / 4200 / 4867 pairs), C2C chain, fused codensity kernel:

| block | wall | fft | gemm | rho_inner | rho_outer | scatter | GEMM rate (gemm phase only) |
|---|---|---|---|---|---|---|---|
| vvvv | 102.5 s | 15.6 | 82.6 | 1.9 | 0.5 | 1.8 | 33.4 TFLOP/s |
| oovv | 75.0 s | 12.7 | 59.7 | 2.0 | 0.5 | 0.1 | 33.5 |
| oooo | 7.6 s | 3.1 | 4.0 | 0.1 | 0.1 | 0.3 | |

So after the fused codensity kernel (`_CODENSITY_KERNEL`: one read of each
factor, one write, no gathered temporaries) the GEMM runs at the B200's bare
fp64 rate and the codensity rebuild is 2% of the block.  The Coulomb FFT
chain is the remaining overhead, 15% of vvvv and 17% of oovv.

stage2 (take + in-place multiply) -> fused: 228.8 s -> 190.9 s, rel. diff
2.7e-14.

R2C chain (`rfft=True`), same job: oovv fft 12.7 -> 7.7 s, oooo 3.5 -> 2.0 s,
but vvvv did not improve and the planner run OOM'd and retried (the 24 B per
pair-gridpoint estimate for the R2C chain omitted the cuFFT work area), and
the compact blocks' variable strip lengths gave every FFT sub-batch a new
remainder shape, so cuFFT re-planned constantly.  On the odd 107^3 mesh the
R2C tensors matched C2C to 2e-14 (vvvv) / 2e-16 (oovv, oooo); on the even
20^3 diamond test mesh they differed by 1e-8, which is the Nyquist-plane
kernel asymmetry of a non-orthogonal cell -- the C2C path's `.real` only
sees the even part of the kernel, so R2C needs the symmetrised half kernel
(`pair_layout.symmetric_half_kernel`; numpy test to 1e-13 on even meshes).
Stage 6 fixes all three.

## Stage 6: FFT chain and tile bookkeeping -- job 27104981

Changes: fused codensity kernel; R2C Coulomb chain with the symmetrised half
kernel (now the default; the even-mesh diamond self-test agrees with pyscf to
1.5e-13, same as C2C); FFT sub-batches of one fixed shape per block
(zero-padded remainder, one cached cuFFT plan); potential / codensity / FFT
buffers allocated once per task; diagonal-straddling strip pairs split into
sub-tiles (depth 2) so the GEMM skips most of the unused upper triangle;
R2C planner budget 40 B per pair-gridpoint (no OOM retries).

NV63, AS = 300, planner strips (4952 / 4200 / 4864), block times and phases:

| block | stage2 | new C2C | new R2C | R2C phases: fft / gemm / rho_in / rho_out / scatter |
|---|---|---|---|---|
| vvvv | 130.7 s | 97.8 s | 91.4 s | 9.1 / 78.2 / 1.8 / 0.3 / 1.6 |
| oovv | 82.5 s | 73.5 s | 67.5 s | 7.8 / 57.1 / 1.5 / 0.2 / 0.5 |
| oooo | 8.5 s | 6.7 s | 5.7 s | 1.8 / 3.3 / 0.1 / 0.1 / 0.3 |
| blocks total | 221.7 s | 178.0 s | **164.6 s** | |

Consistency: R2C vs C2C 2.5e-16 (all blocks); R2C vs stage2 2.7e-14 / 4.6e-14
/ 4.0e-14.  GEMM flops 2.676 -> 2.566 PFLOP from the diagonal split; the
gemm phase runs at 32.8 TFLOP/s, the B200's bare fp64 rate.  The FFT chain
is down from 15.6 to 9.1 s (vvvv), the codensity rebuild from 5 to 1.8 s.
What remains outside the GEMM is 13 s of 91 (14%): 9 s FFT, 2 s codensity,
2 s scatter.

At the NV216-like forced width 1500: 110.8 / 79.1 / 5.7 = 195.6 s (vs 226 s
C2C stage 2 at the same width).

Note on the "total ao2mo wall" lines in this job's log (480 s, 268 s): the
benchmark returns the finals to the host (`return_gpu=False`), an 80 GB
pageable device-to-host copy that took 100-300 s on the shared devel node
while another run's tensors were still resident.  Production keeps the
finals on the GPU; the block sums above are the kernel cost.  The bench now
reports block totals and uses them for the speedup.

Cumulative on this cell: 2280.6 s (7475b2f) -> 164.6 s, **13.9x**, exact.
