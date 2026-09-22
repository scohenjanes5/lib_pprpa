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
