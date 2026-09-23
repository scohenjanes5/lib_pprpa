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

## Stage 7: compare-only flow against a saved reference -- job 27106768

The stage-2 tensors are now stored once on scratch
(`/nfs/roberts/scratch/pi_tz324/sc3352/ao2mo_refs/nv63_ke300_as300_stage2`,
80 GB, with PROVENANCE.txt) and every later run compares against them with
`--compare`; no earlier kernel is recomputed.

Current kernel (R2C default, fused codensity, diagonal split), AS = 300:

| run | vvvv | oovv | oooo | blocks total | rel. diff vs stage2 ref |
|---|---|---|---|---|---|
| planner (4952 / 4200 / 4864) | 91.0 s | 66.6 s | 5.6 s | **163.3 s** | 2.7e-14 / 4.6e-14 / 4.0e-14 |
| forced 1500 (NV216-like) | 109.3 s | 77.7 s | 6.3 s | 193.2 s | 2.6e-14 / 3.0e-14 / 3.8e-14 |
| planner, host-staged (7581 / 7500) | 159.7 s | 79.7 s | 7.1 s | 246.5 s | 2.8e-14 / 3.0e-14 / 2.9e-14 |

Two problems in the host-staged run, both from the very wide strips the
unconstrained planner picked when nothing else was on the card:

* the bare tile GEMM is not monotonic in the strip width (K = 1.2e6):
  3300 -> 26.7, 4950 -> 34.1, 6000 -> 26.8, 7600 -> 27.5 TFLOP/s; the vvvv
  gemm phase went from 77.6 s at 4952 to 94.6 s at 7581;
* 41 s of the 159.7 s vvvv block were outside every timed phase: the two
  73 GB operand buffers were allocated per task, and near the card's
  capacity the pool had to free and re-map them every time.

Stage 8 caps the planner at 5000 pairs (`_PAIR_BLK_CAP`; explicit
`pair_blk` and `GPU_AO2MO_MAX_PAIR_BLK` still override) and allocates the
three strip buffers once per block per slot.

## Stage 8: strip cap and per-block buffers (commit below) -- job 27115393

`_PAIR_BLK_CAP = 5000` and the three strip buffers allocated once per block
per slot.  Compared against the saved stage-2 reference only.

| run | vvvv (phases: fft / gemm / rho_in / rho_out / scatter) | oovv | oooo | blocks total | rel. diff |
|---|---|---|---|---|---|
| planner, resident (4952 / 4200 / 4864) | 90.7 s (9.0 / 77.6 / 1.8 / 0.3 / 2.1) | 66.7 s | 5.6 s | **163.1 s** | 2.7e-14 / 4.6e-14 / 4.0e-14 |
| forced 1500 (NV216-like) | 110.0 s (11.0 / 92.3 / 4.5 / 0.3 / 1.8) | 77.4 s | 5.6 s | 193.0 s | 2.6e-14 / 3.0e-14 / 3.8e-14 |
| planner, host-staged (5000 / 4800) | 102.9 s (9.5 / 78.0 / 1.8 / 0.3 / 13.3) | 66.8 s | 7.8 s | **177.5 s** | 2.7e-14 / 6.4e-14 / 4.0e-14 |

The host-staged run is back at the resident GEMM rate (78.0 s vs 94.6 s at
7581-wide strips) and has no untimed time left (102.9 s vs 104.9 s of
phases); the 13 s host scatter of the 65 GB vvvv tensor is the only cost of
staging, and it does not grow with the cell (the tensor size is set by the
active space, so at NV216 it is the same 13 s against hours of GEMM).

## Where this leaves the kernel

NV63, AS = 300, one B200, block totals:

| kernel | resident | host-staged |
|---|---|---|
| 7475b2f (all tiles, full pair index, strip 600) | 2280.6 s | -- |
| stage 1 symmetric tiles, strip 600 | 874.5 s | 436.3 s (planner) |
| stage 2 strip decoupled from FFT | 231.1 s | 226.7 s |
| stage 6 fused codensity, R2C, diagonal split | 164.6 s | 246.5 s (7581 strips) |
| stage 8 strip cap, per-block buffers | **163.1 s** | **177.5 s** |

Every step is exact (<= 6.4e-14 relative against the previous kernel, and
the module's own check against pyscf stays at 1.5e-13).  The vvvv block is
now 86% tile GEMM at the B200's fp64 rate (77.6 s of 90.7); the FFT chain is
9 s, the codensity rebuild 2 s, the scatter 2 s.  The next factor of two is
the cuBLAS fp64 emulation (stage 4: 239 -> 123 s on the same cell), which
needs the CUDA 13 environment.  Overlapping the FFT of the next strip with
the running GEMMs on a second stream would hide most of the 9 s, but it
needs a second potential buffer (24 instead of 16 B per pair-gridpoint) and
at NV216 memory that narrows the strips from ~2200 to ~1500 pairs, where the
GEMM loses more than the FFT would gain; it was not pursued.

## Stage 9: the other two FFT kernels -- job 27132757

The shared FFT chain moved to `lib_pprpa/gpu_coulomb.py` (fused codensity
kernel, R2C potential with the symmetrised half kernel, per-point costs,
`plan_fft_batch`); each kernel keeps its own planner for what it contracts
with the potentials.  NV63, AS = 300, one B200, element-wise against the
frozen kernels (commit 3f56e5f copies in benchmarks/).

**Low-rank exchange** (`gpu_fft_k`), the two production sets X (rank 300) and
Y (rank 128), quantity compared = `mo_coeff.T @ K @ orbp`:

| variant | transforms | blocks | time | rel. diff vs base |
|---|---|---|---|---|
| base (full K, AO grid resident, 64 B/pt plan) | nao x r | row_blk 6 / 15, one rank chunk | 123.0 s | -- |
| new, full K (`ket=None`) | nao x r | row_blk 103, fft_blk 1500 (5 rows) | 81.6 s | 4.2e-11 / 6.0e-11 |
| new, `ket=orbp` (K @ orbp) | nact x r | row_blk 54, fft_blk 1500 | **42.6 s** | 7.5e-12 / 2.0e-11 |

The 1e-11 (not 1e-14) agreement is the expected rounding of moving the
Coulomb operator from the bra to the ket codensity: W is symmetric on an odd
mesh, so the two orders are equal in exact arithmetic, but the sums over
3.7e8 terms per element cancel heavily; both agree with dense
`fft_jk.get_k` inside the tests' 1e-9.  The ket path is what the gradient
uses (`grad/pprpa.py` now asks for `K @ orbp`); it needs r x nact instead of
r x nao transforms (1.9x fewer here, 4.7x at NV216) and never holds the AO
grid (90 GB at NV216).  Expected at NV216: the 29 min exchange build becomes
a few minutes.

**Pairing force** (`gpu_pairing_force`), rank 428 (= nocc + nvir), 91,806
pairs: 32.4 s -> 17.8 s (1.82x; job 27133118, strip 2058, no retries), rel.
diff 1.0e-15.  Fused codensity + R2C chain; the strip planner takes the
chain's cost from `gpu_coulomb` with 20% slack (the first run's 2573-pair
strip OOM'd once and finished in 23.0 s after the retry).

## Stage 10: the Davidson ERI stream -- job 27134027

`pprpa_eri_gpu` streams the three host-staged tensors to the GPU on every
Davidson matrix-vector product (MVP) when they do not fit resident: 78.7 GB
per MVP on NV63 at AS=300, 194 GB at NV216.  Measured on the saved stage-2
reference tensors, 32 trial vectors per MVP, best of 3, two B200s:

| mode | host memory | MVP | effective rate | rel. diff vs resident |
|---|---|---|---|---|
| resident (fits on NV63) | -- | 0.026 s | -- | -- |
| **split** (row halves on 2 GPUs, partials summed on the host) | pinned | 0.026 s | 0.1 GB/MVP of trial vectors | 1.5e-16 |
| tiled, as before | pageable | 3.71 s (first MVP 19.8 s) | 8.6 GB/s | 1.5e-16 |
| tiled | **pinned** | 1.39 s | 56.6 GB/s | 1.5e-16 |

Three changes, all in `pprpa_eri_gpu` and `gpu_ao2mo._host_tensor`:

* **Pinned staging.** ao2mo allocates a host-staged final with
  `cupyx.empty_pinned` (`GPU_AO2MO_PINNED=0` for pageable; falls back on
  failure).  The Davidson inherits it through the copy-free reshape, so its
  uploads run at the link rate instead of through the driver's bounce
  buffer: 6.6x on the stream (the memo's estimate was 2-2.5x).  At NV216,
  194 GB/MVP: ~22 s -> ~3.4 s per MVP, ~10 min -> ~1.7 min per force at 30
  MVPs.
* **Row strips.** `vvvv` and `oooo` are symmetric physicist matrices, so the
  tiled path now streams contiguous row strips (`V[P, :]` for `V[:, P].T`);
  the old column slices made numpy materialise a strided copy before every
  upload (part of the 19.8 s first MVP above).
* **Split-resident mode.** With a multi-slot `DeviceGroup` each slot holds a
  row range of every block; an MVP ships the trial vectors to each slot
  (~25 MB) and sums the partial products on the host.  Auto-selected when the
  whole set does not fit one slot but the per-slot shares do (NV216 on two
  B200s: 97 GB each), so the 2-GPU production jobs stop streaming the ERIs
  entirely: the 4-6 min Davidson phase becomes seconds.  Validated against
  the resident and CPU contractions with two virtual slots (tests) and two
  real devices (this job).

Per-MVP bytes / seconds / GB/s are now in the telemetry and printed at
`release_gpu_eri`, which is the measurement the memo asked for first.  The
remaining memo items (sub-tile double buffering, deterministic tile size)
are moot for the split mode and worth little for the pinned tiled one: the
GEMM is a few percent of the transfer, and the tile only shrinks below a
whole block on a contended card.

## Stage 11: the full NV216 force on two B200s -- job 27138544

`work/pprpa/NV216/gpu_ke300/bench/gemm_opt_2gpu/`, the exactly-C3v
`opt/input.vasp`, ke = 300 Ha, gth-dzvp, hh / triplet / AS = 300, driver =
the pair_blk A/B `calc_forces_gpu.py`, `LIB_PPRPA_GPUS=2`, gpu_devel
(256 GB host cap held).  Reference: the 1-GPU Sep 9 / 11 force.

| | this run | reference |
|---|---|---|
| E_state | -1226.51751887 Ha | -1226.51751887 Ha |
| \|F\| | 3.08748e-02 a.u. | 3.08750e-02 a.u. |
| max\|F\| | 1.02903e-02 a.u. | 1.02900e-02 a.u. |
| max\|dF\| vs reference | 2.08e-08 eV/A (8.4e-08 relative) | |

(The Sep 18 regression run at the same geometry gave |F| = 5.06e-02.)

Phases, against the previous 2-GPU run of the old tree (job 25938090):

| phase | old tree, 2 GPUs | gemm-opt, 2 GPUs |
|---|---|---|
| SCF | 4 min | 4.0 min |
| ao2mo (3 blocks) | 3 h 16 min | 14.6 min |
| Davidson (17 iterations) | 4 min | 2.75 s (split-resident) |
| relaxed density (K build + CPHF) | 30 min | 17.4 min (K 1.9 min, CPHF 15.0 min) |
| hcore derivative | 11 min | 11.5 min |
| J, pairing K, Vxc/fxc, overlap | ~3 min | 1.6 min |
| **total** | **4 h 09 min** | **49.6 min** |

ao2mo blocks: vvvv 154.6 s at 53.6 TFLOP/s over both cards (2118-pair
strips, 8.29 PFLOP), oooo and oovv likewise; host scatter 18.7 s per staged
block.  The 194 GB pinned host set uploaded to the two cards in 1.7 s.  The
CPHF is now the largest phase: one GPU, and each of its ~25 iterations
re-evaluates the GGA AOs on the full grid and rebuilds both densities densely
(see the note at the end of stage 10's section for the fix).

## Stage 12: the CPHF response -- job 27145780

`lib_pprpa/grad/gpu_response.py`: the ground-state XC kernel is cached per
slot, the Coulomb potential of the perturbing density is folded into the XC
weights (one GEMM for J + fxc instead of three dense contractions and a second
90 GB AO grid), and the grid is split over the device group.  NV63, PBE,
CPHF-shaped perturbing density, best of 3 calls, two B200s:

| response | per call | breakdown | rel. diff vs dense |
|---|---|---|---|
| dense (`make_gpu_vresp`, 1 GPU) | 0.93 s | | -- |
| grouped, 1 slot | 0.52 s | pass1 0.14 / Coulomb 0.04 / pass2 0.38 | 1.2e-12 |
| grouped, 2 slots | 0.27 s | pass1 0.07 / Coulomb 0.03 / pass2 0.20 | 1.2e-12 |

The dense call scales as nao^2 x ngrid, which puts NV216 at 39x NV63 -- the
36 s per CPHF iteration measured in stage 11 is exactly that -- so the
grouped response on two cards should make the 15 min CPHF about 4.5 min.
Tests: LDA and GGA against the dense response, one and two virtual slots,
small forced chunks, and the `PPRPA_RESPONSE=dense` fallback.

## Stage 13: NV216 force on two B200s with the grouped response -- job 27147836

Same setup as stage 11 (`bench/gemm_opt_2gpu/`).  E_state -1226.51751887 Ha,
|F| 3.08748e-02, max|F| 1.02903e-02 a.u., max|dF| vs reference 5.1e-09 eV/A
(2.6e-08 relative).  The old-tree regression job 27006823 finished the same
day and also reproduced the reference on 1 and 2 GPUs (max|dF| 5.5e-9 /
2.1e-8 eV/A), so the Sep 18 failure was transient.

| phase | stage 11 | stage 13 |
|---|---|---|
| relaxed density | 17.4 min | 7.2 min (K 1.9, CPHF 5.1) |
| CPHF response | 36 s/call, 1 GPU | 4.49 s/call, 68 calls, 2 GPUs (pass1 1.33 / Coulomb 0.01 / pass2 3.15) |
| total force wall | 49.6 min | **39.4 min** |

Note the solver made 68 response calls, not the ~25 assumed from the phase
time -- the old response was ~13 s per call, not 36; the per-call speedup is
2.9x on two cards (3.5x on NV63) and the CPHF phase went 15.0 -> 5.1 min.
Against the reference 2-GPU force of the old tree (4 h 09 min) the full
force is now 6.3x faster; against the 1-GPU old tree (8 h 08 min), 12.4x.
The hcore derivative (11.5 min, 215 `hcore_generator` calls over two cards)
is now the largest phase.

## Stage 14: the hcore derivative in density space -- job 27160009

`lib_pprpa/grad/gpu_hcore_force.py`.  After stage 13 the hcore derivative was
the largest phase of the NV216 force (11.5 min of 39.4).  The per-atom
`hcore_generator` builds the full AO matrix of the local-PP derivative and the
force then traces it against the density; since the AO indices are contracted
on both sides by the same grid point, the trace collapses to
`sum_g vloc_R^x_A(g) rho_T(g)` and, by Parseval, to a reduction over G with no
FFT per atom.  `rho_T` is built once, transformed once, and each atom becomes
three dot products over G:
`natm * 3 * nao^2 * ngrid` -> `nao^2 * ngrid + natm * 3 * ngrid`.
The derivation is section 15 of `docs/NV216_GPU_GRADIENT.md`;
`PPRPA_HCORE=dense` restores the per-atom path.

Correctness, four independent checks:

| check | agreement |
|---|---|
| pyscf's *CPU* `hcore_generator`, distorted diamond, symmetric T | 1.8e-15 rel. |
| same, non-symmetric T | 8.5e-16 rel. |
| gpu4pyscf `hcore_generator`, NV63 ke=600 (nao 819, mesh 151^3) | 6.06e-15 rel. |
| full `grad_elec`, default vs `PPRPA_HCORE=dense` (C2 diamond) | 2.77e-15 rel. |

(The CPU generator returns `[3, nkpts, nao, nao]`; gpu4pyscf returns
`[nkpts, 3, nao, nao]`.  Only the second convention is what the force assembly
contracts.)

Tests: `tests/test_gpu_hcore_force.py` -- pseudo and all-electron branches, one
and two slots, forced grid chunks, forced atom batches (including a partial
last batch on a 4-atom supercell), grid ordering, and the end-to-end dispatch.
`tests/test_scf_chk.py` (CPU only) covers the SCF checkpoint of section 16.
Job 27160009: 8 + 7 + 90 tests passed, no regressions.

NV63, ke=600, nao 819, mesh 151^3, one GPU (RTX PRO 6000 Blackwell):

| path | wall | per atom |
|---|---|---|
| per-atom `hcore_generator` | 786.1 s | 12.48 s |
| density-space | 21.8 s | -- |
| | **36.0x** | |

max abs diff 1.596e-14 (6.06e-15 relative against max\|de\| = 2.63 a.u.),
rms 5.13e-15.

The breakdown confirms the cost model term by term: `rho 4.8 / G-space 3.0 /
AO-derivative 14.0`.  The G-space atom reduction is 3.0 s for *all 63 atoms*;
the 14.0 s AO-derivative term is `krhf.get_hcore`, one array rather than
per-atom work, and costs almost exactly one atom of the old loop (12.48 s).
So the 63-atom phase now costs ~1.7 atoms' worth, and what remains is dominated
by the one term that was never per-atom.

Note on hardware: this job landed on an RTX PRO 6000 Blackwell, whose fp64 rate
is ~1/64 of its fp32.  The per-atom path is an fp64 GEMM wall, so its *absolute*
numbers here are far worse than a B200's (which is why the NV216 step of 27160009
ran out of walltime -- 12.48 s/atom at NV63 extrapolates to ~10 h at NV216 on
that card).  The 36x ratio is measured on one card for both paths and both are
fp64-GEMM bound, so it transfers; the absolute times do not.

NV216 density-path timing: job 27225292 (pending).  The per-atom reference is
deliberately *not* re-run there -- its timing is already in this file (stage 11:
11.5 min on two B200s) and in the table of `docs/NV216_GPU_GRADIENT.md` (24 min
on one), and its value is validated by the four checks above.  What that job adds
is the one thing smaller cells cannot reach: the G-space reduction batches atoms
by free VRAM and every cell so far fits in a single batch (NV63 is 63 atoms
against a cap of 64), so `--selfcheck` re-runs the reduction with a forced small
atom batch and grid chunk and requires the two to agree.

## Stage 14: the hcore derivative in density space -- allocation 27259554

`lib_pprpa/grad/gpu_hcore_force.py`.  The per-atom `hcore_generator` loop is
replaced by one grid density, one FFT and gpu4pyscf's own
`multigrid.eval_vpplocG_SI_gradient` (the reduction `krhf.grad_elec` already
uses on its `multigrid_v2` branch), plus `contract_h1e_dm(..., hermi=0)` for the
AO-derivative half.  109 lines, 38 of them executable; a first version that
re-derived the reduction was 340.

One B200, `benchmarks/bench_hcore_force.py`:

| cell | per-atom generator | density space | agreement | speedup |
|---|---|---|---|---|
| NV63, ke=600, nao 819, mesh 151^3 | 52.7 s (0.84 s/atom) | 6.0 s | 6.33e-15 rel. | 8.8x |
| NV216, ke=300, nao 2795, mesh 159^3 | 6.10 s/atom -> 21.8 min (8/215 sampled) | 17.8 s | 9.28e-14 rel. | 74x |

Breakdown at NV216: rho 2.8 s, G-space reduction 0.1 s, AO-derivative
(`krhf.get_hcore`) 14.9 s.  The sampled per-atom rate extrapolates to 21.8 min
against the 24 min recorded for one GPU in stage 11's table, so the sample is
not flattering itself.

An RTX PRO 6000 Blackwell (fp64 ~1/64 of a B200) gives 786.1 s -> 21.8 s at
NV63 (job 27160009) and 189.6 s at NV216 (job 27225292); same ratios, slower
card, not the numbers quoted above.

Tests (`tests/test_gpu_hcore_force.py`, 6 passed): pseudo and all-electron
against `hcore_generator`, a non-symmetric density (which is what `hermi=0`
buys), the device group being inert, the `PPRPA_HCORE=dense` switch, and the
whole force through `grad_elec` both ways (3.90e-15).
