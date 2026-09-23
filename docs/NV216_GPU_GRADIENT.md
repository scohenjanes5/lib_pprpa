# From "does not run" to a 4-hour gradient: the NV216 pp-RPA force on GPUs

This note records every change that was needed to compute a Gamma-point pp-RPA
nuclear gradient for the 216-atom NV-centre cell on Bouchet GPUs, in the order
the problems appeared, with the measured effect of each. All numbers are from
production Slurm logs (September 2026). Job ids refer to the Bouchet cluster.

## The system

| quantity | value |
|---|---|
| cell | 3x3x3 diamond supercell with one N and one vacancy, 215 atoms, charge -3 |
| basis / pseudopotential / functional | gth-dzvp / gth-pbe / PBE |
| AO functions | nao = 2795 |
| plane-wave cutoff | ke = 300 Ha (needed to suppress a spurious net force) |
| FFT mesh | 159^3 = 4,019,679 grid points |
| pp-RPA active space | 300 occupied + 300 virtual (AS = 300), hh channel, triplet |
| MO-ERI blocks (vvvv, oovv, oooo) | 3 x 64.8 GB, host-staged |
| GPUs | NVIDIA B200 (183 GB), H200 (141 GB), H100 (80 GB) |

Everything in the pipeline is float64 FFT and GEMM work: GPU KRKS SCF, GPU FFT
ao2mo into the active space, a batched MO-ERI Davidson solve, then the GPU
Gamma-point gradient (relaxed density via CPHF, and the force assembly).

## Timeline of failures and fixes

### 0. Before this work: host memory and the MO grid

Two earlier failures had already been fixed by the tiled-ERI commits:

- Host OOM kill (job 25519940): the three 64.8 GB ERI blocks live on the host
  for the tiled Davidson. `--mem=250G` was not enough; the sbatch files now
  request `--mem=400G` (measured peak RSS 261 GB). "Tiled" refers only to how
  the matrix-vector product is executed: the Davidson eigensolver is unchanged,
  but instead of holding vvvv/oovv/oooo on the GPU and running one GEMM per
  block ("resident" mode, still the default when they fit in 75 % of VRAM),
  the 194 GB of integrals stay in host RAM and row-slices are streamed to the
  GPU for the same GEMMs. Both modes give identical results.
- `_mo_on_grid` OOM (job 25573557): fixed by the tiled `gpu_ao2mo` /
  `pprpa_eri_gpu` path (commits 3665c52, 12a540b, 7c805ca). That path is now
  the largest single phase of the force; appendix A derives it, and section 6
  is a regression in its strip planner.

With those in place, job 25686365 (9 September) became the first run to reach
the gradient: SCF 4 min, ao2mo 5 h 14 min, Davidson 4 min, then a CUDA OOM.

### 1. The dense exchange build could not fit on any GPU

**Failure.** The relaxed-density step calls `mf.get_k` for the two pp-RPA
amplitude densities X and Y. The driver routed that to gpu4pyscf's dense FFT
`fft_jk.get_k_kpts`, which holds several nao-by-grid arrays at once:

| array | size at nao=2795, 159^3 |
|---|---|
| `ao` (AOs on the grid) | 90 GB |
| `vR_dm` (crashed here) | 180 GB |
| `ao_dms` | 180 GB |
| one AO row's codensity + its complex FFT | 90 + 180 GB |

The existing "adaptive blksize" patch in the gpu4pyscf fork could not help: even
a block of one row needs 270 GB of transient. This was structural.

**Fix: low-rank exchange** (`lib_pprpa/gpu_fft_k.py`, hook in
`lib_pprpa/grad/pprpa.py`). X_ao = C_vir x C_vir^T and Y_ao = C_occ y C_occ^T
have rank at most 300. Writing D = L R^T, the exchange matrix

    K_pq = sum_jk (pj|kq) D_jk = sum_m  int int ao_p phiL_m  v  phiR_m ao_q

is built from (AO, active-MO) codensities: only the AO grid stays nao-by-grid,
the FFT batch is `row_blk * rank_blk * ngrid`, and there are about nao/300
times fewer FFTs than the dense kernel would need. `make_rdm1_relaxed_rhf_pprpa`
uses `mf.get_k_lowrank` when the driver attaches it
(`gpu_fft_k.attach_lowrank_getk`), otherwise falls back to dense `get_k`. The
L R^T split (not an eigendecomposition) matters because the triplet core is
antisymmetric.

**Validation.** Matches dense `fft_jk.get_k(hermi=0)` to ~1e-15 on a diamond
cell for symmetric, antisymmetric and general low-rank densities.

**Result** (job 25711753): the gradient completed for the first time, 22 h 53 min.
The exchange build took 29 min.

### 2. Timestamps in every log line

Nothing in the logs said where the day went. Every GPU progress line now
carries a wall-clock stamp (`pprpa_util.tstamp`), and `grad_elec` prints a
per-phase seconds summary. The 22 h 53 min run decomposed as:

| phase | wall | share |
|---|---|---|
| pairing-K force (gpu4pyscf AFT `get_ek_ip1`, dense X, 35,890 G-blocks) | 16 h 21 min | 71 % |
| ao2mo | 5 h 14 min | 23 % |
| relaxed density (low-rank K + CPHF) | 44 min | 3 % |
| hcore derivative (215 x `hcore_generator`) | 23 min | 2 % |
| SCF, Davidson, J, Vxc/fxc, overlap, PP | ~10 min | <1 % |

### 3. Side quest on the 63-atom ke=600 chain: cuFFT size limit and a runaway resubmission loop

While the 216 cell ran, the 63-atom ke=600 optimisation chain failed seven
times in 25 minutes with `CUFFT_INVALID_SIZE`. Cause, confirmed by a probe on a
B200: that cell's mesh is 151^3, and 151 is a prime above 127, so cuFFT uses
its Bluestein (chirp-z) path, which is limited to 2^31 elements per batched
plan. The ao2mo strip planner sized strips from free VRAM alone; on a B200 it
chose 1200 pairs x 3.44 M points = 4.1e9 elements. On an H100 it had chosen
300 and passed. Direct-path meshes (107, 128, 159 = 3 x 53) ran 1.6 x 2^31
without complaint.

Fixes:

- `gpu_mem.max_fft_batch(ngrid, mesh)` caps every batched FFT plan: strict
  2^31 when a mesh dimension needs Bluestein, a relaxed 1.5 x 2^31 otherwise
  (see section 6). Applied in the ao2mo, exchange and pairing-force planners
  -- and, it turned out, applied to the wrong quantity in all three: see
  section 8 and section 11.
- `CUFFT_INVALID_SIZE` and `CUFFT_ALLOC_FAILED` are treated as retryable, so
  an oversized batch halves instead of aborting.
- The resume chain (`pprpa_opt_gpu.sbatch`) got a circuit breaker: two
  consecutive failures faster than 600 s cancel the queued follow-up and write
  `opt_halted`; exit 0/2 or a slow failure resets the counter.

### 4. The pairing-K force: 16 hours to 3 minutes

**Problem.** The pairing exchange force `2 sum_{i in A} sum_l (d_x K[X])_il X_il`
was evaluated by gpu4pyscf's analytic-Fourier-transform kernel on the dense
nao x nao amplitude density: 35,890 G-blocks, each building an
nao x nao x 112 complex tensor.

**Fix: low-rank FFT pairing force** (`lib_pprpa/gpu_pairing_force.py`,
default in `pprpa_gamma_gpu.Gradients`). With X = L R^T (L = [C_vir x, C_occ y],
R = [C_vir, C_occ], rank 600 because the vir-occ cross pairs are needed), the
CPU reference term becomes

    de[A, x] = -2 sum_{i in A} sum_m L_im G[x,i,m]
    G[x,i,m] = (vol/ngrid) sum_g d_x i(g) W_m(g)
    W_m      = sum_n phiL_n V_nm,   V_nm = ifft(coulG * fft(phiR_n phiR_m))

so the cost is one FFT pass over the r(r+1)/2 = 180,300 active-MO codensity
pairs (V is symmetric) plus one chunked pass over the gradient AOs. The formula
is exact for a general X, so singlet (symmetric) and triplet (antisymmetric)
amplitudes need none of the sign bookkeeping the AFT path required. The minus
sign is the electron-coordinate gradient turning into the nuclear derivative.
The AFT kernel remains available via `PPRPA_PAIRING_K=aft` for cross-checks.

**Validation.** Against pyscf's CPU `get_k_e1` reference: below 1e-9 relative.
End-to-end gradient on the diamond cell, low-rank vs AFT: 2e-13 (triplet),
3e-10 (singlet); vs the CPU gradient: 1e-12 and 2e-8.

**Result.** 16 h 21 min became 2.8 min on one B200 (and 1.5 min on two).

### 5. Opt-in two-GPU dispatch

**Motivation.** With the pairing force gone, ao2mo dominated. Each remaining
hot loop (ao2mo pair strips, exchange AO-row batches, pairing codensity strips,
hcore per atom) has the same shape: read-only arrays, independent tasks with
global boundaries, outputs written to disjoint ranges or summed.

**Fix: `lib_pprpa/gpu_multi.py`.** `DeviceGroup` runs one thread per GPU
slot inside `with cp.cuda.Device(id)`, pulls tasks from a shared lock-guarded
iterator (dynamic balance), replicates read-only arrays by chunked
`cudaMemcpyPeer` (host bounce if no peer access), and on an OOM shrinks only
that slot's sub-block and re-runs the same task. Task boundaries never depend
on the slot count. With one slot the code runs inline with no `Device` switch,
so the single-GPU path is unchanged.

It is opt-in: `LIB_PPRPA_GPUS=2`, exported only by the 2-GPU Slurm scripts
(`pprpa_force_gpu2.sbatch`, `pprpa_opt_gpu2.sbatch`,
`submit_pprpa_opt_resumable.sh --gpus 2`; the spectrum workflow's
`setup_compute_env` sets it from `CUDA_VISIBLE_DEVICES`). gpu4pyscf's own
`lib/multi_gpu.py` was not used directly because it always takes every visible
device, cannot run two virtual slots on one GPU for tests, and has `*kwargs`
and `for i in num_devices` bugs. Its idioms were reused.

Device-bound gpu4pyscf objects (`FTOpt`, `_GTOvalOpt`, `UniformGrids`, the
`hcore_generator` closures) are built inside each slot's thread; cupy raises
loudly on cross-device use, so mistakes cannot be silent.

**Validation.** Every kernel agrees between one slot, two virtual slots on one
GPU, and two real NVLink-connected B200s to 1e-10 or better.

**Result** at the finished 216 geometry, both jobs submitted the same second:

| job | GPUs | queue wait | run | queue + run | GPU-hours | max dF vs reference |
|---|---|---|---|---|---|---|
| 25938089 | 1 | 22 min | 8 h 08 min | 8 h 31 min | 8.1 | 1.8e-9 eV/A |
| 25938090 | 2 | 47 min | 4 h 09 min | 4 h 56 min | 8.3 | 7.8e-11 eV/A |

The second GPU is essentially free in GPU-hours; it halves wall-clock.

### 6. A regression I introduced, and its fix

The strict 2^31 cap from section 3 was applied to every mesh. On 159^3 it
allows 534 pairs per strip, which whole-MO-row alignment rounds down to 300.
The reference run had used 600 for two of the three ao2mo blocks (1.5 h each
instead of 2.2 h), so ao2mo got slower: 6 h 35 min on one GPU. The fix
(`gpu_mem.needs_bluestein`) applies the strict cap only when a mesh dimension
has a prime factor above 127 and a relaxed 1.5 x 2^31 otherwise, inside the
range the probe tested. On 159^3 the strip goes back to 600 pairs; on 151^3 it
stays strict. Expected ao2mo: about 4.5 h on one GPU, 2.3 h on two.

The ao2mo planner also lost its whole-tensor restart: an OOM now redoes only
the failing strip on the failing device, and its per-element budget was raised
from 40 to 64 bytes so the first strip size is realistic (the reference run
tried 900, OOMed, and fell to 300).

### 7. The ao2mo GEMM did eight times the work it needed

**Problem.** Appendix A treats the 65 PFLOP per ao2mo block as a fixed cost.
It is not. The kernel computed every (outer, inner) tile of the pair Gram
matrix `E = rho W rho^T` over the full `nA*nB` pair index. But `E` is
symmetric, so tile `(J, I)` is the transpose of `(I, J)`; and with real
orbitals the codensity of `(p, q)` equals that of `(q, p)`, so for vvvv and
oooo the pair index only needs `p >= q`. pyscf's own `ao2mo` exploits both.
The full index had been kept because it makes the chemist-to-physicist write a
contiguous rectangle (A.3) -- a memory-bound scatter worth seconds, paid for
with hours of GEMM.

**Fix** (`lib_pprpa/pair_layout.py`, 22 September). `PairLayout` owns
row-aligned strips over either pair index; `write_tile` scatters each tile
together with its transpose and permutation images (rectangles for oovv,
`(p+1) x (r+1)` sub-blocks for the compact blocks), writing only canonical
`r <= p` row pairs so two tiles never overlap and a retried strip stays
idempotent. GEMM flops: 1/8 for vvvv and oooo, 1/2 for oovv.

**Validation.** A host-only numpy test mirrors the strip loop against a
brute-force tensor for every strip width; the GPU result agrees with the old
kernel to 2e-14 relative on every element of every block (NV63, AS = 300).

**Result** (NV63, nao 819, 107^3, AS = 300, one B200; the same active space as
NV216, so the tensors are the same size): 2281 s -> 875 s at the production
strip width of 600, -> 272 s with the planner's wider strips. The vvvv block
alone: 1541 s -> 233 s.

### 8. The strip width was tied to the FFT batch

**Problem.** The measured 12 TFLOP/s of section A.4 was blamed on the
tall-skinny GEMM shape, and that was half right: the tile GEMM has
`M = N = b` against `K = ngrid`, and a B200 delivers 26 TFLOP/s at b = 600 but
34 at b = 4800. The other half was the planner: it sized the strip for the
FFT chain (64 B per pair-gridpoint of complex transients) and capped it at
the cuFFT plan limit (section 3), although the GEMM needs neither. The GEMM
operands are two real strips, 8 B per pair-gridpoint each.

**Fix.** The potential of an outer strip is formed in FFT sub-batches
(`fft_blk`, ~15 % of the budget) into a real strip buffer, and the GEMM strip
is sized from its own 16 B per pair-gridpoint (`_plan_strips`). Strips went
from 1200-1459 pairs to 2700-3300 (5000 when the finals are host-staged),
and the vvvv GEMM from 17 to 21-25 TFLOP/s in the kernel.

Two corrections followed from measurement rather than reasoning. The bare
tile GEMM is not monotonic in the width -- 3300: 26.7, 4950: 34.1, 6000:
26.8, 7600: 27.5 TFLOP/s -- and a strip of 7581 pairs also made the pool free
and re-map two 73 GB operand buffers on every task (41 s of a 160 s block
outside every timed phase). The planner now caps the strip at 5000 pairs
(`_PAIR_BLK_CAP`) and the three strip buffers are allocated once per block.

**Result.** NV63 resident 274 -> 231 s; host-staged 246 -> 178 s after the
cap. The FFT batch, not the GEMM strip, is the only quantity the cuFFT cap
applies to now.

### 9. cuBLAS fp64 emulation: 2.2x on the kernel, nothing end to end

Since CUDA 13.0 update 2, cuBLAS can emulate DGEMM in fixed point on the
integer tensor cores (`CUBLAS_EMULATE_DOUBLE_PRECISION=1`). Bouchet's newest
CUDA module is 12.9.1 and the project venv runs cuBLAS 12.8, so this was
measured from a scratch venv with the pip CUDA 13 wheels
(`/nfs/roberts/project/pi_tz324/sc3352/venv_cu13`, cupy-cuda13x, cuBLAS 13.8,
gpu4pyscf-cuda13x) on the same B200.

At b = 2400 the tile GEMM runs at 71 TFLOP/s against 28-33 native, 2.2-2.6x,
and the emulated result is *closer* to the exact sum (3e-15 vs 4e-14 on a
64 x 64 corner: the fixed-point accumulation over K = 4e6 terms loses less
than fp64's rounding). It only pays from b ~ 1200 up, which is exactly where
section 8 puts the kernel. The full NV63 ao2mo in that venv: 239 -> 123 s,
tensors within 4e-14 of native.

**That was ao2mo alone at ke = 300, and it does not survive contact with the
production setting.** Measured 2026-09-23 on one B200, the whole NV63
gth-dzvp ground-state force at ke = 600 Ha (mesh 151^3, AS = 300, hh,
triplet, istate 0 -- the `bench_1gpu` workload), all three environments in
one job:

| phase (s) | project venv | CUDA 13 | CUDA 13 + emulation |
|---|---|---|---|
| SCF | 43 | 48 | 77 |
| ao2mo | 539.0 | 535.0 | 537.0 |
| CPHF relaxed density | 148.2 | 150.4 | 148.9 |
| hcore derivative | 53.7 | 53.7 | 58.1 |
| pairing K | 38.9 | 39.9 | 42.0 |
| J + Vxc/fxc | 6.8 | 7.2 | 6.8 |
| **wall** | **839** | **843** | **917** |
| | 1.00x | 1.00x | **0.91x** |

Forces agree (max|dF| 4.4e-12 and 3.6e-11 eV/A against the project venv, on
max|F| = 3.45e-03), and the 75-test suite passes unmodified under
gpu4pyscf 1.8.1 / pyscf 2.14 / cupy-cuda13x. So the CUDA 13 move by itself is
**neutral**, and the emulation is a **9 % loss**.

**Why.** The strip width is set by VRAM against the grid, and at ke = 600 the
grid is 2.8x the ke = 300 one: the planner picks `pair_blk = 1648` for `vvvv`
and 1500 for `oovv`, not 2400. Probed at exactly those shapes
(K = ngrid = 3442951):

| b | native | `performant` | `eager` |
|---|---|---|---|
| 1500 | 26.4 TFLOP/s (rel 3.7e-14) | 26.3 -- declined | 60.0 (rel 3.2e-15) |
| 1648 | 33.4 TFLOP/s (rel 3.5e-14) | 33.5 -- declined | 55.7 (rel 2.4e-15) |

1500-1648 falls in the gap between the b <= 1200 the `performant` heuristic
refused above and the 2400 it accepted, so emulation never fires and only its
per-call overhead is paid -- visible as SCF 43 -> 77 s and grad_elec
247.9 -> 262.5 s.

**And `eager` is not the way out**, because the switch is global. Forced on,
it does everything the kernel benchmark promises -- ao2mo 539 -> 359 s,
`vvvv` 195.8 s at 36.2 TFLOP/s against 299.2 at 23.7, `oovv` 148.9 s at 34.4
against 223.8 at 22.9 -- and then `Calculate d_prime`, the Z-vector/CPHF
solve, ran for over 60 minutes against 53 s in every other mode, with the GPU
at 88 % utilization drawing 193 W of a ~1 kW part. That is a flood of tiny
emulated GEMMs whose fixed-point decomposition costs far more than the
arithmetic; the run was cancelled. The stage-3 table in `docs/RESULTS.md`
already shows eager losing to native at b = 600, and the MO-space GEMMs in the
Z-vector solve are very much smaller than that.

**Ceiling, and why it was not taken.** Scoped to the ao2mo tile GEMM only,
with everything else left native, the gain is the 180 s eager saves there:
839 -> 659 s, **1.27x**. That is a code change rather than an environment
change -- `cublasSetEmulationStrategy(handle, strategy)` is present in
`libcublas.so.13` but cupy does not wrap it, so it would have to be called
through ctypes on cupy's handle and toggled around `gpu_ao2mo_blocks`.
1.27x on one cell does not pay for migrating the production environment to
CUDA 13 plus a ctypes hook into a vendor library. Not done.

Two findings from the exercise that outlive it:

- Do **not** set `CUPY_ACCELERATORS=cutensor`. It routes cupy's elementwise
  ufuncs through cuTENSOR, and cuTENSOR 2.8.1 fails the strided
  `W[ma:mb] += scratch[:k]` accumulate in `gpu_pairing_force.pairing_strip`
  with `CUTENSOR_STATUS_NOT_SUPPORTED`. Only `LD_LIBRARY_PATH` needs to point
  at a libcutensor, which is all gpu4pyscf checks when picking its
  `contract()` engine -- and is the state the project venv is already in.
- The two gpu4pyscf `blksize` patches are no longer needed for this workload:
  the 2-RDM exchange goes through `gpu_fft_k.get_k_lowrank` (section 1) and
  the pairing-K force through the low-rank FFT path (section 4), so neither
  `fft_jk.get_k_kpts` nor `aft_jk.get_ek_ip1` is on the NV63 hot path. The
  unpatched pip wheel ran the whole force correctly.

Harness and full record (job ids 27156218, 27159994/5, 27225069, 27232931):
`work/pprpa/NV63/gpu_dzvp_ke600/bench_cu13/` on bouchet --
`bench_force_all.sbatch` runs all environments in one job, `cu13_env.sh`
selects one, `probe_and_eager.sbatch` reruns the shape probe, `summarize.py`
rebuilds the table, `RESULTS.md` holds the write-up.

### 10. The FFT chain and the tile bookkeeping

**Measurement first.** `GPU_AO2MO_PROFILE=1` synchronises after every phase.
After sections 7-8 a vvvv block on NV63 was 102.5 s: GEMM 82.6 s (33.4
TFLOP/s -- the card's bare fp64 rate), FFT chain 15.6 s, codensity rebuilds
2.4 s, scatter 1.8 s. So the GEMM was done; the FFT chain was the remaining
overhead at 15 %.

**Fixes.**

- One fused gather-multiply kernel builds each codensity strip (two reads,
  one write, no gathered temporaries) instead of `take` + in-place multiply:
  codensity rebuild 5 -> 1.8 s per block.
- The Coulomb chain is real-to-complex: `rfftn`, an in-place product with the
  half-mesh kernel, `irfftn` straight back to real -- half the spectral
  traffic, no complex cast, no `.real` copy. FFT phase 15.6 -> 9.0 s.
- FFT sub-batches have one fixed shape per block (the last is zero-padded).
  The compact pair rows have variable length and had been giving every
  remainder a new cuFFT plan.
- Strip pairs that straddle the diagonal are split into sub-tiles so most of
  the unused upper triangle is never computed: GEMM flops down 4 %.

**A subtlety worth recording.** The naive R2C chain was 1e-8 off on the
20^3 diamond test mesh while matching to 2e-16 on 107^3. The Coulomb kernel
`w(G) = 4 pi / |G|^2` is not even under `G -> -G` on the Nyquist planes of an
*even* mesh dimension in a non-orthogonal cell, because `-G` is not
representable there and the fftfreq index aliases it. pyscf's convention
`ifft(fft(rho) * w).real` therefore silently discards a purely imaginary
contribution -- it only ever sees the even part of `w`. The C2R transform
assumes a Hermitian spectrum, so it must be fed the symmetrised kernel
`(w(G) + w(-G)) / 2` (`pair_layout.symmetric_half_kernel`); with that it
reproduces the C2C convention to rounding on every mesh. The production
meshes (107, 151, 159) are odd and were never affected.

**Result.** NV63 vvvv block 91 s: 86 % GEMM at the fp64 rate, 9 s FFT, 2 s
codensity, 2 s scatter. All three blocks 163 s resident, 178 s host-staged.
Cumulative from the 9 September kernel on this cell: 2281 -> 163 s, 14x,
exact throughout (<= 6e-14 relative against the previous kernel at every
step, and the module's own check against pyscf unchanged at 1.5e-13).

### 11. One shared planner was the wrong abstraction

The three batched-FFT kernels -- ao2mo strips, the low-rank exchange
(section 1), the pairing force (section 4) -- shared one idea of a planner:
size the FFT batch from a hard-coded 64 B per point, derive every other block
from it, cap it at the cuFFT limit. What they actually share is the chain
itself. `lib_pprpa/gpu_coulomb.py` now owns the fused codensity kernel, the
R2C potential chain with the symmetrised kernel, the per-point costs of one
FFT batch (56 B C2C, 40 B R2C including the cuFFT work area) and the batch
planner; each kernel plans the block it *contracts* from its own costs.

**Exchange** (`gpu_fft_k`, rewritten). The old planner pinned the AO row block
at 1-2 on 159^3 whatever the VRAM, so the final contraction ran as 2795
per-row GEMVs each re-reading the 90 GB AO grid. Now `fft_blk` (codensities
per transform batch, whole rows or balanced within-row chunks) and `row_blk`
(rows of the accumulator `U`) are planned separately and the final
contraction is a real GEMM. More importantly, the relaxed density only ever
uses `mo_coeff.T @ K @ orbp` with `orbp` the 600 active orbitals, so the full
AO-basis `K` was never needed: with `ket=orbp` the transforms run over the
(rank, active-MO) pairs -- 300 x 600 instead of 300 x 2795 at NV216 -- and
the AO grid is never held (it is evaluated in chunks for the last
contraction). NV63: 123 -> 42 s. NV216: 29 min -> 1.9 min.

The exchange agrees with the old kernel to 1e-11, not 1e-14: moving the
Coulomb operator from the bra to the ket codensity is exact in exact
arithmetic (W is symmetric on an odd mesh) but changes the rounding of heavily
cancelling sums over 3.7e8 terms. Both sit within 1e-9 of dense
`fft_jk.get_k` in the tests.

**Pairing force.** Fused codensity and the R2C chain, with the strip planner
taking the chain's cost from `gpu_coulomb`: NV63 32.4 -> 17.8 s, 1e-15.

### 12. The Davidson ERI stream

When the three ERI blocks do not fit one GPU the Davidson streams them from
the host on every matrix-vector product: 194 GB per MVP at NV216, from
pageable memory, and `vvvv` / `oooo` were read as strided column slices that
numpy copied before every upload. Measured on NV63 (78.7 GB per MVP, 32
trial vectors, two B200s):

| mode | MVP | rate |
|---|---|---|
| resident (fits on NV63) | 0.026 s | -- |
| split: row halves on 2 GPUs, partials summed on the host | 0.026 s | trial vectors only |
| tiled, pageable host (before) | 3.71 s (first MVP 19.8 s) | 8.6 GB/s |
| tiled, pinned host | 1.39 s | 56.6 GB/s |

Three changes. `gpu_ao2mo` allocates host-staged finals with
`cupyx.empty_pinned` (`GPU_AO2MO_PINNED=0` opts out), which the Davidson
inherits through the copy-free reshape: 6.6x on the stream, not the 2-2.5x
guessed beforehand. The physicist `vvvv` / `oooo` matrices are symmetric, so
the tiled path streams contiguous row strips. And `mode="split"` keeps a row
range of every block resident on each `DeviceGroup` slot, so the 2-GPU jobs
stop streaming ERIs at all: the NV216 Davidson (17 iterations) went from
4 min to 2.75 s, the 194 GB upload taking 1.7 s. All modes agree to 1.5e-16.

### 13. The CPHF response

After sections 7-12 the relaxed density was the largest phase, and inside it
the CPHF solve: 15 min at NV216, on one GPU. Each of its response calls
re-evaluated the GGA AOs and their gradients on the whole grid, rebuilt the
*ground-state* density and XC kernel from scratch, ran a separate FFTDF
`get_j` that evaluates the full 90 GB AO grid at once with two more dense
contractions, and used a single card.

`lib_pprpa/grad/gpu_response.py` caches the ground-state kernel per slot,
folds the Coulomb potential of the perturbing density into the XC weights so
J + fxc come out of one GEMM, and splits the grid over the device group. Per
call on NV63: 0.93 -> 0.52 s (one slot) -> 0.27 s (two), 1.2e-12 against the
dense response. At NV216 the solver makes 68 calls; they went from ~13 s to
4.49 s each and the CPHF phase from 15.0 to 5.1 min. Per-slot state lives
under `resp_*` keys -- the first version stored its chunk size as `gchunk` and
the exchange build, running on the same group in between, freed it.

### 14. The regression that was not, and the validation runs

At the exactly-C3v `opt/input.vasp` the 18 September 2-GPU run gave
|F| = 5.06e-02 a.u. against the validated 3.0875e-02 (section 5's jobs), and
the optimisation built on it was set aside. Two things settled it on
22 September:

- Job 27006823, the old tree (7f45fc7) on 1 and 2 GPUs, reproduces the
  reference: max |dF| 5.5e-9 and 2.1e-8 eV/A, C3v residual 1e-8 and 3e-9.
  The failure was transient -- library files had been edited under the
  running job -- not a code defect.
- The new tree on two devel B200s reproduces it as well, twice: job 27138544
  (49.6 min; before section 13) and job 27147836 (39.4 min; complete) with
  |F| = 3.08748e-02, max|F| = 1.02903e-02, max |dF| = 2.1e-8 and 5.1e-9 eV/A,
  E_state identical to 1e-8 Ha. The gpu_devel QOS's 256 GB host cap held with
  the 194 GB pinned ERI set.

Every stage was checked element-wise against the kernel before it (the
reference tensors are kept once on scratch,
`/nfs/roberts/scratch/pi_tz324/sc3352/ao2mo_refs/`, and later runs compare
against them instead of recomputing an earlier kernel); the per-stage numbers
and job ids are in `docs/RESULTS.md`, and `benchmarks/bench_gemm.sbatch`
reruns any of them.

### 15. The hcore derivative in density space

After section 13 the hcore derivative was the largest phase: 11.5 min of the
39.4 min force, 215 calls of gpu4pyscf's `hcore_generator`, one per atom.  Each
call built the full AO matrix of the local-PP derivative

    vloc_R^x(g) = ifft(i G_x . SI_A(G) . vlocG_A(G))        3 FFTs
    hcore^x_ij  = sum_g ao_i(g) ao_j(g) vloc_R^x(g)         3 GEMMs, nao^2 x ngrid

and `_hcore_force` then traced it against the density T.  That is
`natm * 3 * nao^2 * ngrid` flops -- plus a fresh evaluation of the AO values on
the whole grid inside every one of the 215 calls, because `block_loop` starts
over each time.

The matrix is never needed.  The AO indices are contracted on both sides by the
same grid point, so the trace collapses:

    Tr[hcore^x_A T] = sum_g vloc_R^x_A(g) rho_T(g),
    rho_T(g)        = sum_ij ao_i(g) T_ij ao_j(g)

and since pyscf's `ifft` carries the 1/N while `fft` does not, and rho_T is
real, Parseval turns the grid sum into a reduction over G with no FFT per atom:

    F^x_A = (1/N) Re sum_G [i G_x SI_A(G) w_A(G)] conj(rho_T(G))

    w_A(G) = vlocG_A(G)     (pseudo-potential)  |  Z_A coulG(G)  (all-electron)

    natm * 3 * nao^2 * ngrid   ->   nao^2 * ngrid + natm * 3 * ngrid

**gpu4pyscf already has this reduction.**  `multigrid.eval_vpplocG_SI_gradient`
(and `eval_nucG_SI_gradient` all-electron) is exactly the expression above --

    dSI_prefactor = -1j * Gv.T * rho_g.conj()
    de[ia, :] = dSI_prefactor @ vlocG            # then .real / cell.vol

-- and `krhf.grad_elec` uses it on its `multigrid_v2` branch.  It is the default
`KNumInt` branch that falls through to the per-atom `hcore_generator` loop.  The
first version of this work re-derived the reduction (structure factors, atom
batching by free VRAM, a device-group grid split: 340 lines); the version that
landed is 109, because it only supplies the wiring:

1. rho_T(G) from the uniform-grid `KNumInt.get_rho`, un-sorted back out of
   `block_loop`'s grid order and transformed once.  Normalisation follows
   `multigrid_v2.evaluate_density_on_g_mesh`: the grid weight `vol/ngrids/nkpts`
   rides along in rho(G) and the SI-gradient helper divides it back out.
2. `eval_vpplocG_SI_gradient` / `eval_nucG_SI_gradient` for the reduction.  Its
   separable `SIx/SIy/SIz` structure factors are lighter than the `(natm, ngrid)`
   array the generator holds (13.8 GB at NV216) *and* than the batched
   `(nbatch, ngrid)` blocks the first version used, so the atom batching and its
   OOM retry could go.
3. `contract_h1e_dm(..., hermi=0)` for the AO-derivative half -- `int1e_ipkin`
   plus the local PP acting on the moved basis functions, one array with an
   `nao_A x nao` slice per atom.  `hermi=0` evaluates both of the generator's
   `-=` lines rather than doubling the first, so a non-symmetric T stays exact.

`lib_pprpa/grad/gpu_hcore_force.py`; `PPRPA_HCORE=dense` restores the per-atom
generator.  Everything here is single-device, which is what gpu4pyscf's kernels
are; the `DeviceGroup` the force assembly passes is recorded in the telemetry
and otherwise unused.

Measured on **one B200** (the card the rest of this document's numbers use),
inside a gpu_devel allocation:

| | NV63, ke=600 | NV216, ke=300 |
|---|---|---|
| per-atom generator | 52.7 s (0.84 s/atom) | 6.10 s/atom -> 1311 s = 21.8 min (8 atoms sampled) |
| density space | **6.0 s** | **17.8 s** |
| breakdown | rho 0.9, G-space 0.1, AO-deriv 5.0 | rho 2.8, G-space 0.1, AO-deriv 14.9 |
| agreement | 1.67e-14 (**6.33e-15 relative**) | 4.27e-14 (9.28e-14 relative, sampled) |
| speedup | 8.8x | **74x** |

The extrapolated 21.8 min for the per-atom loop at NV216 lines up with the
24 min recorded for one GPU in the table below, which is the cross-check that
the sampled timing is honest.

The AO-derivative term (`krhf.get_hcore`) is 14.9 s of the 17.8 -- 84% of what
is left -- but 15 s inside a 28-minute force is not worth chasing.  An earlier
version split its `deriv=1` grid pass over the device group; at 15 s that buys
~7 s for ~70 lines, so it was dropped.  (The same measurements on an RTX PRO
6000 Blackwell, whose fp64 rate is ~1/64 of a B200's, read 786.1 s -> 21.8 s at
NV63 and 189.6 s at NV216 -- job 27160009 / 27225292 -- which is why the
slow-card numbers are not the ones quoted here.)

The algebra was also checked against pyscf's *own CPU* `hcore_generator` to
1.8e-15 relative, for a symmetric and a non-symmetric T (note the CPU generator
returns `[3, nkpts, nao, nao]` where gpu4pyscf returns `[nkpts, 3, nao, nao]`).

### 16. SCF checkpointing across geometries

The same cell is run many times at nearby geometries: every optimization step,
and every one of the 200+ phonopy displacements, each one atom away from the
*same* optimized geometry.  None of the gradient can be carried across those
runs -- the relaxed density changes everywhere when any atom moves, and the
per-atom hcore derivative matrices would be 40 TB at NV216 even if it could --
but the SCF's *starting point* can.

`lib_pprpa/scf_chk.py` writes the converged AO density next to the geometry it
was converged at, and `run_scf` hands it back as `dm0` when the next cell
matches in the ways an AO density matrix depends on (same atoms in the same
order, same lattice, basis, pseudo, charge, spin); a different geometry is the
point, so the displacement is reported, not rejected.  A missing, unreadable or
mismatched checkpoint logs a line and falls back to the default guess -- it
never raises -- and an unconverged SCF is never written.

* `PPRPA_SCF_CHK` -- read and write, a rolling checkpoint (optimization)
* `PPRPA_SCF_CHK_IN` -- read only (every displacement reads the optimized
  geometry's checkpoint; none of them write, so there is no race)
* `PPRPA_SCF_CHK_OUT` -- write only

This changes only the path the SCF takes, never where it lands.  The one thing
a guess can change is *which* self-consistent solution is reached when a system
has more than one -- a charged defect can -- and there reuse is the
conservative choice: every displacement starts from the reference state instead
of rediscovering one.

## Where the time goes now

| phase | reference (old code) | 1 GPU (Sep 11) | 2 GPUs (Sep 11) | 2 GPUs (Sep 22, sections 7-13) |
|---|---|---|---|---|
| SCF | 4 min | 4 min (14 on a shared node) | 4 min | 4.0 min |
| ao2mo | 5 h 14 min | 6 h 35 min (cap regression; ~4.5 h after section 6) | 3 h 16 min | 14.6 min |
| Davidson | 4 min | 6 min | 4 min | 3 s |
| relaxed density | 44 min | 46 min | 30 min | 7.2 min (K 1.9, CPHF 5.1) |
| hcore | 23 min | 24 min | 11 min | 11.5 min |
| pairing-K | 16 h 21 min | 2.8 min | 1.5 min | 0.8 min |
| Vxc/fxc, J, overlap, PP | 2 min | 1 min | 1 min | 0.8 min |
| total | 22 h 53 min | 8 h 09 min | 4 h 09 min | **39.4 min** |

The hcore derivative was the largest phase in that last column -- 215 calls of
gpu4pyscf's `hcore_generator`, one per atom.  Section 15 takes it to 17.8 s on
a single B200, so the 11.5 min row becomes ~0.3 min and the force should land
near **28 min**; that end-to-end run has not been repeated yet, the 17.8 s is a
measurement of the phase on its own.

For the 63-atom cell at ke = 600 Ha (nao 819, mesh 151^3, AS 300) the same
code gives a full force in 1 h 47 min on a B200 and 3 h 02 min on an H100
(medians over 34 and 13 production tasks).

## Appendix A: the tiled ao2mo and ERI contraction, in matrix form

Section 0 disposes of the tiled ERI path in one line. It was the phase that
dominated the force when this appendix was written (11 September); the Gram
factorisation in A.1 and the write in A.3 still describe the kernel, while
the strip planner of A.2, the fixed 65 PFLOP of A.4 and the streaming of A.5
are superseded by sections 7, 8, 10 and 12.

### A.1 The chemist ERI is a Gram matrix in the Coulomb metric

Let `Ng` be the number of grid points and `Phi` the matrix of active MOs sampled
on the uniform grid,

    Phi in R^{n x Ng},   Phi[p, g] = phi_p(r_g)

and for a pair index `P = (p, q)` let `rho_P` be the codensity row

    rho[P, g] = Phi[p, g] * Phi[q, g],    rho in R^{npair x Ng},  npair = nA*nB

Define the discrete Coulomb operator on the grid,

    W = (vol/Ng) * F^-1 diag(coulG) F,    W in R^{Ng x Ng},   W = W^T

(`F` the 3D DFT on the mesh). Then the chemist-convention MO ERI is exactly a
Gram matrix of the codensities in that metric:

    E = rho W rho^T,      E[(pq),(rs)] = (pq|rs)

This is the whole algorithm. Everything else is a consequence.

A Gram matrix block-factorises with no approximation: split the pair index into
strips `I`, `J` of width `b` and

    E[I, J] = (rho_I W) rho_J^T = vR_I rho_J^T,   vR_I := rho_I W  in R^{b x Ng}

so a tile of `E` needs two strips of `rho` and never the whole thing. That
matters because `rho` is enormous and `E` is not:

| array | nv = 300, mesh 159^3 (Ng = 4,019,679) |
|---|---|
| `rho`, all 90,000 pairs | 2.9 TB |
| `E`, 90,000^2 float64 | 64.8 GB |
| one strip `rho_I` at b = 600 | 19.3 GB |

`_strip_task` is that factorisation literally: `vR = ifft(fft(rho_p)*wcoulG)`
applies `W` to the outer strip, and `vR.dot(rho_q.T)` is `E[I, J]`. The inner
strip is regenerated on the fly (`_codensity_pairs`) rather than stored.

Note where the FFT sits: on the **outer** strip only. Each pair is transformed
exactly twice (forward, inverse) per block whatever `b` is, while the
`O(npair^2)` work is pure DGEMM. Section A.4 says what that buys.

### A.2 Choosing the strip width

Per pair-gridpoint the kernel holds `rho_p` (8 B) + its complex transform (16) +
`vG` (16) + `vR` (16) + `rho_q` (8) = 64 B, plus the `b x b` output tile:

    8*b^2 + 64*Ng*b <= budget,   budget = free - max(cushion, 0.06*free)

which `_estimate_pair_blk` solves in closed form. Three corrections follow:

    b <- min(b, floor(cufft_limit / Ng))     cuFFT plan cap (section 3)
    b <- nB * floor(b / nB)                  whole second-MO rows (A.3)
    b <- min(b, npair)

Worked for the 216-atom vvvv block on a B200 with the final tensor host-staged
(`free ~ 180 GB`, `budget ~ 169 GB`, `64*Ng = 2.57e8` B per pair):

    quadratic  ->  b = 656          (the 8*b^2 term is ~3 MB here, negligible)
    cuFFT      ->  min(656, 801)    relaxed cap 1.5*2^31 / Ng = 801
    row align  ->  300 * floor(656/300) = 600

With section 6's strict cap instead, `2^31/Ng = 534` and `300*floor(534/300) =
300`. The whole 0.7 h-per-block regression is that one `floor`: 534 and 656
differ by 23 %, but after rounding down to whole MO rows they differ by 2x.

### A.3 Chemist to physicist costs nothing

A tile of `E` carries chemist indices `(a q | b s)`, with `a, b` from `moA` and
`q, s` from `moB`. The physicist tensor pp-RPA wants is

    out[a, b, q, s] = <ab|qs> = (aq|bs) = E[(a,q), (b,s)]

which is exactly an `(A, nB, B, nB) -> (A, B, nB, nB)` axis swap of the tile —
the `reshape(...).transpose(0, 2, 1, 3)` in `_strip_task`. Because strips are
aligned to whole `nB`-rows, the strip bounds are multiples of `nB`, so a tile
spans complete outer-MO ranges `[a0, a1) x [b0, b1)` and its result lands in one
contiguous rectangle `out[a0:a1, b0:b1]`. No gather, no separate reorder pass,
no second copy of the tensor.

Two properties fall out of writing rectangles rather than accumulating:

- **Idempotent retry.** Every tile is an assignment to a disjoint region, so
  redoing a strip after an OOM is safe and cannot corrupt a partial sum. That is
  what lets `_make_strip_shrink` halve one slot's `sub_blk` and re-run only the
  failing task (section 6) instead of restarting the tensor.
- **Multi-GPU safety with no coordination.** Outer strips are the dispatched
  tasks, so two slots can never touch the same elements and `DeviceGroup` needs
  no reduction step — only the shared task iterator.

### A.4 What the cost model predicts

Per block, with `no = nv = 300` so all three blocks have `npair = 90,000`:

| term | count | fp64 flops |
|---|---|---|
| GEMM `vR_I rho_J^T` | `nblk^2` tiles | `2*npair^2*Ng` = **65.1 PFLOP** |
| FFT | `2*npair` transforms of 159^3 | ~0.08 PFLOP (**0.12 %**) |
| codensity rebuild | `nblk*npair*Ng` products | negligible |

The transform — the part that sounds expensive — is a tenth of a percent. The
Gram GEMM is everything, and its flop count does **not** depend on `b`.

HBM traffic does. Each tile reads `vR_I` and `rho_J`, so

    bytes = 16 * Ng * npair * nblk = 16 * Ng * npair^2 / b

868 TB at `b = 600` and 1736 TB at `b = 300`, about 108 s and 217 s at ~8 TB/s.
Against block times of 1.5 h and 2.2 h that is 2-3 % either way, so the strip
width is **not** a bandwidth story. It is a GEMM-shape story: the tile GEMM has
`M = N = b` against `K = Ng ~ 4e6`, an extremely tall-skinny reduction whose
efficiency falls off as `b` shrinks. Sustained rates from the production logs:

| b | block wall | sustained fp64 |
|---|---|---|
| 600 | 1 h 30 min | 12.1 TFLOP/s |
| 300 | 2 h 12 min | 8.2 TFLOP/s |

Those two rates reproduce every ao2mo timing in this note:

| run | strip widths | predicted | measured |
|---|---|---|---|
| reference (job 25686365) | 600, 600, 300 | 5 h 12 min | 5 h 14 min |
| strict-cap regression | 300, 300, 300 | 6 h 36 min | 6 h 35 min |
| after section 6, 1 GPU | 600, 600, 600 | 4 h 30 min | (expected) |
| 2 GPUs at b = 300 | 300, 300, 300 | 3 h 18 min | 3 h 16 min |

The last row is also the cleanest evidence that the dispatch is sound: 150 to
300 independent outer strips per block is far more than two slots need for
balance, and the measured speedup is 2.01x.

### A.5 The tiled Davidson contraction

The same host-staging rule (`gpu_mem.fits_resident`, 75 % of VRAM) then governs
the eigensolve. At `no = nv = 300` the three tensors are 194.4 GB against 137 GB
of budget on a B200, so `_choose_mode` picks `tiled` on every current device.

Flatten the physicist blocks to matrices over pair indices,

    V = vvvv -> R^{nv^2 x nv^2},   O = oooo -> R^{no^2 x no^2},
    Wov = oovv -> R^{no^2 x nv^2}

and for a trial vector let `Z_oo`, `Z_vv` be the unpacked triangles, with

    x_oo = vec(Z_oo^T),   x_vv = vec(Z_vv^T)

The ERI part of the pp-RPA matrix-vector product is then one symmetric 2x2 block
operator:

    [ y_vv ]   [ V     Wov^T ] [ x_vv ]
    [      ] = [             ] [      ]
    [ y_oo ]   [ Wov   O     ] [ x_oo ]

`eri_mvp_tiled` evaluates this transposed, with all `ntri` trial vectors as rows
of `X`, i.e. `Y = X K^T` — which is why one sweep of the ERI serves the entire
Davidson block. Around it, `_prepare_z` and `_finish_mv` apply the packing:

    scatter t -> Z (tril, k = 0 singlet / -1 triplet);  diag(Z) *= 1/sqrt(2)
    y = K vec(Z^T)  ->  Y = unvec(y)
    Y <- Y + Y^T  (singlet)   or   Y - Y^T  (triplet)
    m = tril(Y^T with diag *= 1/sqrt(2))
    m += (sigma*(eps_p + eps_q) - 2*mu) . t,    sigma = -1 on the hh block

Tiling is a column-block decomposition of each `K` block: for a strip `P` of
columns,

    y_vv += X_vv[:, P] V[:, P]^T        (one strip of V on device at a time)

so device residency drops from `O(n^4)` to `ntri*n^2` accumulators plus one
`n^2 x tile` strip, while the host keeps the full tensors. `estimate_eri_tile`
sizes the strip as `budget / (8*n^2)`; once ao2mo has released the MO grids
there is enough free VRAM that the tile saturates at `n^2` and each block is
uploaded in a single transfer.

The cost balance inverts relative to ao2mo. Per matrix-vector product:

    transferred:  2.43e10 elements = 194.4 GB   (one pass per block)
    flops:        2 * ntri * 2.43e10
    intensity:    ntri / 4 flops per byte

At `ntri` of order 100 that is ~25 flops per byte, against a host link two
orders of magnitude slower than HBM: the tiled MVP is **transfer-bound**. The
only lever is `ntri`, because the transfer is per Davidson *iteration*, not per
trial vector, so a wide trial block is nearly free. That is why the Davidson
stays at 4-6 min while streaming ~200 GB per iteration.

### A.6 One fixed, one still on the table

Both follow from the algebra above.

1. ~~**`oovv` is streamed twice per MVP.**~~ **Fixed in 38ab6f5.** The third loop
   of `eri_mvp_tiled` read row strips for `y_vv += X_oo[:, P] Wov[P, :]` and the
   fourth re-read *column* strips for `y_oo`. The same row strip already gives
   the other product, as a disjoint column block:

       y_oo[:, P] = X_vv Wov[P, :]^T

   so one pass now serves both, cutting per-MVP traffic from 259.2 GB to
   194.4 GB (-25 %). Because the old kernel was correct, only wasteful, the
   guard (`test_eri_mvp_tiled_streams_each_block_once`) counts uploaded bytes
   rather than checking values. Validated on a B200, job 26947087.

2. **Column strips of `V` and `O` force a host-side gather.** Both are stored
   C-contiguous, so `vvvv[:, p0:p1]` is strided and numpy must copy it before
   the transfer. They are symmetric — `<ab|cd> = <cd|ab>` for real Gamma-point
   orbitals, and the reshaped matrix inherits it, `V = V^T` — so `V[:, P]^T` can
   be replaced by the contiguous row strip `V[P, :]`: same math, no gather.
   (Equal to the ~1e-15 asymmetry the independent FFTs leave in `E`, not
   bitwise.) This only bites when `tile < n^2`, i.e. on smaller-VRAM devices; at
   full tile the slice is the whole array and is already contiguous.

### A.7 The MO grid: the last untiled step, now tiled

`_mo_on_grid` held the whole (ngrid, nao) AO array in one `eval_ao_kpts` call --
89.9 GB at nao = 2795 on 159^3. It is the only term in ao2mo that grows as
natoms^2 (every other term is 8*nmo*Ng or 64*b*Ng, all linear in Ng), so it is
the term that decides how large a cell the transform can reach. `_mos_on_grid`
now evaluates the AOs in grid chunks:

> **Φ**[:, g₀:g₁] = **C**ᵀ **χ**[g₀:g₁, :]ᵀ

The grid index is free in both operations, and an output element sums over `nao`
only, so no reduction crosses a chunk boundary: the tiling is exact, not
approximate. One AO block also feeds every coefficient set, so the occupied and
virtual grids now share a single AO evaluation instead of repeating it.

Measured on the real 215-atom cell on two B200s (job 26655445; SCF skipped,
the grid build depends only on the shape of the coefficients):

| variant | wall | peak VRAM | AO block |
|---|---|---|---|
| untiled, 1 device | 2.7 s | 109.3 GB | 89.9 GB |
| tiled, 1 slot | 1.0 s | 53.6 GB | 34.5 GB |
| tiled, 2 slots, replicated (default) | 1.4 s | 53.7 / 53.7 GB | 34.5 GB |
| tiled, 2 slots, chunks dispatched | 21.1 s | 38.3 / 38.3 GB | 34.5 GB |

Agreement with the untiled grid is 5e-15 relative.

The last row is the interesting one. Splitting the chunks across slots halves
the AO evaluation, but the grids have to reach *every* slot, so the result is
staged on the host and broadcast back: 19.3 GB down and 19.3 GB up per slot,
which costs far more than the 0.5 s it saves. The default is therefore
**replicated** -- each slot chunks the whole grid into its own device arrays,
concurrently -- which needs no host staging and no peer copies at all, and
replaces the old build-on-device-0-then-broadcast. Dispatching is kept behind
`GPU_AO2MO_GRID_DISPATCH=1` for the regime where the AO evaluation actually
dominates.

The next ceiling is not this one: `_estimate_pair_blk` floors the strip at
`b = nB`, so the strip loop cannot go below `64*n*Ng` (77 GB on NV216).

### A.8 A negative result: `gpu_fft_k.ao_on_grid` does not need the same fix

`ao_on_grid` ends in `cp.ascontiguousarray(cp.asarray(ao).T)`, which looks like
it must hold two full (nao, ngrid) arrays — 180 GB at NV216, more than a B200
has. It does not. `eval_ao_kpts` returns an **F-contiguous** (ngrid, nao) array,
i.e. a transposed view of a C-contiguous (nao, ngrid) buffer, so `.T` is already
C-contiguous and `ascontiguousarray` returns the *same device pointer*. Measured
directly (job 26660756): the copy allocates 0.00 GB at both the small-cell and
NV216 scales, and the peak is one array, 89.98 GB.

Chunking it the way `_mos_on_grid` was chunked is therefore a regression — the
chunked form must allocate the output *and* a block:

| `ao_on_grid` at NV216 | wall | peak |
|---|---|---|
| `ascontiguousarray(ao.T)` (current) | 0.63 s | **90.0 GB** |
| grid-chunked | 0.78 s | 110.1 GB |

Both are bit-identical; the chunked one is 1.2x slower and 20 GB worse. Not
adopted.

The block planner at that point (rank = 300, AO grid + phiL/phiR resident):

    free at plan time = 81.6 GB  ->  n_el = 269   (vs rank = 300)
    _plan_blocks      ->  row_blk = 1, rank_blk = 269, 2 rank chunks

so row batching is off, as `row_blk = ... if rank_blk == rank else 1` requires.
Splitting the AO axis across two devices would free ~44.9 GB and raise `n_el` to
418: enough to make `rank_blk = rank` (one rank chunk of 300 instead of 269+31),
but **not** enough to turn row batching on, which needs `n_el >= 2*rank = 600`.
There is a second, harder ceiling behind that one. `_plan_blocks` also clamps
`n_el` to the cuFFT batched-plan cap, which on 159^3 is `1.5*2^31/Ng = 801`
elements. So `row_blk` tops out at `801 // 300 = 2` **however much VRAM the
device has** — above roughly 250 GB free, memory stops being the binding
constraint at all. Row batching is not a prize worth crossing the network for.

The AO-axis split therefore buys better-shaped FFT batches, not row batching,
and it costs an all-reduce per density pair plus shipping `vR_dm` once per AO
row. It was not pursued. The same batch shaping is now had for free by
**balancing the rank chunks**: ceil-dividing 300 by a cap of 269 leaves 269+31,
while spreading the same two chunks evenly gives 150+150. Because
`ceil(r/ceil(r/c)) <= c`, the balanced chunk never exceeds the memory cap and
the chunk count never changes, so it costs nothing; and when the whole rank fits
(`cap >= rank`) it reduces to `rank_blk = rank` exactly as before, leaving the
row-batching path untouched. Measured on the shared checkout: the NV216 planner
now reports `row_blk=1 rank_blk=150 chunks=2`.

### A.9 Knobs

| variable | effect |
|---|---|
| `GPU_AO2MO_MAX_PAIR_BLK` | upper bound on the ao2mo strip width `b` |
| `GPU_AO2MO_GRID_CHUNK` | grid points per AO chunk in `_mos_on_grid` |
| `GPU_AO2MO_GRID_DISPATCH` | `1` splits grid chunks across slots (host-staged) |
| `PPRPA_ERI_MODE` | force `resident` / `tiled` / `auto` |
| `DAVIDSON_ERI_TILE` | force the Davidson column-strip width |
| `LIB_PPRPA_GPUS` | slot count for `DeviceGroup` (section 5) |

## Hardware and scheduler findings

- **RTX PRO 6000 Blackwell is unsuitable**: median 19 h 27 min per 63-atom
  force, 10.8x a B200, at 100 % utilisation but ~213 W. It is a workstation
  part with a heavily reduced float64 rate; this workload is all float64.
- **The gpu_devel A40 node** lacks the compute capability of the gpu4pyscf
  build (every kernel fails with "named symbol not found"). Pin devel tests
  with `--gres=gpu:b200:1`.
- **gpu_h200** is good hardware but had ~500 queued jobs; a 2-day request
  waited more than three days. gpu_b200 gave the best run time and a
  competitive wait.
- Slurm's start-time estimate assumes every running job uses its full time
  limit, so it is an upper bound that drifts earlier; only jobs the backfill
  pass reaches get one. Priority here is fairshare-dominated (weight 100000
  versus 10000 for age), and shorter time limits did not measurably reduce
  waits on these partitions.

## The gpu4pyscf fork

The checkout at `/nfs/roberts/project/pi_tz324/sc3352/gpu4pyscf` is
`v1.7.1-30-g8bc3264`, installed editable with in-tree compiled libraries, and
differs from upstream in two files (`pbc/df/fft_jk.py`, `pbc/df/aft_jk.py`:
adaptive block sizes plus telemetry prints). Neither patched function is
called on the current production path; their entry prints count zero in every
recent log. They are still needed for hybrid functionals at NV scale (the
stock `get_k_kpts` block of 32 rows would need 1.4 TB at nao 819, 151^3), for
`PPRPA_PAIRING_K=aft`, and for the AO-direct Davidson example. Reverting the
two files in place is safe for PBE work; replacing the checkout with a pip
release is a separate change with API-drift risk in the gradient internals.
`scripts/compare_pprpa_cpu_gpu_forces.py` calls a telemetry symbol that
exists only in the patched file.

## Files

Added 22 September (sections 15-16): `lib_pprpa/grad/gpu_hcore_force.py`
(density-space hcore force), `lib_pprpa/scf_chk.py` (SCF checkpoint reuse),
`benchmarks/bench_hcore_force.py`; wired into `lib_pprpa/grad/pprpa_gamma_gpu.py`
and both `examples/gpu/pbc_pprpa_gamma_opt*.py`.  Tests:
`tests/test_gpu_hcore_force.py`, `tests/test_scf_chk.py` (CPU only).

New modules: `lib_pprpa/gpu_multi.py`, `lib_pprpa/gpu_fft_k.py`,
`lib_pprpa/gpu_pairing_force.py`. Modified: `lib_pprpa/gpu_mem.py`,
`lib_pprpa/gpu_ao2mo.py`, `lib_pprpa/pprpa_eri_gpu.py`, `lib_pprpa/pprpa_util.py`,
`lib_pprpa/grad/pprpa.py`, `lib_pprpa/grad/pprpa_gamma_gpu.py`,
`examples/gpu/README.md`, `examples/gpu/pbc_pprpa_gamma_opt_nv_gpu.py`.

Tests (all need a GPU; two virtual slots on one GPU cover the multi-device
paths): `tests/test_gpu_multi.py`, `tests/test_gpu_fft_k_lowrank.py`,
`tests/test_gpu_pairing_force.py`, `tests/test_gpu_ao2mo_multi.py`,
`tests/test_gpu_grad_pairing_e2e.py`, `tests/test_fft_batch_cap.py`.

Slurm side (in the spectra project's `scripts/`): `pprpa_force_gpu2.sbatch`,
`pprpa_opt_gpu2.sbatch`, `submit_pprpa_opt_resumable.sh --gpus`, the fast-failure
breaker in `pprpa_opt_gpu.sbatch`, and `LIB_PPRPA_GPUS` handling in
`spectrum_common.sh`.

Added 22 September (sections 7-13): `lib_pprpa/pair_layout.py` (pair index,
symmetric tile scatter, diagonal split, symmetrised half kernel; numpy-only,
tested without a GPU), `lib_pprpa/gpu_coulomb.py` (fused codensity kernel,
R2C Coulomb chain, FFT batch planner), `lib_pprpa/grad/gpu_response.py` (the
CPHF response), `benchmarks/` (frozen copies of every superseded kernel, the
benchmark drivers and `bench_gemm.sbatch` with one stage per section --
*not tracked*, see the note at the top of `docs/RESULTS.md`); `gpu_fft_k.py` rewritten, `pprpa_eri_gpu.py` gains the split
mode. Tests: `test_pair_layout.py`, `test_gpu_response.py`, and the
exchange / Davidson tests extended for the ket path and the split mode.
