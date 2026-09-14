# GPU Gamma-point pp-RPA — requirements

This folder has two GPU optimization examples:

- **`pbc_pprpa_gamma_opt_gpu.py`** — small periodic cell, **AO-direct** energy path.
  Library-only (lib_pprpa + gpu4pyscf + ase). Runs in a few minutes on one GPU.
- **`pbc_pprpa_gamma_opt_nv_gpu.py`** — large defect cell (NV-center style), **MO-eri**
  energy path with a frozen-core active space. Tunable `ke_cutoff` on the command
  line. This is the path used for the 63-atom NV production runs.

Both share the same back end:

```
GPU KRKS SCF (gpu4pyscf)
  -> [energy solve: AO-direct OR MO-eri — see below]
  -> GPU Gamma-point pp-RPA gradient    (lib_pprpa.grad.pprpa_gamma_gpu)
  -> ASE BFGS                           (lib_pprpa.grad.ase_utils.kernel)
```

The energy solve differs:

```
AO-direct (small):  lib_pprpa.pprpa_davidson_gpu.attach_gpu_contraction(mp, kg)
MO-eri    (large):  lib_pprpa.gpu_ao2mo.gpu_ao2mo_blocks  -> active-space vvvv/oovv/oooo
                    lib_pprpa.pprpa_eri_gpu.attach_gpu_eri_contraction(mp, ...)
```

## Software required
- **gpu4pyscf** with working CUDA libraries (provides `gpu4pyscf.pbc.dft.KRKS`,
  `pbc.df.fft_jk.get_k`, AFTDF `get_k_e1`, and `pbc.grad.krhf` primitives).
- **pyscf** (the periodic `gto`/`dft` driver and `lib_pprpa`).
- **ase** (the BFGS optimizer used by `ase_utils.kernel`).
- **lib_pprpa** on `PYTHONPATH`.
- A GPU (the small example fits in any modern GPU; NV-scale needs ~32-80 GB).

## REQUIRED gpu4pyscf source patches (for large systems)
We patched two gpu4pyscf files so the periodic exchange kernels don't OOM at
large `nao` / fine mesh (e.g. the 63-atom NV cell, 819 AO, mesh 59^3). These are
**pure-Python edits — no rebuild needed**, but a `git checkout`/reinstall of
gpu4pyscf will wipe them, so re-apply from here. They are **NOT needed for the
small example in this folder**, only for production-scale cells.

### 1. `gpu4pyscf/pbc/df/fft_jk.py` — `get_k_kpts` block size (~line 163)
The hardcoded `blksize = 32` makes the `rho1/vR` intermediate
(`~ blksize * nao * ngrids * 16 B`) OOM. Replace:
```python
blksize = 32
```
with:
```python
from gpu4pyscf.lib.cupy_helper import get_avail_mem
blksize = max(1, min(32, int(get_avail_mem() * 0.2 / (nao * ngrids * 16))))
```

### 2. `gpu4pyscf/pbc/df/aft_jk.py` — `get_ek_ip1` block size (~line 522)
The K energy-gradient kernel (pp-RPA pairing-K and hybrid reference-K force)
under-budgets its `nao^2 * blk` arrays. Change the divisor factor `*2` -> `*8`:
```python
# from:
blksize = int(avail_mem/(nao**2*bvk_ncells*16*2))//16*16
# to:
blksize = int(avail_mem/(nao**2*bvk_ncells*16*8))//16*16
```

Keep `.orig_bak` backups next to each file; to revert, copy them back.

### Patch 1 is no longer enough at the 216-atom cell — use `get_k_lowrank`
Even with adaptive `blksize`, `fft_jk.get_k_kpts` keeps `ao` (nao x ngrid),
`ao_dms` and `vR_dm` (nset x nao x ngrid each) resident and FFTs one full
nao x ngrid codensity row at a time. For nao=2795, mesh 159^3 (216-atom NV cell,
ke=300 Ha) those are 90 / 180 / 180 GB plus a 270 GB per-row transient, so the
2-RDM exchange OOMs on a 183 GB B200 regardless of block size (job 25686365).

The pp-RPA densities are low rank (X_ao = C_a x C_a^T, Y_ao = C_i y C_i^T, rank
<= AS), so the drivers now attach `lib_pprpa.gpu_fft_k.attach_lowrank_getk(mf, cell)`;
`make_rdm1_relaxed_rhf_pprpa` uses `mf.get_k_lowrank` when present and builds
K from (AO, active-MO) codensities: only the AO grid is nao x ngrid, the FFT
batch is `row_blk * rank_blk * ngrid`, and there are ~nao/AS fewer FFTs.

### Optional gpu4pyscf follow-up (NOT implemented yet)
`get_k_kpts` never needs the persistent `vR_dm`: the per-block result can be
contracted into `vk` immediately (`vk[i, p0:p1] += weight * blk.dot(ao1)`),
saving `nset * nao * ngrid * 8` bytes for every dense caller (e.g. hybrid
reference K in the CPHF response). It does not by itself make the 216-atom
2-RDM exchange fit (ao_dms and the per-row FFT remain), so it was deferred in
favour of the low-rank path above.

## MO-eri version (`pbc_pprpa_gamma_opt_nv_gpu.py`) — what it additionally needs

The AO-direct contraction rebuilds the pairing kernel from the AOs every Davidson
matvec; at large active spaces (AS=100) that is the bottleneck. The MO-eri path
instead forms the active-space MO integrals once per geometry and contracts them
on the GPU — about 4 orders of magnitude faster per Davidson solve (≈0.3 s vs
≈90 min at AS=100). What it requires on top of the AO-direct example:

1. **Two extra lib_pprpa modules** (now shipped in the package):
   - `lib_pprpa.gpu_ao2mo` — `gpu_ao2mo_blocks(cell, cocc, cvir, mesh, …)` builds the
     active-space `vvvv` / `oovv` / `oooo` blocks via GPU FFT ao2mo (mirrors PySCF's
     `_contract_compact` in real space; validated to ~1e-13 vs CPU reference).
   - `lib_pprpa.pprpa_eri_gpu` — `attach_gpu_eri_contraction(mp, vvvv, oovv, oooo)`
     swaps the Davidson matvec to a batched `use_eri` GPU contraction.
2. **The two blksize patches above ARE required here** (the small AO-direct example
   does not need them). At NV scale (819 AO, mesh 59³) the unpatched exchange and
   K-gradient kernels OOM.
3. **A geometry file** (POSCAR/`.vasp` or `.xyz`) — read via ASE; the lattice is
   taken from it and held fixed during the relaxation.
4. **Enough GPU memory.** The peak is the FFT ao2mo, not the stored ERIs: the
   `vvvv` codensity array is `nvir²·ngrid·8 B`, e.g. AS=100 on the NV cell
   (nvir=100, mesh 59³ ⇒ ngrid≈2.05e5) is ≈16 GB for that one intermediate, plus
   the FFT transients. A 32 GB V100 OOMs at AS=100; use an **A100-80g**
   (`--constraint=a100-80g` on Grace). The example frees the SCF GPU pool before
   ao2mo. To shrink the peak on a smaller GPU, lower `AS` or pass a smaller
   `pair_blk` to `gpu_ao2mo_blocks`.

Tunable knobs exposed on the command line: `ke_cutoff` (plane-wave cutoff),
`mult` (s/t), `AS` (active-space size), `fmax`, `maxsteps`. The defect defaults
(NV⁻-style: `charge=-3`, hh channel, gth-dzvp/gth-pbe, PBE) are set near the top
of the script — edit there for a different defect or charge state.

## Environment notes
- Load CUDA (we used `CUDA/12.8.0`) and the gpu4pyscf conda env before running.
  Bouchet: `module purge && module load CUDA/12.8.0 && source ~/.bash_gpu4pyscf`.
  Grace: `module load miniconda CUDA/12.8.0 imkl/2024.2.0 && source activate gpu4pyscf`,
  with `PYTHONPATH=~/project/gpu4pyscf:~/project/pyscf:~/project/lib_pprpa_public`
  and `CUPY_ACCELERATORS=cutensor,cub`.
- Run on a GPU node (login nodes have no GPU).

## Scope / notes
- Gamma point, RKS/RHF reference, LDA/GGA/hybrid functionals.
- The example uses the **AO-direct** GPU Davidson (library-only). At large active
  spaces the **MO-eri** path (GPU FFT ao2mo + batched `use_eri` contraction) is
  much faster — that is the path used for the NV-center production runs.
- The CPU `make_rdm1_relaxed_rhf_pprpa` builds the relaxed density (small MO-space
  algebra); only `mf.get_k` for the 2-RDM term and the response are routed to GPU.

## Multi-GPU (opt-in) and the low-rank pairing-K force (2026-09-11)

The 216-atom force took 23 h on one B200: pairing-K force 16.4 h (AFT
`get_ek_ip1` on the dense amplitude density), ao2mo 5.2 h, everything else
~1.3 h.  Two changes address this:

1. **Low-rank FFT pairing-K force** (`lib_pprpa/gpu_pairing_force.py`, default in
   `pprpa_gamma_gpu.Gradients`).  X = L R^T with rank <= nocc + nvir, so the
   force reduces to one FFT pass over the r(r+1)/2 active-MO codensity pairs
   plus a chunked gradient-AO contraction.  Exact for a general X (matches the
   CPU reference `pprpa_gamma.py` pairing term to ~1e-10 on the diamond cell);
   `PPRPA_PAIRING_K=aft` or `Gradients.pairing_k_method = "aft"` restores the
   AFT kernel for cross-checks.  Patch 2 above is then only needed for hybrid
   reference functionals.
2. **Device group dispatch** (`lib_pprpa/gpu_multi.py`).  The strip loops of
   ao2mo, the low-rank exchange, the pairing force and the per-atom hcore
   derivative pull global tasks from a shared iterator on one thread per GPU;
   read-only arrays are replicated (peer copy or host bounce), an OOM shrinks
   only that slot's sub-block and redoes the failing task (no whole-tensor
   restart).  It is **opt-in**: `LIB_PPRPA_GPUS=2` (exported by the 2-GPU
   Slurm scripts `pprpa_force_gpu2.sbatch` / `pprpa_opt_gpu2.sbatch`, or
   `submit_pprpa_opt_resumable.sh --gpus 2`) enables two slots; unset means one
   slot and the single-device code path is unchanged.  `DeviceGroup([0, 0])`
   runs two virtual slots on one GPU for tests.  Multi-GPU halves wall-clock
   of those phases only; GPU-hours are unchanged, the algorithmic change above
   is what lowers the budget.

Tests: `tests/test_gpu_multi.py`, `tests/test_gpu_pairing_force.py`,
`tests/test_gpu_ao2mo_multi.py`, `tests/test_gpu_grad_pairing_e2e.py`
(1-GPU node suffices; a `--gres=gpu:2` job exercises real peer copies).
