#!/bin/bash
#SBATCH --job-name=ao2mo_gemm_bench
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:b200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=05:30:00
#SBATCH --output=bench_%j.out
#SBATCH --error=bench_%j.out
#
# gpu_ao2mo GEMM benchmark on the NV63 cell (nao=819, ke=300 -> mesh 107^3).
# Runs from the isolated checkout: sbatch --chdir=<checkout>/benchmarks bench_gemm.sbatch
#   STAGE=1  symmetric tiles (compact pairs + lower triangle) vs frozen baseline
# Optional: CUDA_MODULE=CUDA/12.9.1 to pick up cuBLAS 12.9 (fp64 emulation, stage 3).
set -euo pipefail

CHECKOUT="$(cd "$(dirname "$(pwd)")" && pwd)"
export PYTHONPATH="${CHECKOUT}${PYTHONPATH:+:$PYTHONPATH}"
GEOM="${GEOM:-/home/sc3352/project/vibrationally-aware-spectra/work/pprpa/NV63/grace_opt.xyz}"
STAGE="${STAGE:-1}"

module purge
source /home/sc3352/project/vibrationally-aware-spectra/scripts/setup_env.sh
if [[ -n "${CUDA_MODULE:-}" ]]; then
  module unload CUDA 2>/dev/null || true
  module load "${CUDA_MODULE}"
fi
module list 2>&1 | grep -i cuda || true
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "[job] checkout=${CHECKOUT} stage=${STAGE} cwd=$(pwd)"
python -c "import lib_pprpa, sys; print('[job] lib_pprpa ->', lib_pprpa.__file__)"

echo "=== unit tests (host-only layout, CPU-reference diamond cell, planner caps) ==="
python "${CHECKOUT}/tests/test_pair_layout.py"
python "${CHECKOUT}/tests/test_gpu_ao2mo_multi.py"
python "${CHECKOUT}/tests/test_fft_batch_cap.py"
python "${CHECKOUT}/lib_pprpa/gpu_ao2mo.py"      # __main__: element-wise vs pyscf, two strip sizes

B="python ${CHECKOUT}/benchmarks/bench_gpu_ao2mo.py --geom ${GEOM} --ke 300"
# Reference tensors (vvvv/oovv/oooo .npy, ~80 GB at AS=300) live on scratch and are
# written once (--save) so later stages only --compare against them instead of
# recomputing an earlier kernel.  REF_STAGE2 = commit f138fc3 (stage 2), the
# baseline every later change is checked against.
REF_ROOT="${REF_ROOT:-/nfs/roberts/scratch/pi_tz324/sc3352/ao2mo_refs}"
REF_STAGE2="${REF_ROOT}/nv63_ke300_as300_stage2"
ensure_ref_stage2() {
  if [[ -f "${REF_STAGE2}/oooo.npy" && -f "${REF_STAGE2}/vvvv.npy" && -f "${REF_STAGE2}/oovv.npy" ]]; then
    echo "[job] stage2 reference present: ${REF_STAGE2}"
  else
    echo "=== writing the stage2 reference tensors to ${REF_STAGE2} (once) ==="
    $B --as 300 --impl stage2 --tag ref_stage2 --save "${REF_STAGE2}"
  fi
}
if [[ "$STAGE" == "1" ]]; then
  echo "=== AS=100, pair_blk=600: base vs new (shape check, seconds) ==="
  $B --as 100 --pair-blk 600 --impl base,new --tag s1_as100_pb600
  echo "=== AS=300, pair_blk=600: base vs new (production strip width) ==="
  $B --as 300 --pair-blk 600 --impl base,new --tag s1_as300_pb600
  echo "=== AS=300, planner pair_blk: new ==="
  $B --as 300 --impl new --tag s1_as300_auto
  echo "=== AS=300, pair_blk=600, host-staged finals: new (NV216 write path) ==="
  $B --as 300 --pair-blk 600 --impl new --force-host --tag s1_as300_pb600_host
fi
if [[ "$STAGE" == "2" ]]; then
  # stage1 = symmetric tiles with the strip still tied to the FFT batch (64 B/pt, cuFFT cap);
  # new    = FFT sub-batches decoupled from the GEMM strip (24 B/pt), planner picks the width
  echo "=== tile GEMM micro-benchmark: NV63 K (1.2e6) and NV216 K (4.0e6) ==="
  python "${CHECKOUT}/benchmarks/bench_gemm_shapes.py" --K 1225043 --widths 300,600,1200,2400,4800
  python "${CHECKOUT}/benchmarks/bench_gemm_shapes.py" --K 4019679 --widths 300,600,1200,2400
  echo "=== AS=300, planner widths: stage1 vs new ==="
  $B --as 300 --impl stage1,new --tag s2_as300_auto
  echo "=== AS=300, new at forced strip widths (rate vs width) ==="
  $B --as 300 --pair-blk 600 --impl new --tag s2_as300_pb600
  $B --as 300 --pair-blk 1200 --impl new --tag s2_as300_pb1200
  $B --as 300 --pair-blk 2400 --impl new --tag s2_as300_pb2400
  echo "=== AS=300, planner widths, host-staged finals: new (NV216 write path) ==="
  $B --as 300 --impl new --force-host --tag s2_as300_auto_host
fi
if [[ "$STAGE" == "7" ]]; then
  # Compare-only flow: the current kernel (R2C default) against the saved stage-2
  # reference; no earlier kernel is recomputed.
  export GPU_AO2MO_PROFILE=1
  ensure_ref_stage2
  echo "=== AS=300, planner widths: new vs stage2 reference ==="
  $B --as 300 --impl new --tag s7_as300 --compare "${REF_STAGE2}"
  echo "=== AS=300, pair_blk=1500 (NV216-like width): new vs stage2 reference ==="
  $B --as 300 --pair-blk 1500 --impl new --tag s7_as300_pb1500 --compare "${REF_STAGE2}"
  echo "=== AS=300, planner widths, host-staged: new vs stage2 reference ==="
  $B --as 300 --impl new --force-host --tag s7_as300_host --compare "${REF_STAGE2}"
fi
if [[ "$STAGE" == "6" ]]; then
  # Stage 6: fixed-size FFT sub-batches (one cached plan), task-level buffers,
  # diagonal tile splitting, symmetrised R2C kernel, R2C budget 40 B/pt.
  export GPU_AO2MO_PROFILE=1
  echo "=== AS=300, planner widths: stage2 vs new_rfft ==="
  $B --as 300 --impl stage2,new_rfft --tag s6_as300_rfft
  echo "=== AS=300, planner widths: new (C2C) vs new_rfft (R2C) ==="
  $B --as 300 --impl new,new_rfft --tag s6_as300_c2c_vs_r2c
  echo "=== AS=300, pair_blk=1500 (NV216-like width): new_rfft ==="
  $B --as 300 --pair-blk 1500 --impl new_rfft --tag s6_as300_pb1500
  echo "=== AS=300, planner widths, host-staged: new_rfft ==="
  $B --as 300 --impl new_rfft --force-host --tag s6_as300_rfft_host
fi
if [[ "$STAGE" == "5" ]]; then
  # Non-GEMM overhead: fused codensity kernel (new) vs take+multiply (stage2), and
  # the R2C Coulomb chain (new_rfft) vs C2C.  GPU_AO2MO_PROFILE syncs and times
  # every strip phase (rho_outer / fft / rho_inner / gemm / scatter).
  export GPU_AO2MO_PROFILE=1
  echo "=== AS=300, planner widths: stage2 vs new (fused codensity, C2C) ==="
  $B --as 300 --impl stage2,new --tag s5_as300_fused
  echo "=== AS=300, planner widths: new (C2C) vs new_rfft (R2C) ==="
  $B --as 300 --impl new,new_rfft --tag s5_as300_rfft
  echo "=== AS=300, pair_blk=1500 (NV216-like width): new vs new_rfft ==="
  $B --as 300 --pair-blk 1500 --impl new,new_rfft --tag s5_as300_pb1500
  echo "=== AS=300, planner widths, host-staged: new_rfft ==="
  $B --as 300 --impl new_rfft --force-host --tag s5_as300_rfft_host
fi
if [[ "$STAGE" == "3" ]]; then
  # cuBLAS fp64 fixed-point emulation needs cuBLAS >= 13.0u2; the project venv is
  # CUDA 12.8, so the tile GEMM is measured from a scratch CUDA 13 cupy venv
  # (CU13_VENV) with the emulation toggled by environment variable.
  CU13_VENV="${CU13_VENV:-/nfs/roberts/project/pi_tz324/sc3352/venv_cu13}"
  P13="${CU13_VENV}/bin/python"
  "$P13" -c "import cupy; print('[s3] cupy', cupy.__version__, 'cuBLAS', cupy.cuda.cublas.getVersion(cupy.cuda.device.get_cublas_handle()))"
  for K in 1225043 4019679; do
    echo "=== K=${K}: native fp64 ==="
    CUBLAS_EMULATE_DOUBLE_PRECISION=0 "$P13" "${CHECKOUT}/benchmarks/bench_gemm_shapes.py" --K "$K" --widths 600,1200,2400
    for strat in performant eager; do
      echo "=== K=${K}: fp64 emulation, strategy=${strat} ==="
      CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY="$strat" \
        "$P13" "${CHECKOUT}/benchmarks/bench_gemm_shapes.py" --K "$K" --widths 600,1200,2400
    done
  done
fi
if [[ "$STAGE" == "4" ]]; then
  # Full ao2mo in the CUDA 13 venv (pip gpu4pyscf-cuda13x): native fp64 vs cuBLAS
  # fp64 emulation, same SCF orbitals (cached), element-wise comparison via .npy.
  CU13_VENV="${CU13_VENV:-/nfs/roberts/project/pi_tz324/sc3352/venv_cu13}"
  P13="${CU13_VENV}/bin/python"
  export CUPY_ACCELERATORS=cub
  B13="$P13 ${CHECKOUT}/benchmarks/bench_gpu_ao2mo.py --geom ${GEOM} --ke 300"
  REF="$(pwd)/s4_native_ref"
  "$P13" -c "import gpu4pyscf, cupy; print('[s4] gpu4pyscf', gpu4pyscf.__version__, 'cupy', cupy.__version__, 'cuBLAS', cupy.cuda.cublas.getVersion(cupy.cuda.device.get_cublas_handle()))"
  echo "=== AS=300, CUDA 13 venv, native fp64: new (saves reference) ==="
  CUBLAS_EMULATE_DOUBLE_PRECISION=0 $B13 --as 300 --impl new --tag s4_cu13_native --save "$REF"
  echo "=== AS=300, CUDA 13 venv, fp64 emulation (eager): new vs native ==="
  CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=eager \
    $B13 --as 300 --impl new --tag s4_cu13_emul_eager --compare "$REF"
  echo "=== AS=300, CUDA 13 venv, fp64 emulation (performant): new vs native ==="
  CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=performant \
    $B13 --as 300 --impl new --tag s4_cu13_emul_perf --compare "$REF"
  rm -rf "$REF"
fi
echo "[job] done"
