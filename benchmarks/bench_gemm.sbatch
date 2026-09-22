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
echo "[job] done"
