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
echo "[job] done"
