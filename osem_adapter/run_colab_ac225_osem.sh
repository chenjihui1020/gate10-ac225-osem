#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs cuda_outputs
ITERATIONS="${ITERATIONS:-48}"
SUBSETS="${SUBSETS:-4}"
SIGMA_DEG="${SIGMA_DEG:-6}"
NX="${NX:-40}"
NY="${NY:-40}"
NZ="${NZ:-25}"
FOV_X_MM="${FOV_X_MM:-100}"
FOV_Y_MM="${FOV_Y_MM:-100}"
FOV_Z_MM="${FOV_Z_MM:-50}"
EVAL_ENERGY_WINDOW_FRAC="${EVAL_ENERGY_WINDOW_FRAC:-0.15}"

echo "=== CUDA device ===" | tee logs/colab_run.log
nvidia-smi | tee -a logs/colab_run.log

echo "=== Compile ===" | tee -a logs/colab_run.log
nvcc -O3 -std=c++17 -arch=sm_75 src/ac225_osem_cuda.cu -o ac225_osem_cuda 2>&1 | tee logs/nvcc_build.log

run_one() {
  local name="$1"
  local input="$2"
  if [[ ! -s "${input}" ]]; then
    echo "skip ${name}: missing or empty ${input}" | tee -a logs/colab_run.log
    return 0
  fi
  local outdir="cuda_outputs/${name}"
  mkdir -p "${outdir}"
  echo "=== Run ${name} from ${input} ===" | tee -a logs/colab_run.log
  ./ac225_osem_cuda \
    --input "${input}" \
    --outdir "${outdir}" \
    --nx "${NX}" --ny "${NY}" --nz "${NZ}" \
    --fov-x-mm "${FOV_X_MM}" --fov-y-mm "${FOV_Y_MM}" --fov-z-mm "${FOV_Z_MM}" \
    --iterations "${ITERATIONS}" --subsets "${SUBSETS}" --sigma-deg "${SIGMA_DEG}" \
    2>&1 | tee "logs/${name}_cuda.log"
  python3 scripts/plot_cuda_recon.py --output-dir "${outdir}" --fov-x-mm "${FOV_X_MM}" --fov-y-mm "${FOV_Y_MM}" \
    2>&1 | tee "logs/${name}_plot.log"
}

run_one "ac225_all_variable_energy" "data/ac225_osem_events_all.csv"
run_one "ac225_line218" "data/ac225_osem_events_line218.csv"
run_one "ac225_line218_sourceqa20" "data/ac225_osem_events_line218_sourceqa20.csv"
run_one "ac225_line440" "data/ac225_osem_events_line440.csv"
run_one "ac225_line440_sourceqa20" "data/ac225_osem_events_line440_sourceqa20.csv"
run_one "ac225_sourceqa20" "data/ac225_osem_events_sourceqa20.csv"

echo "=== Paper-style evaluation plots ===" | tee -a logs/colab_run.log
python3 scripts/evaluate_ac225_results.py \
  --converted-dir data \
  --cuda-output-root cuda_outputs \
  --output-dir evaluation \
  --energy-window-frac "${EVAL_ENERGY_WINDOW_FRAC}" \
  2>&1 | tee logs/evaluation.log

echo "=== Summaries ===" | tee -a logs/colab_run.log
find cuda_outputs -name summary.txt -print -exec cat {} \; | tee logs/all_cuda_summaries.txt

zip -r ac225_osem_cuda_results.zip cuda_outputs evaluation logs data README.md src scripts run_colab_ac225_osem.sh >/dev/null
echo "created ${ROOT_DIR}/ac225_osem_cuda_results.zip" | tee -a logs/colab_run.log
