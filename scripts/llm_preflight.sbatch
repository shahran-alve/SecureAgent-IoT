#!/usr/bin/env bash
#SBATCH -J saiot-preflight
#SBATCH -p gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH -o logs/%j.out
#
# Model preflight (one job, one GPU): looks every candidate up on the Hub, downloads the ones that qualify, runs a
# small SMOKE_ benchmark with the prompt cache on and off, and writes preflight/<model>.json and
# preflight/selected.json. See scripts/llm_preflight.py for the rules.
#   sbatch scripts/llm_preflight.sbatch
#   CANDIDATES="Qwen/Qwen2.5-7B-Instruct some/other-model" sbatch scripts/llm_preflight.sbatch
# The job array (scripts/llm_array.sbatch) is submitted with --dependency=afterany:<this job> and the SAME CANDIDATES.
# This job needs the network (Hub lookups and downloads). HF_TOKEN is used only if it is already in the submitting
# environment; it is never printed.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source scripts/llm_candidates.sh
source scripts/llm_env.sh
unset HF_HUB_OFFLINE
export SAIOT_CODE_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "preflight job=$SLURM_JOB_ID node=$(hostname) commit=$SAIOT_CODE_COMMIT dirty_files=$(git status --porcelain -- src scripts configs 2>/dev/null | wc -l) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "candidates: $CANDIDATES"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
fi
python scripts/gpu_proof.py
python scripts/freeze_prompts.py --check
python scripts/llm_preflight.py
echo "== preflight done =="
