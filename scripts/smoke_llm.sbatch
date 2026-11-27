#!/usr/bin/env bash
#SBATCH -J saiot-smoke
#SBATCH -p gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH -o logs/%j.out
#
# Smoke test of the real-LLM path on ONE GPU for ONE prompt condition, with a limited number of cases. Outputs are
# prefixed SMOKE_. One job per condition; run the conditions in parallel on separate GPUs:
#   for c in base defended policy policy_defended; do LLM_PROMPT=$c sbatch scripts/smoke_llm.sbatch; done
# Resubmitting the same line after a kill resumes.
#   LLM_PROMPT       required: base | defended | policy | policy_defended
#   BENCH            both (default) | response | attack
#   RESPONSE_LIMIT   scenarios in the response benchmark (default 68 = all of them)
#   ATTACK_LIMIT     cases per attack variant in the attack benchmark (default 2)
#   LLM_MODEL (default Qwen/Qwen2.5-7B-Instruct), SEED (42), OUT (results/raw), LLM_DTYPE, ENV_NAME
# The model must already be in HF_HOME (scripts/hf_download.sbatch); this job runs offline.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source scripts/llm_env.sh
export LLM_PROVIDER=hf
export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export LLM_PROMPT="${LLM_PROMPT:?set LLM_PROMPT=base|defended|policy|policy_defended}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
BENCH="${BENCH:-both}"
RESPONSE_LIMIT="${RESPONSE_LIMIT:-68}"
ATTACK_LIMIT="${ATTACK_LIMIT:-2}"
SEED="${SEED:-42}"
OUT="${OUT:-results/raw}"
echo "job=$SLURM_JOB_ID node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} model=$LLM_MODEL condition=$LLM_PROMPT bench=$BENCH seed=$SEED response_limit=$RESPONSE_LIMIT attack_limit=$ATTACK_LIMIT out=$OUT"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
fi
python scripts/gpu_proof.py
if [ "$BENCH" = both ] || [ "$BENCH" = response ]; then
  python scripts/run_response_benchmark.py --llm --seed "$SEED" --limit-cases "$RESPONSE_LIMIT" --out "$OUT"
fi
if [ "$BENCH" = both ] || [ "$BENCH" = attack ]; then
  python scripts/run_attack_benchmark.py --llm --seed "$SEED" --limit-cases "$ATTACK_LIMIT" --out "$OUT"
fi
echo "== smoke done: $LLM_MODEL [$LLM_PROMPT] seed $SEED =="
