#!/usr/bin/env bash
#SBATCH -J saiot-llm
#SBATCH -p gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-99%6
#SBATCH -o logs/%A_%a.out
#
# The full real-LLM run as a Slurm job array over (candidate model, prompt condition, seed): one single-GPU task
# per triple. Each task runs the response benchmark, then the attack benchmark, for ITS model, condition and seed.
# Files are named LLM_<model>__<condition>_..._seed<S>, so tasks never write the same file. Both benchmarks
# checkpoint every finished attempt, so resume is always on: resubmitting a task skips whatever is already done.
#
# A task whose model is NOT in preflight/selected.json (written by scripts/llm_preflight.sbatch) exits at once
# with exit code 0 and a message; only selected models are run. Submit it after the preflight:
#
#   source scripts/llm_candidates.sh
#   pre=$(sbatch --parsable --exclude=<busy nodes> scripts/llm_preflight.sbatch)
#   sbatch --dependency=afterany:$pre --exclude=<busy nodes> --array=0-99%<clean GPUs> scripts/llm_array.sbatch
#   bash scripts/llm_progress.sh                                  # progress, any time
#   # after ALL tasks finished, once per selected model and condition:
#   python scripts/merge_llm_shards.py --model <model> --condition <condition>
#   # resubmit one task (it resumes):   sbatch --array=<index> scripts/llm_array.sbatch
#
# Array index -> (model, condition, seed), with n_c conditions and n_s seeds:
#   model = CANDIDATES[index / (n_c * n_s)],  condition = CONDITIONS[(index / n_s) % n_c],  seed = SEEDS[index % n_s].
# The --array range must be 0..(n_candidates * n_c * n_s - 1): 0-99 for the default 5 x 4 x 5. CANDIDATES must be
# the list the preflight was given. Check the mapping without running anything (safe on the head node):
#   for i in $(seq 0 99); do SLURM_ARRAY_TASK_ID=$i DRY_RUN=1 bash scripts/llm_array.sbatch; done
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source scripts/llm_candidates.sh
CONDITIONS="${CONDITIONS:-base defended policy policy_defended}"     # LLM_PROMPT values; see llm_adapter.system_prompt
SEEDS="${SEEDS:-11 23 42 77 101}"          # must match `seeds:` in the config for merge_llm_shards.py to accept the result
OUT="${OUT:-results/raw}"
SELECTED="${SELECTED:-preflight/selected.json}"

read -r -a models <<< "$CANDIDATES"
read -r -a conditions <<< "$CONDITIONS"
read -r -a seeds <<< "$SEEDS"
n_models=${#models[@]}
n_cond=${#conditions[@]}
n_seeds=${#seeds[@]}
n_tasks=$((n_models * n_cond * n_seeds))
idx="${SLURM_ARRAY_TASK_ID:?run this with sbatch as a job array}"
if [ "$idx" -ge "$n_tasks" ]; then
  echo "array index $idx has no (model, condition, seed): ${n_models} x ${n_cond} x ${n_seeds} = ${n_tasks} tasks; nothing to do"
  exit 0
fi
if [ -n "${SLURM_ARRAY_TASK_MAX:-}" ] && [ "$SLURM_ARRAY_TASK_MAX" -lt $((n_tasks - 1)) ] && [ "$idx" -eq "${SLURM_ARRAY_TASK_MIN:-0}" ] && [ "${SLURM_ARRAY_TASK_COUNT:-2}" -gt 1 ]; then
  echo "WARNING: --array ends at ${SLURM_ARRAY_TASK_MAX} but ${n_models} models x ${n_cond} conditions x ${n_seeds} seeds need 0-$((n_tasks - 1)); some triples will not run"
fi
model="${models[$((idx / (n_cond * n_seeds)))]}"
condition="${conditions[$(((idx / n_seeds) % n_cond))]}"
seed="${seeds[$((idx % n_seeds))]}"
if [ -n "${DRY_RUN:-}" ]; then echo "index $idx -> model=$model condition=$condition seed=$seed out=$OUT"; exit 0; fi

source scripts/llm_env.sh
echo "array job ${SLURM_ARRAY_JOB_ID:-?}_${idx} (job ${SLURM_JOB_ID:-?}) on $(hostname): model=$model condition=$condition seed=$seed out=$OUT CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Only models the preflight selected are run.
if [ ! -f "$SELECTED" ]; then
  echo "NOT RUN: $SELECTED does not exist. The preflight did not finish (see its log); nothing is selected, so this task does nothing."
  exit 3
fi
if ! python - "$SELECTED" "$model" <<'PY'
import json, sys
sel = json.load(open(sys.argv[1]))
if sys.argv[2] in sel.get("selected", []):
    sys.exit(0)
print(f"NOT RUN: {sys.argv[2]} is not in {sys.argv[1]} (selected: {sel.get('selected')}). "
      f"Reason: {sel.get('why', {}).get(sys.argv[2], 'not a candidate of that preflight')}")
sys.exit(1)
PY
then
  echo "== skipped (model not selected): $model [$condition] seed $seed =="
  exit 0
fi

export LLM_PROVIDER=hf
export LLM_MODEL="$model"
export LLM_PROMPT="$condition"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"     # weights were downloaded by the preflight; tasks never touch the network
export SAIOT_CODE_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "commit=$SAIOT_CODE_COMMIT dirty_files=$(git status --porcelain -- src scripts configs 2>/dev/null | wc -l) LLM_CACHE=${LLM_CACHE:-unset (cache on)}"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
fi
python scripts/gpu_proof.py
python scripts/freeze_prompts.py --check       # a prompt edited after the freeze stops the task here

echo "== $model [$condition] seed $seed: response benchmark (resumes if a checkpoint exists) =="
python scripts/run_response_benchmark.py --llm --seed "$seed" --out "$OUT"
echo "== $model [$condition] seed $seed: attack benchmark (resumes if a checkpoint exists) =="
python scripts/run_attack_benchmark.py --llm --seed "$seed" --out "$OUT"
echo "== done: $model [$condition] seed $seed =="
