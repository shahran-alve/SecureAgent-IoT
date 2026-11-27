#!/usr/bin/env bash
#SBATCH -J saiot-heldout
#SBATCH -p gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-29%6
#SBATCH -o logs/llm_heldout_%A_%a.out
#
# Optional real-LLM follow-up runs: repeated sampling and held-out injection styles. No result in the paper comes
# from this script. It reuses llm_array.sbatch's plumbing (one model, condition and seed per task, resumable shards) and
# sets two switches that run 1 never set:
#
#   EXPERIMENT=sampling   non-greedy decoding, LLM_TEMPERATURE=0.7, LLM_TOP_P=0.95, five repeated generations per
#                         prompt (LLM_SAMPLE_SEED = 1..5), main injection styles, prompt conditions policy and
#                         policy_defended, scenario seed 42. Answers: how much of run 1's greedy result is one sample.
#   EXPERIMENT=heldout    greedy decoding, the four held-out injection styles (attacks.HELDOUT_STYLES: json_field,
#                         prior_decision, split, vendor_request), written after run 1 and never tuned on any model,
#                         all four prompt conditions, scenario seed 42. Answers: does the prompt-level reduction
#                         carry over to phrasings the prompts were not revised against.
#
# Each experiment writes to its own directory so checkpoints never mix with run 1 (results/raw):
#   results/raw_sampling/s<k>/   and   results/raw_heldout/
# Analyse with the run-1 script, pointing it at that directory, e.g.
#   LLM_STYLE_SET=heldout python scripts/analyze_llm_run.py --raw results/raw_heldout --tables results/tables_heldout
#   (the analysis expects seeds 11..101; set SEEDS in that script to [42] for these single-seed runs)
#
# Array index -> (model, condition, sample) for EXPERIMENT=sampling: 3 models x 2 conditions x 5 samples = 30 tasks;
# for EXPERIMENT=heldout: 3 models x 4 conditions = 12 tasks (submit with --array=0-11).
# A larger model can be added by putting it first in MODELS; it must pass scripts/llm_preflight.sbatch first.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
EXPERIMENT="${EXPERIMENT:?set EXPERIMENT=sampling or EXPERIMENT=heldout}"
MODELS="${MODELS:-Qwen/Qwen2.5-7B-Instruct google/gemma-4-12b-it mistralai/Mistral-Nemo-Instruct-2407}"
read -r -a models <<< "$MODELS"
idx="${SLURM_ARRAY_TASK_ID:?run this with sbatch as a job array}"
if [ "$EXPERIMENT" = sampling ]; then
  conds=(policy policy_defended); n_s=5
  model="${models[$((idx / (2 * n_s)))]:-}"; condition="${conds[$(((idx / n_s) % 2))]}"; sample=$((idx % n_s + 1))
  export LLM_TEMPERATURE=0.7 LLM_TOP_P=0.95 LLM_SAMPLE_SEED=$sample
  OUT="results/raw_sampling/s${sample}"
else
  conds=(base defended policy policy_defended)
  model="${models[$((idx / 4))]:-}"; condition="${conds[$((idx % 4))]}"
  export LLM_STYLE_SET=heldout
  OUT="results/raw_heldout"
fi
[ -z "$model" ] && { echo "index $idx: nothing to do"; exit 0; }
mkdir -p "$OUT" logs
if [ -n "${DRY_RUN:-}" ]; then echo "index $idx -> $EXPERIMENT model=$model condition=$condition out=$OUT ${LLM_SAMPLE_SEED:+sample=$LLM_SAMPLE_SEED}"; exit 0; fi
source scripts/llm_env.sh
export LLM_PROVIDER=hf LLM_MODEL="$model" LLM_PROMPT="$condition"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
echo "$EXPERIMENT: model=$model condition=$condition out=$OUT temperature=${LLM_TEMPERATURE:-greedy} sample=${LLM_SAMPLE_SEED:-} styles=${LLM_STYLE_SET:-main}"
python scripts/gpu_proof.py
python scripts/freeze_prompts.py --check       # system prompts must be the frozen run-1 prompts
python scripts/run_response_benchmark.py --llm --seed 42 --out "$OUT"
python scripts/run_attack_benchmark.py --llm --seed 42 --out "$OUT"
