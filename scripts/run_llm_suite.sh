#!/usr/bin/env bash
# Real-LLM evaluation with open-weight Hugging Face models run locally.
#
#   1. pip install -e ".[dev,local-llm]"
#   2. export HF_TOKEN=...            # only for gated models; never commit it
#   3. ./scripts/run_llm_suite.sh                       # default: the four models of the paper, all 5 seeds
#      MODELS="Qwen/Qwen2.5-7B-Instruct" SEED=42 ./scripts/run_llm_suite.sh   # one seed: a fifth of the cost
#   4. python scripts/analyze_llm_run.py              # LLM tables from results/raw/LLM_*
#
# Both benchmarks are checkpointed: if this script is interrupted, run it again and it resumes where it stopped.
# On a Slurm cluster, prefer scripts/llm_array.sbatch (one single-GPU job per model and seed, run in parallel).
#
# Each model replaces BOTH scripted agents. Its output is untrusted: it goes through strict schema
# validation and the same deterministic policy engine as every scripted agent. Nothing executes outside
# the simulated controller. Decoding is greedy, so a rerun on the same hardware reproduces the outputs.
# Rough cost: ~2,000 model calls for the clean benchmark and ~5,000 for the attack benchmark per model
# with 5 seeds (about 1-3 h per 7-8B model on one modern GPU). Use SEED=42 for a fifth of that.
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS="${MODELS:-Qwen/Qwen2.5-7B-Instruct google/gemma-4-12b-it mistralai/Mistral-Nemo-Instruct-2407 microsoft/Phi-4-mini-instruct}"
CONDITIONS="${CONDITIONS:-base defended policy policy_defended}"   # prompt conditions (LLM_PROMPT); each is a separate run
SEED="${SEED:-}"
export LLM_PROVIDER=hf
python - <<'EOF'
import importlib.util, sys
missing = [m for m in ("transformers", "torch") if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"BLOCKED: install {missing} first (pip install 'transformers>=4.45' torch accelerate)")
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU will be very slow)")
EOF
for m in $MODELS; do
  export LLM_MODEL="$m"
  for c in $CONDITIONS; do
    export LLM_PROMPT="$c"
    echo "== $m [$c]: clean response benchmark =="
    python scripts/run_response_benchmark.py --llm ${SEED:+--seed "$SEED"}
    echo "== $m [$c]: injection benchmark =="
    python scripts/run_attack_benchmark.py --llm ${SEED:+--seed "$SEED"}
  done
done
# results/raw/LLM_environment.json (dtype, model revision hash, torch/transformers versions, GPU, peak GPU memory) is
# written by the benchmark scripts themselves, one entry per model.
echo "Done. Send back: zip -r llm_results.zip results/raw/LLM_* results/tables/LLM_*"
