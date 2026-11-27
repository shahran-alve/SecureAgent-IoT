#!/usr/bin/env bash
#SBATCH -J saiot-final
#SBATCH -p gpuq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -o logs/%j.out
#
# Final job of run1 (CPU only, no GPU). Submit with --dependency=afterany:<Mistral array> from the main repository.
#   0. Check that all 40 Mistral shards (4 conditions x 5 seeds x 2 benchmarks) are complete. If any is missing or
#      incomplete: write logs/FINAL_JOB_BLOCKED.txt naming them and exit 1, doing nothing else.
#   a. Merge Mistral for each of the 4 conditions (scripts/merge_llm_shards.py).
#   b. Archive every per-seed JSONL shard and call log of all four models into ~/saiot_archive/run1_raw.tar.gz and
#      write ~/saiot_archive/run1_raw.sha256 (one line per member file, then the tarball).
#   c. Run scripts/analyze_llm_run.py (results/tables/LLM_analysis_*.csv and results/LLM_ANALYSIS.md).
#   d. Pack results/tables/LLM_* and results/raw/LLM_*_runs.csv into ~/saiot_results_csv.tar.gz.
# It commits nothing.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this with sbatch from the main repository}"
source scripts/llm_env.sh
MISTRAL="mistralai/Mistral-Nemo-Instruct-2407"
MODELS_ALL="Qwen/Qwen2.5-7B-Instruct google/gemma-4-12b-it microsoft/Phi-4-mini-instruct $MISTRAL"
ARCHIVE_DIR="$HOME/saiot_archive"
echo "final job ${SLURM_JOB_ID:-?} on $(hostname) at $(date), commit $(git rev-parse HEAD) dirty=$(git status --porcelain -- src scripts configs | wc -l)"
rm -f logs/FINAL_JOB_BLOCKED.txt

echo "== 0. Mistral shard check =="
if ! python - "$MISTRAL" <<'PY'
import sys
from pathlib import Path
from secureagent_iot.config import load_config
from secureagent_iot.llm_harness import KEY_FIELDS, file_prefix, load_checkpoint, ordered_rows, plan_keys
model, cfg = sys.argv[1], load_config()
problems, ok = [], 0
for cond in ("base", "defended", "policy", "policy_defended"):
    for kind in ("response", "attack"):
        for seed in cfg["seeds"]:
            path = Path("results/raw") / f"{file_prefix(model, cond)}{kind}_runs_seed{seed}.jsonl"
            if not path.exists():
                problems.append(f"MISSING     {path}")
                continue
            try:
                meta, rows = load_checkpoint(path, KEY_FIELDS[kind])
                ordered_rows(rows, kind, plan_keys(kind, cfg, [int(seed)]), where=path.name)
                if meta.get("prompt_condition") != cond or meta.get("model") != model:
                    raise ValueError(f"header says {meta.get('model')} / {meta.get('prompt_condition')}")
                ok += 1
            except Exception as exc:
                problems.append(f"INCOMPLETE  {path}: {str(exc)[:200]}")
if problems:
    Path("logs/FINAL_JOB_BLOCKED.txt").write_text(
        f"Final job BLOCKED: {len(problems)} of 40 Mistral shards are missing or incomplete ({ok} complete).\n"
        "Nothing was merged, archived or analysed. Resubmit the failed array tasks (they resume), then this job.\n\n"
        + "\n".join(problems) + "\n")
    print("\n".join(problems))
    sys.exit(1)
print(f"all {ok} Mistral shards complete")
PY
then
  echo "BLOCKED: see logs/FINAL_JOB_BLOCKED.txt"
  exit 1
fi

echo "== a. merge Mistral =="
for c in base defended policy policy_defended; do
  python scripts/merge_llm_shards.py --model "$MISTRAL" --condition "$c"
done

echo "== b. archive per-seed shards and call logs of all four models =="
mkdir -p "$ARCHIVE_DIR"
members=()
for m in $MODELS_ALL; do
  slug=$(python -c "from secureagent_iot.llm_harness import model_slug; import sys; print(model_slug(sys.argv[1]))" "$m")
  for f in results/raw/LLM_${slug}__*_runs_seed*.jsonl results/raw/LLM_${slug}__*_calls_seed*.jsonl; do
    [ -f "$f" ] && members+=("$f")
  done
done
echo "${#members[@]} files (expected 160: 4 models x 4 conditions x 5 seeds x 2 benchmarks x {shard, call log})"
if [ "${#members[@]}" -ne 160 ]; then echo "WARNING: expected 160 member files, found ${#members[@]}"; fi
tmp="$ARCHIVE_DIR/run1_raw.tar.gz.tmp$$"
tar -czf "$tmp" "${members[@]}"
mv "$tmp" "$ARCHIVE_DIR/run1_raw.tar.gz"
{ sha256sum "${members[@]}"; ( cd "$ARCHIVE_DIR" && sha256sum run1_raw.tar.gz ); } > "$ARCHIVE_DIR/run1_raw.sha256"
tar -tzf "$ARCHIVE_DIR/run1_raw.tar.gz" | wc -l | xargs echo "members in tarball:"
ls -la "$ARCHIVE_DIR"

echo "== c. analysis =="
python scripts/analyze_llm_run.py

echo "== d. CSV results tarball =="
tar -czf "$HOME/saiot_results_csv.tar.gz.tmp$$" results/tables/LLM_* results/raw/LLM_*_runs.csv
mv "$HOME/saiot_results_csv.tar.gz.tmp$$" "$HOME/saiot_results_csv.tar.gz"
ls -la "$HOME/saiot_results_csv.tar.gz"
echo "== final job done at $(date) =="
