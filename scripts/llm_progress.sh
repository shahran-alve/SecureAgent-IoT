#!/usr/bin/env bash
# Progress of the full real-LLM run. Shell only (squeue, sacct, wc), so it is safe on the head node:
#   bash scripts/llm_progress.sh
# A finished shard has 204 response rows and 2560 attack rows (the first line of each .jsonl is a header).
cd "$(dirname "$0")/.."
OUT="${OUT:-results/raw}"
echo "== queue =="
squeue -u "$USER" -o "%.12i %.9P %.16j %.2t %.10M %.6D %R" 2>/dev/null
if [ -f logs/full_run_jobs.env ]; then
  . logs/full_run_jobs.env
  echo "== accounting (preflight $PREFLIGHT_JOB, array $ARRAY_JOB) =="
  sacct -X -j "$PREFLIGHT_JOB,$ARRAY_JOB${RESUBMITTED_JOBS:+,$RESUBMITTED_JOBS}" --format=JobID%14,State%12,ExitCode,Elapsed,NodeList%9 -n 2>/dev/null \
    | awk '{s[$2]++} END {for (k in s) printf "  %-12s %d\n", k, s[k]}'
  echo "== preflight =="
  if [ -f preflight/selected.json ]; then grep -A6 '"selected"' preflight/selected.json | head -8
  else echo "  preflight/selected.json not written yet; last log lines:"; grep "\[preflight" "logs/$PREFLIGHT_JOB.out" 2>/dev/null | tail -4; fi
fi
echo "== shards: rows done / rows planned =="
shopt -s nullglob
n=0; done_n=0
for f in "$OUT"/LLM_*_runs_seed*.jsonl; do
  case "$f" in *response_runs*) total=204 ;; *) total=2560 ;; esac
  rows=$(( $(wc -l < "$f") - 1 ))
  n=$((n + 1)); [ "$rows" -ge "$total" ] && done_n=$((done_n + 1))
  printf "  %5d / %-5d %s\n" "$rows" "$total" "$(basename "$f" .jsonl)"
done
echo "$done_n of $n shard files complete (a selected model has 40: 4 conditions x 5 seeds x 2 benchmarks)"
