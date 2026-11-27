#!/usr/bin/env bash
# Regenerates every result in results/ that the available resources allow, then writes results/SHA256SUMS.
#
#   ./RUN_ALL.sh           tests and all CPU experiments; real-data steps run only if CICIoT2023 is in data/raw/
#   ./RUN_ALL.sh --demo    also a toy detection run that checks the wiring (outputs prefixed DEMO_)
#   ./RUN_ALL.sh --quick   one seed and fewer resamples, for a fast check (numbers will differ from the paper)
#
# The real-LLM runs need GPUs and are not repeated here; their raw outputs are committed under results/raw/LLM_*
# and the analysis below recomputes every LLM table from them. To rerun the models, see README.md.
set -uo pipefail
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate
DEMO=0; QUICK=""
for arg in "$@"; do
  case "$arg" in --demo) DEMO=1 ;; --quick) QUICK="--quick" ;; *) echo "unknown option: $arg"; exit 2 ;; esac
done
fail=0
step() { echo; echo "== $* =="; }

step "tests"; pytest -q || fail=1

step "detection on CICIoT2023"
if [ -d data/raw/CICIoT2023_train ] && find data/raw/CICIoT2023_train -name '*.csv*' | grep -q . \
   && find data/raw/CICIoT2023 -name '*.csv*' | grep -q .; then
  python scripts/prepare_data.py --input data/raw/CICIoT2023_train --test-input data/raw/CICIoT2023 --out data/processed --seed 42 \
    && python scripts/train_ids.py --input data/processed --out results/raw --seed 42 || fail=1
elif [ -d data/raw/CICIoT2023 ] && find data/raw/CICIoT2023 -name '*.csv*' | grep -q .; then
  python scripts/prepare_data.py --input data/raw/CICIoT2023 --out data/processed --seed 42 \
    && python scripts/train_ids.py --input data/processed --out results/raw --seed 42 || fail=1
else
  echo "skipped: no CICIoT2023 files in data/raw/ (see data/MANIFEST.md)"
fi
if [ "$DEMO" = 1 ]; then
  step "toy detection run (SANITY CHECK ONLY - NOT RESEARCH RESULTS)"
  python scripts/prepare_data.py --demo --seed 42 && python scripts/train_ids.py --demo --seed 42 || fail=1
  python scripts/run_response_benchmark.py --detector --demo --seed 42 || fail=1
fi

step "response benchmark";        python scripts/run_response_benchmark.py || fail=1
step "attack benchmark";          python scripts/run_attack_benchmark.py || fail=1
step "proposal-space search";     python scripts/run_envelope_search.py || fail=1
step "E1-E23, including SMT proofs and mutation testing"; python scripts/run_extended.py $QUICK || fail=1
step "E30-E32";                   python scripts/run_approvals_corroboration.py $QUICK || fail=1
step "real-LLM analysis from the committed raw runs"; python scripts/analyze_llm_run.py || fail=1
step "E40-E47";                   python scripts/run_baselines_enforcement.py $QUICK || fail=1
step "template-clustered reanalysis"; python scripts/reanalysis_template_clusters.py || fail=1

if [ -f results/raw/models/random_forest.joblib ] && grep -q '"real-data"' data/processed/data_report.json 2>/dev/null; then
  step "real-traffic evaluation (E20) and random-split comparison"
  python scripts/run_realdata.py || fail=1
  python scripts/random_split_baseline.py || fail=1
  if [ -d data/raw/CICIoT2023_second ]; then
    step "detector calibration"
    python scripts/run_calibration.py --processed data/processed --model results/raw/models/random_forest.joblib \
      --validation data/raw/CICIoT2023_second --meta results/raw/detection_meta.json || fail=1
    python scripts/run_calibration_response.py --processed data/processed --model results/raw/models/random_forest.joblib \
      --validation data/raw/CICIoT2023_second --meta results/raw/detection_meta.json || fail=1
  fi
fi

step "kernel enforcement in an isolated loopback-only namespace (needs root, unshare, nft)"
python scripts/run_netns_enforcement.py; rc=$?
[ "$rc" = 2 ] && echo "skipped"; [ "$rc" != 0 ] && [ "$rc" != 2 ] && fail=1
step "emulated multi-host network (needs root, unshare, nft, pyroute2; about 10 min)"
python scripts/run_netemu_enforcement.py; rc=$?
[ "$rc" = 2 ] && echo "skipped"; [ "$rc" != 0 ] && [ "$rc" != 2 ] && fail=1

step "figures";        python scripts/make_figures.py || fail=1
step "hash manifest";  python scripts/hash_results.py || fail=1

[ "$fail" = 0 ] && echo "RUN_ALL: OK" || { echo "RUN_ALL: FAILURES"; exit 1; }
