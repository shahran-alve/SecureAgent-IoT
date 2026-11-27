# SecureAgent-IoT

Code, data pointers, and results for the paper *SecureAgent-IoT: Evaluating Deterministic Authorization and Residual
Attack Risk in Agentic IoT Incident Response*.

An agent (scripted or LLM) proposes typed response actions for IoT incidents. A deterministic, default-deny policy
engine decides each proposal, and a simulated controller applies the allowed ones as expiring rules. The experiments
measure what the policy blocks and what harm it still permits: wrongful restriction of benign devices, suppressed
containment, and review load.

Nothing here touches a real network, shell, firewall, or device. Agents cannot run commands: the four allowlisted
actions map to methods of an in-memory simulator. The two optional kernel-enforcement experiments run nftables inside
isolated network namespaces that are created and destroyed by the script.

## Layout

```
configs/default.yaml        policy thresholds, budgets, seeds (11, 23, 42, 77, 101)
src/secureagent_iot/        policy engine, simulator, agents, attacks, enforcement, formal model, experiments
scripts/                    one entry point per experiment (see below)
tests/                      unit, property, and regression tests (pytest)
results/raw/                per-run outputs, including the real-LLM runs and their call logs (*.jsonl.gz)
results/tables/             aggregate tables used in the paper
results/figures/            figures
results/SHA256SUMS          hash of every result file
data/MANIFEST.md            how to obtain and place CICIoT2023 (not redistributed)
reproducibility/            environment, package lock files, failure-injection log, notes on the SMT findings
PROMPTS_FROZEN.json         hashes and text of the LLM system prompts and injection wrappers
```

## Install

Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,xgb]'
pytest -q
```

The exact versions used for the paper are in `reproducibility/requirements.lock.txt` (simulation and detection) and
`reproducibility/llm_requirements.lock.txt` (real-LLM runs).

## Reproduce

```bash
./RUN_ALL.sh            # tests, every CPU experiment, figures, results/SHA256SUMS
./RUN_ALL.sh --quick    # one seed, fewer resamples: a fast check whose numbers differ from the paper
```

Steps that need something missing are skipped with a message: the detection and calibration steps need the
CICIoT2023 files (`data/MANIFEST.md`), and the kernel-enforcement steps need root, `unshare`, `nft`, and `pyroute2`.
Compare a run with the committed results by

```bash
sha256sum -c results/SHA256SUMS | grep -v ': OK$'
```

Wall-clock timing tables (`ext_scalability.csv`, the latency figures, and the timing columns of the
kernel-enforcement tables) differ between machines; everything else is seeded.

### Where each result comes from

| Result | Script | Main outputs (`results/tables/` unless stated) |
|---|---|---|
| Clean response benchmark, systems B0-B3 and ablations | `run_response_benchmark.py` | `response_summary.csv`, `response_paired.csv`, `review_queue.csv` |
| Prompt-injection benchmark | `run_attack_benchmark.py` | `attack_overall.csv`, `attack_by_variant.csv`, `attack_by_category.csv` |
| Exhaustive proposal grid (2,798,880 proposals) | `run_envelope_search.py` | `envelope_summary.csv`, `envelope_actions.csv` |
| E1-E23: population, profiles, detector quality, thresholds, mutation testing, review queue, repeated restriction, adaptive campaigns, harm bound, SMT proofs | `run_extended.py` | `ext_*.csv` |
| E30-E32: analyst approvals, filter false positives, manufactured corroboration | `run_approvals_corroboration.py` | `approvals_*.csv`, `filter_fpr.csv`, `manufactured_corroboration.csv`, `fleet_crowding.csv` |
| Real-LLM analysis (four models, four prompts, five seeds) | `analyze_llm_run.py` | `LLM_analysis_family*.csv`, `results/LLM_ANALYSIS.md` |
| E40 generic tool-call baseline | `run_baselines_enforcement.py` | `generic_*.csv` |
| E41 corroboration budget and hard ceilings | `run_baselines_enforcement.py` | `corroboration_mitigations.csv`, `harm_bound_mitigations.csv` |
| E42 approvals behind a finite queue | `run_baselines_enforcement.py` | `approvals_queue.csv` |
| E43 scenario selection, look-alike prevalence | `run_baselines_enforcement.py` | `residual_by_template.csv`, `scenario_subsets.csv`, `leave_one_template_out.csv`, `lookalike_prevalence.csv` |
| E44 four-model summary | `run_baselines_enforcement.py` | `llm_all_models.csv` |
| E45 enforcement failure injection (11 modes, 20 trials each) | `run_baselines_enforcement.py` | `failure_injection.csv`, `results/raw/failure_injection_trials.csv` |
| E46 generated holdout | `run_baselines_enforcement.py` | `generated_holdout.csv` |
| E47 SMT proofs on the mitigated configuration | `run_baselines_enforcement.py` | `formal_mitigated_*.csv` |
| Template-clustered reanalysis | `reanalysis_template_clusters.py` | `template_cluster_*.csv` |
| Detection on CICIoT2023 | `prepare_data.py`, `train_ids.py` | `results/raw/detection_*.csv` |
| E20 real-traffic evaluation, random-split comparison | `run_realdata.py`, `random_split_baseline.py` | `real_*.csv` |
| Detector calibration and its effect on responses | `run_calibration.py`, `run_calibration_response.py` | `calibration_*.csv` |
| Kernel enforcement, loopback namespace | `run_netns_enforcement.py` | `ext_netns_enforcement.csv` |
| E21 emulated multi-host network | `run_netemu_enforcement.py` | `ext_netemu_enforcement.csv` |
| Figures | `make_figures.py` | `results/figures/` |

The experiment scripts take `--config` (default `configs/default.yaml`). Each experiment can be rerun on its own, for example
`python scripts/run_baselines_enforcement.py --only e45`.

## Real-LLM runs

The four open-weight models (Qwen2.5-7B-Instruct, gemma-4-12b-it, Mistral-Nemo-Instruct-2407, Phi-4-mini-instruct)
were run greedily in float16 on single V100 GPUs. Their raw outputs are committed, so `RUN_ALL.sh` recomputes every LLM
table without a GPU. The model revisions, library versions, and GPU of each run are recorded in
`results/raw/LLM_*environment*.json`.

To rerun the models:

```bash
pip install -e '.[local-llm]'
bash scripts/run_llm_suite.sh                 # one machine; set MODELS="..." to choose models
```

On a Slurm cluster:

```bash
sbatch scripts/setup_llm_env.sbatch                              # conda environment and a GPU check
pre=$(sbatch --parsable scripts/llm_preflight.sbatch)            # download and smoke-test each candidate model
sbatch --dependency=afterany:$pre scripts/llm_array.sbatch       # one task per (model, prompt condition, seed)
python scripts/merge_llm_shards.py --model Qwen/Qwen2.5-7B-Instruct --condition policy   # per model and condition
```

`LLM_PROMPT` selects the prompt condition (`base`, `defended`, `policy`, `policy_defended`). The prompts are frozen in
`PROMPTS_FROZEN.json`; `python scripts/freeze_prompts.py --check` verifies them.
`scripts/llm_heldout_sampling.sbatch` runs two optional follow-ups (repeated sampling and held-out injection styles);
no result in the paper comes from it.

## Toy run

`./RUN_ALL.sh --demo` adds a detection run on generated toy data to check the pipeline wiring. Its outputs are
prefixed `DEMO_`, marked SANITY CHECK ONLY — NOT RESEARCH RESULTS, ignored by git, and never used in the paper.

## Citation

See `CITATION.cff`.
