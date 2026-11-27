"""Experiments E40-E47: E40 generic tool-call baseline, E41 corroboration mitigations,
E42 approvals behind a finite queue, E43 scenario selection, E44 four-model table, E45 enforcement failure injection.
Usage: python scripts/run_baselines_enforcement.py [--quick] [--only e40,e41,e42,e43,e44,e45]"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from secureagent_iot import extended5 as R
from secureagent_iot.config import load_config, set_global_seed
from secureagent_iot.enforcement import failure_injection

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--only", default="e40,e41,e42,e43,e44,e45,e46,e47")
ap.add_argument("--config", default=None)
a = ap.parse_args()
cfg = load_config(a.config)
seeds = list(cfg["seeds"])
set_global_seed(seeds[0])
if a.quick:
    seeds = seeds[:1]
ROOT = Path(__file__).resolve().parents[1]
T, RAW = ROOT / "results" / "tables", ROOT / "results" / "raw"
T.mkdir(parents=True, exist_ok=True)
only = set(a.only.split(","))


def save(df, name):
    df.to_csv(T / name, index=False)
    print(f"wrote {name} ({len(df)} rows)", flush=True)


t0 = time.time()
if "e40" in only:
    summ, viol, env, camp = R.generic_baseline(cfg, seeds)
    save(summ, "generic_baseline.csv"); save(viol, "generic_violations.csv")
    save(env, "generic_envelope.csv"); save(camp, "generic_campaigns.csv")
if "e41" in only:
    save(R.corroboration_mitigations(cfg, seeds), "corroboration_mitigations.csv")
if "e42" in only:
    save(R.approvals_queue(cfg, seeds), "approvals_queue.csv")
if "e43" in only:
    subsets, per_t, loo, prev = R.scenario_selection(pd.read_csv(RAW / "attack_runs.csv"), pd.read_csv(RAW / "response_runs.csv"))
    save(subsets, "scenario_subsets.csv"); save(per_t, "residual_by_template.csv")
    save(loo, "leave_one_template_out.csv"); save(prev, "lookalike_prevalence.csv")
if "e44" in only:
    save(R.unified_llm_table(T, RAW), "llm_all_models.csv")
if "e45" in only:
    trial_log = []
    save(failure_injection(cfg, trials=5 if a.quick else 20, trial_log=trial_log), "failure_injection.csv")
    pd.DataFrame(trial_log).to_csv(RAW / "failure_injection_trials.csv", index=False)
if "e46" in only:
    save(R.generated_holdout(cfg, seeds, n_templates=100 if a.quick else 400), "generated_holdout.csv")
if "e47" in only:
    from secureagent_iot.harmbound import mitigation_bounds
    save(mitigation_bounds(cfg), "harm_bound_mitigations.csv")
print(f"done in {time.time() - t0:.0f} s")
