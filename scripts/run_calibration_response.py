"""Response-level effect of detector calibration.

The trained Random Forest's score on a held-out flow drives each incident (DetectorSampler, one flow per incident).
Four score maps are passed through the unchanged engine:
  raw          the Random Forest score (as in the real-traffic benchmarks)
  isotonic     isotonic regression fitted on the official validation file (flows identical to test flows removed)
  prior 0.10   isotonic, then a Bayes prior shift from the calibration sample's attack share to 0.10
  prior 0.01   the same to 0.01
and the clean-scenario and attack benchmarks are rerun for B0 and B2. Only the score map changes.

Usage: python scripts/run_calibration_response.py --processed <dir with test.csv> --model <random_forest.joblib>
                                                  --validation <dir with the official validation CSV> --meta <json>
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from secureagent_iot.config import load_config
from secureagent_iot.data import family_of, load_ciciot2023
from secureagent_iot.evaluation import run_attack_benchmark, run_response_benchmark
from secureagent_iot.extended2 import _wrongful_seconds_attack
from secureagent_iot.simulator import DetectorSampler

EC = "real-data scores (CICIoT2023) driving simulated incidents"
ap = argparse.ArgumentParser()
for k in ("processed", "model", "validation", "meta"):
    ap.add_argument(f"--{k}", required=True)
a = ap.parse_args()
cfg = load_config(None)
seeds = list(cfg["seeds"])
meta = json.load(open(a.meta))
feats = meta["features"]
rf = joblib.load(a.model)
test = pd.read_csv(Path(a.processed) / "test.csv", low_memory=False)
test["y"] = (test["label"] != "BenignTraffic").astype(int)
test["family"] = test["label"].map(family_of)
val, _ = load_ciciot2023(a.validation, 10000, 42)
val["y"] = (val["label"] != "BenignTraffic").astype(int)
key = lambda d: pd.util.hash_pandas_object(d[feats].round(6), index=False)
val = val[~key(val).isin(set(key(test)))].reset_index(drop=True)
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(rf.predict_proba(val[feats])[:, 1], val["y"])
p_cal = float(val["y"].mean())


def shift(p, pi):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    o = p / (1 - p) * (pi / (1 - pi)) / (p_cal / (1 - p_cal))
    return o / (1 + o)


class Mapped:
    def __init__(self, f):
        self.f = f

    def predict_proba(self, X):
        s = self.f(rf.predict_proba(X)[:, 1])
        return np.column_stack([1 - s, s])


MAPS = {"raw": lambda s: s, "isotonic": lambda s: iso.predict(s),
        "isotonic + prior 0.10": lambda s: shift(iso.predict(s), 0.10),
        "isotonic + prior 0.01": lambda s: shift(iso.predict(s), 0.01)}
rows = []
for name, f in MAPS.items():
    sampler = DetectorSampler(Mapped(f), test, feats, "real-data")
    r = run_response_benchmark(cfg, seeds, systems=["B0_rules", "B2_guarded"], sampler=sampler)
    c = dict(cfg); c["_sampler"] = sampler
    k = run_attack_benchmark(c, seeds, systems=["B2_guarded"])
    ex = k[k.injected & ~k.suppression_scored]
    sup = k[k.injected & k.suppression_scored]
    for sysname in ("B0_rules", "B2_guarded"):
        d = r[r.system == sysname]
        row = {"evidence_class": EC, "score_map": name, "system": sysname,
               "safe_containment": d.loc[d.containment_worthy, "safe_contained"].mean(),
               "false_restriction_benign": d.loc[~d.truth_malicious, "false_isolation"].mean(),
               "missed_malicious": d.loc[d.truth_malicious, "missed"].mean(),
               "review_rate_clean": d.escalated.mean(), "unauthorized_clean": d.spec_violation.mean()}
        if sysname == "B2_guarded":
            row.update({"attack_prohibited": int(ex.any_violation.sum()), "attack_attempts": len(ex),
                        "harmful_permitted": int((ex.goal_executed & ex.benign_isolated & ~ex.any_violation).sum()),
                        "wrongful_s_per_100_attacked": 100 * _wrongful_seconds_attack(ex).mean(),
                        "suppression": sup.suppression_success.mean(), "review_rate_attacked": ex.escalated.mean()})
        rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results" / "tables" / "calibration_response.csv", index=False)
pd.set_option("display.width", 250)
print(out.drop(columns="evidence_class").round(3).to_string())
