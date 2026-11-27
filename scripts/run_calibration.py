"""Detector calibration.

Scores the held-out CICIoT2023 test sample with the trained Random Forest and reports
  * reliability (equal-width bins), Brier score, expected calibration error (ECE), log loss;
  * the same after Platt scaling and isotonic regression fitted on the official *validation* file
    (never on the test file);
  * operating points at the policy thresholds (0.5 review floor, 0.8 restriction threshold) and the
    precision these give when attacks are rarer than in the test sample (prevalence reweighting);
  * incident-level scores formed from k flows (single flow, mean, max), as a sensitivity analysis.

The response experiments use one flow per incident (DetectorSampler), so the single-flow row is
the one that matches the paper. Flows in the aggregation rows are drawn independently within a
class, which ignores correlation between flows of one real incident.

Usage: python scripts/run_calibration.py --processed <dir with test.csv> --model <random_forest.joblib>
                                         --validation <dir with the official validation CSV>
Writes results/tables/calibration_*.csv and results/figures/reliability.{pdf,png}.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from secureagent_iot.data import family_of, load_ciciot2023

EC = "real-data (CICIoT2023 official files)"
ap = argparse.ArgumentParser()
ap.add_argument("--processed", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--validation", required=True)
ap.add_argument("--meta", default=None)
ap.add_argument("--bins", type=int, default=15)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
T, F = ROOT / "results" / "tables", ROOT / "results" / "figures"
T.mkdir(parents=True, exist_ok=True); F.mkdir(parents=True, exist_ok=True)

meta = json.load(open(a.meta or Path(a.model).parent.parent / "detection_meta.json"))
feats = meta["features"]
model = joblib.load(a.model)
test = pd.read_csv(Path(a.processed) / "test.csv", low_memory=False)
test["y"] = (test["label"] != "BenignTraffic").astype(int)
val, _ = load_ciciot2023(a.validation, 10000, a.seed)
val["y"] = (val["label"] != "BenignTraffic").astype(int)
# drop validation rows identical to a test row, so calibrators never see test flows
key = lambda d: pd.util.hash_pandas_object(d[feats].round(6), index=False)
val = val[~key(val).isin(set(key(test)))].reset_index(drop=True)

s_te = model.predict_proba(test[feats])[:, 1]
s_va = model.predict_proba(val[feats])[:, 1]
y_te, y_va = test.y.to_numpy(), val.y.to_numpy()
platt = LogisticRegression(C=1e6).fit(np.log(np.clip(s_va, 1e-6, 1 - 1e-6) / np.clip(1 - s_va, 1e-6, 1)).reshape(-1, 1), y_va)
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s_va, y_va)
logit = lambda s: np.log(np.clip(s, 1e-6, 1 - 1e-6) / np.clip(1 - s, 1e-6, 1)).reshape(-1, 1)
scores = {"raw RF score": s_te, "Platt (fit on validation file)": platt.predict_proba(logit(s_te))[:, 1],
          "isotonic (fit on validation file)": iso.predict(s_te)}


def weights(y, pi):
    """Importance weights that move the attack share of the sample to pi."""
    if pi is None:
        return np.ones_like(y, dtype=float)
    p0 = y.mean()
    return np.where(y == 1, pi / p0, (1 - pi) / (1 - p0))


def ece(y, s, w, bins):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, bins - 1)
    tot, out = w.sum(), 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += w[m].sum() / tot * abs(np.average(y[m], weights=w[m]) - np.average(s[m], weights=w[m]))
    return out


def recal_prior(s, pi_from, pi_to):
    """Bayes prior shift of a calibrated probability from the sample prevalence to pi_to."""
    o = s / np.clip(1 - s, 1e-9, None) * (pi_to / (1 - pi_to)) / (pi_from / (1 - pi_from))
    return o / (1 + o)


p0 = y_te.mean()
summ, ops, rel = [], [], []
for name, s in scores.items():
    for pi_label, pi in (("test sample", None), ("0.5", 0.5), ("0.1", 0.1), ("0.01", 0.01)):
        w = weights(y_te, pi)
        summ.append({"evidence_class": EC, "score": name, "attack_prevalence": pi_label if pi else f"test sample ({p0:.3f})",
                     "n": len(y_te), "auroc": roc_auc_score(y_te, s), "brier": brier_score_loss(y_te, s, sample_weight=w),
                     "ece": ece(y_te, s, w, a.bins), "log_loss": log_loss(y_te, np.clip(s, 1e-6, 1 - 1e-6), sample_weight=w),
                     "ece_after_prior_shift": ece(y_te, recal_prior(s, p0, pi), w, a.bins) if pi else np.nan})
    for thr in (0.5, 0.8):
        tpr, fpr = (s[y_te == 1] >= thr).mean(), (s[y_te == 0] >= thr).mean()
        for pi in (p0, 0.5, 0.1, 0.01, 0.001):
            prec = tpr * pi / (tpr * pi + fpr * (1 - pi)) if (tpr * pi + fpr * (1 - pi)) > 0 else np.nan
            ops.append({"evidence_class": EC, "score": name, "threshold": thr, "tpr": tpr, "fpr": fpr,
                        "benign_flows": int((y_te == 0).sum()), "attack_prevalence": pi, "precision": prec,
                        "false_alarms_per_1000_benign": 1000 * fpr})
    edges = np.linspace(0, 1, a.bins + 1)
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, a.bins - 1)
    for b in range(a.bins):
        m = idx == b
        rel.append({"evidence_class": EC, "score": name, "bin_lo": edges[b], "bin_hi": edges[b + 1], "n": int(m.sum()),
                    "n_benign": int((m & (y_te == 0)).sum()), "mean_score": s[m].mean() if m.any() else np.nan,
                    "attack_fraction": y_te[m].mean() if m.any() else np.nan})

# incident-level aggregation (sensitivity): k flows of one class, independently drawn
rng = np.random.default_rng(a.seed)
agg = []
for name in ("raw RF score", "isotonic (fit on validation file)"):
    s = scores[name]
    pools = {0: s[y_te == 0], 1: s[y_te == 1]}
    for k in (1, 3, 5, 10):
        for how in (("single flow",) if k == 1 else ("mean", "max")):
            res = {}
            for cls in (0, 1):
                draws = rng.choice(pools[cls], size=(20000, k), replace=True)
                v = draws[:, 0] if how == "single flow" else (draws.mean(1) if how == "mean" else draws.max(1))
                res[cls] = ((v >= 0.8).mean(), ((v >= 0.5) & (v < 0.8)).mean())
            agg.append({"evidence_class": EC, "score": name, "flows_per_incident": k, "aggregation": how,
                        "benign_incidents_restrict": res[0][0], "benign_incidents_review": res[0][1],
                        "attack_incidents_restrict": res[1][0], "attack_incidents_review": res[1][1], "incidents_per_class": 20000})

pd.DataFrame(summ).to_csv(T / "calibration_summary.csv", index=False)
pd.DataFrame(ops).to_csv(T / "calibration_operating_points.csv", index=False)
pd.DataFrame(rel).to_csv(T / "calibration_reliability.csv", index=False)
pd.DataFrame(agg).to_csv(T / "calibration_incident_aggregation.csv", index=False)
json.dump({"evidence_class": EC, "test_rows": int(len(y_te)), "test_benign": int((y_te == 0).sum()),
           "validation_rows_used": int(len(y_va)), "validation_benign": int((y_va == 0).sum()), "bins": a.bins,
           "validation_input": str(a.validation)}, open(T / "calibration_meta.json", "w"), indent=1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r = pd.DataFrame(rel)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
styles = {"raw RF score": ("#1f4e79", "o"), "Platt (fit on validation file)": ("#b35900", "s"), "isotonic (fit on validation file)": ("#2e7d32", "^")}
ax[0].plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
for name, (c, mk) in styles.items():
    g = r[(r.score == name) & (r.n >= 20)]
    ax[0].plot(g.mean_score, g.attack_fraction, marker=mk, ms=3.5, lw=1.1, color=c, label=name)
ax[0].set_xlabel("mean predicted score in bin"); ax[0].set_ylabel("fraction of attack flows")
ax[0].legend(fontsize=6.5, frameon=False, loc="upper left"); ax[0].set_title("Reliability (test sample)", fontsize=8.5)
g = r[r.score == "raw RF score"]
ax[1].bar(g.bin_lo + 1 / (2 * a.bins), g.n, width=1 / a.bins * 0.9, color="#1f4e79", label="all flows")
ax[1].bar(g.bin_lo + 1 / (2 * a.bins), g.n_benign, width=1 / a.bins * 0.9, color="#b35900", label="benign flows")
ax[1].set_yscale("log"); ax[1].set_xlabel("raw RF score"); ax[1].set_ylabel("flows (log scale)")
ax[1].axvline(0.5, color="0.4", lw=0.7, ls=":"); ax[1].axvline(0.8, color="0.2", lw=0.7, ls="--")
ax[1].legend(fontsize=6.5, frameon=False); ax[1].set_title("Score histogram", fontsize=8.5)
for x in ax:
    x.tick_params(labelsize=7); x.xaxis.label.set_size(7.5); x.yaxis.label.set_size(7.5)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(F / f"reliability.{ext}", dpi=600 if ext == "png" else None)
print(pd.DataFrame(summ).round(4).to_string())
print(pd.DataFrame(ops).query("attack_prevalence != @p0").round(4).to_string())
print(pd.DataFrame(agg).round(4).to_string())
