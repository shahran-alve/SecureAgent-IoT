#!/usr/bin/env python
"""Train and evaluate IDS baselines on prepared data. Output evidence class
(real-data vs toy-sanity) is inherited from the data report and stamped on
every file."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from secureagent_iot.config import load_config, set_global_seed
from secureagent_iot.data import feature_columns
from secureagent_iot.detection import HAVE_XGB, describe_rule, fit_and_evaluate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/processed")
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--demo", action="store_true", help="read DEMO_ toy files (SANITY CHECK ONLY)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    set_global_seed(a.seed)
    prefix = "DEMO_" if a.demo else ""
    src = Path(a.input)
    need = [src / f"{prefix}{n}" for n in ("train.csv", "test.csv", "data_report.json")]
    if not all(p.exists() for p in need):
        print(f"BLOCKED: prepared data not found in '{src}' ({[p.name for p in need]}). Run scripts/prepare_data.py first "
              "(real data), or pass --demo to both scripts for a toy sanity run.", file=sys.stderr)
        return 2
    report = json.loads(need[2].read_text())
    cls = report["evidence_class"]
    if (cls == "toy-sanity") != a.demo:
        print("BLOCKED: evidence class of prepared data does not match --demo flag.", file=sys.stderr)
        return 2
    if a.demo:
        print("*** SANITY CHECK ONLY - NOT RESEARCH RESULTS ***")
    train, test = pd.read_csv(need[0]), pd.read_csv(need[1])
    feats, _ = feature_columns(train)
    metrics, cv, curves, fitted = fit_and_evaluate(train, test, feats, cfg, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for df in (metrics, cv):
        df.insert(0, "evidence_class", cls)
        df.insert(1, "seed", a.seed)
    metrics["train_fingerprint"] = report["split"]["train_fingerprint"]
    metrics["test_fingerprint"] = report["split"]["test_fingerprint"]
    metrics.to_csv(out / f"{prefix}detection_metrics.csv", index=False)
    cv.to_csv(out / f"{prefix}detection_cv.csv", index=False)
    for name, c in curves.items():
        c.to_csv(out / f"{prefix}pr_curve_{name}.csv", index=False)
    models_dir = out / "models"
    models_dir.mkdir(exist_ok=True)
    for name, m in fitted.items():
        joblib.dump(m, models_dir / f"{prefix}{name}.joblib")
    meta = {"evidence_class": cls, "seed": a.seed, "features": feats, "xgboost_available": HAVE_XGB,
            "rule": describe_rule(fitted["rule_threshold"], feats), "imbalance_handling": "class_weight=balanced (RF, rule); scale_pos_weight (XGB)",
            "preprocessing": "median imputation inside Pipeline (fit on train folds only); tree models, no scaling",
            "data_report": str(need[2])}
    (out / f"{prefix}detection_meta.json").write_text(json.dumps(meta, indent=1))
    cols = ["model", "precision_attack", "recall_attack", "f1_attack", "macro_f1", "pr_auc", "fpr"]
    print(f"[{cls}]\n" + metrics[cols].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
