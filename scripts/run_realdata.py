#!/usr/bin/env python
"""Real-traffic evaluation (E20). Requires prepared CICIoT2023 data and a trained
detector (scripts/prepare_data.py, scripts/train_ids.py). Never substitutes toy data.

1. Leakage checks beyond exact duplicates: near-duplicate flows across the split.
2. Unseen-family generalization: the detector is retrained without one traffic
   family and tested on that family's held-out flows (real false-negative rates).
3. Clean response and injection benchmarks with every scenario's detector
   probability drawn from the trained Random Forest's scores on held-out real
   flows of the matching class (and traffic family where one exists).
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from secureagent_iot.config import load_config
from secureagent_iot.data import feature_columns
from secureagent_iot.evaluation import run_attack_benchmark, run_response_benchmark, summarize_attacks, summarize_response
from secureagent_iot.simulator import DetectorSampler


def near_duplicates(train, test, feats, digits=3):
    r = lambda d: pd.util.hash_pandas_object(d[feats].astype(float).apply(lambda c: np.round(c, digits)), index=False).values
    tr = set(r(train))
    return int(np.isin(r(test), list(tr)).sum())


def unseen_family(train, test, feats, seed, n_train=60000):
    rows = []
    for fam in sorted(f for f in test.family.unique() if f != "benign"):
        tr = train[train.family != fam]
        tr = tr.sample(min(n_train, len(tr)), random_state=seed)
        m = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=seed)
        m.fit(tr[feats].fillna(tr[feats].median()), tr["y"])
        te = test[test.family.isin([fam, "benign"])]
        X = te[feats].fillna(tr[feats].median())
        s = m.predict_proba(X)[:, 1]
        att = te.family.to_numpy() == fam
        rows.append({"held_out_family": fam, "test_flows_of_family": int(att.sum()),
                     "detection_rate_at_0.5": float((s[att] >= 0.5).mean()), "detection_rate_at_0.8": float((s[att] >= 0.8).mean()),
                     "median_p_attack": float(np.median(s[att])), "benign_false_positive_rate_at_0.8": float((s[~att] >= 0.8).mean()),
                     "auc_vs_benign": float(roc_auc_score(att, s))})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--models", default="results/raw")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    a = ap.parse_args(argv)
    cfg = load_config()
    seeds = a.seeds or list(cfg["seeds"])
    d, m = Path(a.data), Path(a.models)
    need = [d / "train.csv", d / "test.csv", d / "data_report.json", m / "models" / "random_forest.joblib", m / "detection_meta.json"]
    if not all(p.exists() for p in need):
        print(f"BLOCKED: real-data inputs missing: {[str(p) for p in need if not p.exists()]}", file=sys.stderr)
        return 2
    report, meta = json.loads(need[2].read_text()), json.loads(need[4].read_text())
    if report["evidence_class"] != "real-data" or meta["evidence_class"] != "real-data":
        print("BLOCKED: prepared data or model is not real-data.", file=sys.stderr)
        return 2
    train, test = pd.read_csv(need[0]), pd.read_csv(need[1])
    feats, _ = feature_columns(train)
    T, R = Path(a.out) / "tables", Path(a.out) / "raw"
    T.mkdir(parents=True, exist_ok=True)
    cls = "real-data detector (CICIoT2023 flows) driving testbed-synthetic scenarios"

    leak = {"evidence_class": "real-data", "train_rows": len(train), "test_rows": len(test),
            "exact_cross_split_duplicates": report["split"]["cross_split_identical_feature_rows"],
            "near_duplicates_3_decimals": near_duplicates(train, test, feats, 3),
            "near_duplicates_1_decimal": near_duplicates(train, test, feats, 1),
            "split_method": report["split"]["method"], "exact_duplicates_removed_before_split": report["exact_duplicates_removed"],
            "identical_rows_removed_from_test": report["split"].get("identical_rows_removed_from_test", 0)}
    pd.DataFrame([leak]).to_csv(T / "real_leakage_checks.csv", index=False)
    print("leakage:", leak)

    second = Path("data/raw/CICIoT2023_second")
    if second.exists() and any(second.glob("*.csv*")):
        from secureagent_iot.data import load_ciciot2023, validate_and_report
        from secureagent_iot.detection import binary_metrics
        raw2, man2 = load_ciciot2023(second, int(cfg["detection"]["max_rows_per_class_per_file"]), 7)
        d2, rep2 = validate_and_report(raw2)
        h_tr = set(pd.util.hash_pandas_object(train[feats], index=False).values)
        overlap = int(np.isin(pd.util.hash_pandas_object(d2[feats], index=False).values, list(h_tr)).sum())
        model = joblib.load(need[3])
        sc2 = model.predict_proba(d2[meta["features"]])[:, 1]
        y2 = d2["y"].to_numpy()
        cf = {"evidence_class": "real-data", "second_file": man2[0]["file"], "second_file_sha256": man2[0]["sha256"],
              "rows_evaluated": len(d2), "identical_rows_shared_with_training": overlap,
              **{k: v for k, v in binary_metrics(y2, (sc2 >= 0.5).astype(int), sc2).items()},
              "recall_attack_at_0.8": float((sc2[y2 == 1] >= 0.8).mean()), "fpr_at_0.8": float((sc2[y2 == 0] >= 0.8).mean())}
        pd.DataFrame([cf]).to_csv(T / "real_cross_file.csv", index=False)
        print("cross-file:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cf.items()})

    uf = unseen_family(train, test, feats, 42) if not (T / "real_unseen_family.csv").exists() else pd.read_csv(T / "real_unseen_family.csv").drop(columns=["evidence_class"])
    uf.insert(0, "evidence_class", "real-data")
    uf.to_csv(T / "real_unseen_family.csv", index=False)
    print(uf.round(3).to_string(index=False))

    sampler = DetectorSampler(joblib.load(need[3]), test, meta["features"], "real-data")
    resp = run_response_benchmark(cfg, seeds, sampler=sampler)
    resp["evidence_class"] = cls
    resp.to_csv(R / "real_response_runs.csv", index=False)
    rs = summarize_response(resp, cfg)
    rs["evidence_class"] = cls
    rs.to_csv(T / "real_response_summary.csv", index=False)
    c = dict(cfg); c["_sampler"] = sampler
    atk = run_attack_benchmark(c, seeds, systems=["B1_unguarded", "B1v_schema_only", "B2_guarded"])
    atk["evidence_class"] = cls
    atk.to_csv(R / "real_attack_runs.csv", index=False)
    for name, t in summarize_attacks(atk, c).items():
        t["evidence_class"] = cls
        t.to_csv(T / f"real_{name}.csv", index=False)
    print(rs[rs.metric.isin(["safe_containment", "false_isolation", "unauthorized_action", "review_rate"])].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
