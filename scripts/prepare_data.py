#!/usr/bin/env python
"""Validate, subsample and split CICIoT2023 CSVs. Refuses to substitute toy
data unless --demo is given explicitly."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from secureagent_iot.config import load_config, set_global_seed
from secureagent_iot.data import DataUnavailable, file_split, load_ciciot2023, make_toy, split, validate_and_report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/raw/CICIoT2023")
    ap.add_argument("--test-input", default=None, help="directory with a separate official test file; if given, no random split")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--demo", action="store_true", help="use TOY data (SANITY CHECK ONLY - NOT RESEARCH RESULTS)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    set_global_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.demo:
        print("*** SANITY CHECK ONLY - NOT RESEARCH RESULTS: using generated toy data ***")
        raw, manifest, cls, prefix = make_toy(seed=a.seed), [], "toy-sanity", "DEMO_"
    else:
        try:
            raw, manifest = load_ciciot2023(a.input, int(cfg["detection"]["max_rows_per_class_per_file"]), a.seed)
        except DataUnavailable as exc:
            print(f"BLOCKED: {exc}", file=sys.stderr)
            return 2
        cls, prefix = "real-data", ""
    df, report = validate_and_report(raw)
    if a.test_input and not a.demo:
        raw_te, man_te = load_ciciot2023(a.test_input, int(cfg["detection"]["max_rows_per_class_per_file"]), a.seed + 1)
        df_te, rep_te = validate_and_report(raw_te)
        train, test, info = file_split(df, df_te)
        report["test_file"] = {k: rep_te[k] for k in ("rows_before_dedup", "exact_duplicates_removed", "rows_after_dedup", "binary_distribution")}
        manifest = manifest + [{**m, "role": "test"} for m in man_te]
    else:
        train, test, info = split(df, float(cfg["detection"]["test_fraction"]), a.seed)
    report.update({"evidence_class": cls, "seed": a.seed, "split": info, "input": "generated toy" if a.demo else str(a.input),
                   "sampling": "toy generator" if a.demo else
                   f"per file, per label: up to {cfg['detection']['max_rows_per_class_per_file']} rows, uniform without replacement",
                   "test_input": a.test_input})
    train.to_csv(out / f"{prefix}train.csv", index=False)
    test.to_csv(out / f"{prefix}test.csv", index=False)
    (out / f"{prefix}data_report.json").write_text(json.dumps(report, indent=1, default=str))
    if manifest:
        pd.DataFrame(manifest).to_csv(out / "raw_file_checksums.csv", index=False)
    if report["unmapped_labels"]:
        print(f"WARNING: unmapped labels (treated as attack, family=UNMAPPED): {report['unmapped_labels']}")
    print(f"[{cls}] rows={report['rows_after_dedup']} features={report['n_features']} split={info['method']} "
          f"train={info['train_rows']} test={info['test_rows']} cross-split identical rows={info['cross_split_identical_feature_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
