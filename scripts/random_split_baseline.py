#!/usr/bin/env python
"""Comparison for the detector evaluation: the same models trained and tested on a stratified random 80/20 split of the
official test file alone, the design used when only one file is available. Shows how much a within-file random split
flatters the detector compared with the official file split. Writes results/tables/real_random_split.csv."""
import tempfile
from pathlib import Path

import pandas as pd

from secureagent_iot.config import load_config
from secureagent_iot.data import load_ciciot2023, split, validate_and_report
from secureagent_iot.detection import fit_and_evaluate
from secureagent_iot.data import feature_columns


def main() -> int:
    cfg = load_config()
    src = Path("data/raw/CICIoT2023")
    if not (src.exists() and any(src.glob("*.csv*"))):
        print("BLOCKED: data/raw/CICIoT2023 has no CSV files")
        return 2
    raw, _ = load_ciciot2023(src, int(cfg["detection"]["max_rows_per_class_per_file"]), 42)
    df, _ = validate_and_report(raw)
    train, test, info = split(df, float(cfg["detection"]["test_fraction"]), 42)
    feats, _ = feature_columns(train)
    metrics, _, _, _ = fit_and_evaluate(train, test, feats, cfg, 42)
    metrics.insert(0, "evidence_class", "real-data")
    metrics.insert(1, "design", "stratified random 80/20 split of the test file")
    out = Path("results/tables/real_random_split.csv")
    metrics.to_csv(out, index=False)
    print(metrics[["model", "recall_attack", "fpr", "macro_f1"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
