"""Dataset loading, validation, subsetting and splitting.

Real data: CICIoT2023 flow-feature CSVs obtained by the user from the
official UNB/CIC distribution and placed under data/raw/CICIoT2023/.
Nothing here downloads data. `make_toy` produces an obviously artificial
table used only for pipeline sanity checks (never a research result).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

LABEL_CANDIDATES = ["label", "Label", "LABEL"]
GROUP_COL = "_source_file"
BENIGN_NAMES = {"benigntraffic", "benign", "normal"}
# CICIoT2023 label prefix -> family. PENDING VERIFICATION against the real label
# set on first real run; unmapped labels are reported, never silently dropped.
FAMILY_PREFIXES = [
    ("ddos", "ddos"), ("dos", "dos"), ("mirai", "mirai"), ("recon", "recon"), ("vulnerabilityscan", "recon"),
    ("mitm", "spoofing"), ("dns_spoofing", "spoofing"), ("dictionarybruteforce", "bruteforce"),
    ("sqlinjection", "web"), ("xss", "web"), ("commandinjection", "web"), ("uploading_attack", "web"),
    ("browserhijacking", "web"), ("backdoor_malware", "web"),
]
LEAKY_PATTERNS = ("label", "file", "path", "attack", "timestamp")
LEAKY_EXACT = {"ts", "id", "time"}


class DataUnavailable(Exception):
    pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def frame_fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


def family_of(label: str) -> str:
    l = str(label).strip().lower()
    if l in BENIGN_NAMES:
        return "benign"
    for prefix, fam in FAMILY_PREFIXES:
        if l.startswith(prefix):
            return fam
    return "UNMAPPED"


def find_label_column(df: pd.DataFrame) -> str:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise DataUnavailable(f"No label column found (looked for {LABEL_CANDIDATES}); columns={list(df.columns)[:8]}...")


def load_ciciot2023(input_dir: str | Path, max_rows_per_class_per_file: int, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    """Stratified per-file subsample of the official CSVs. Each file keeps at
    most `max_rows_per_class_per_file` rows of every label, so rare classes
    are retained in full and file (group) boundaries are preserved."""
    d = Path(input_dir)
    files = sorted(list(d.rglob("*.csv")) + list(d.rglob("*.csv.gz"))) if d.exists() else []
    if not files:
        raise DataUnavailable(
            f"No CSV files under '{d}'. Download the CICIoT2023 CSV feature files from the official page "
            "(https://www.unb.ca/cic/datasets/iotdataset-2023.html) and place them there, or pass --demo "
            "for an explicitly labelled toy sanity run.")
    rng = np.random.default_rng(seed)
    parts, manifest = [], []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        lab = find_label_column(df)
        df = df.rename(columns={lab: "label"})
        manifest.append({"file": str(f.relative_to(d)), "bytes": f.stat().st_size, "sha256": sha256_file(f),
                         "rows": len(df), "labels": int(df["label"].nunique())})
        keep = []
        for _, g in df.groupby("label", sort=True):
            n = min(len(g), max_rows_per_class_per_file)
            keep.append(g.iloc[np.sort(rng.choice(len(g), size=n, replace=False))])
        sub = pd.concat(keep)
        sub[GROUP_COL] = str(f.relative_to(d))
        parts.append(sub)
    return pd.concat(parts, ignore_index=True), manifest


def make_toy(n: int = 6000, seed: int = 0, n_files: int = 8) -> pd.DataFrame:
    """TOY SANITY DATA. Two overlapping Gaussian blobs with `toy_` columns."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.7).astype(int)
    X = rng.normal(0, 1, size=(n, 10)) + y[:, None] * np.linspace(0.2, 1.2, 10)[None, :]
    df = pd.DataFrame(X, columns=[f"toy_f{i}" for i in range(10)])
    df.loc[rng.choice(n, 30, replace=False), "toy_f3"] = np.nan
    df["label"] = np.where(y == 1, "ToyAttack", "BenignTraffic")
    df[GROUP_COL] = [f"toy_part_{i % n_files}.csv" for i in range(n)]
    return df


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Numeric, non-leaky predictor columns; returns (features, dropped)."""
    feats, dropped = [], []
    for c in df.columns:
        if c in ("label", "y", "family", GROUP_COL):
            continue
        if (any(p in c.lower() for p in LEAKY_PATTERNS) or c.lower() in LEAKY_EXACT or c.lower().endswith("_id")
                or not pd.api.types.is_numeric_dtype(df[c])):
            dropped.append(c)
        else:
            feats.append(c)
    return feats, dropped


def validate_and_report(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["family"] = df["label"].map(family_of)
    df["y"] = (df["family"] != "benign").astype(int)
    feats, dropped = feature_columns(df)
    X = df[feats].replace([np.inf, -np.inf], np.nan)
    n_inf = int(np.isinf(df[feats].to_numpy(dtype=float, na_value=np.nan)).sum())
    df[feats] = X
    dup_mask = df.duplicated(subset=feats + ["label"], keep="first")
    report = {
        "rows_before_dedup": int(len(df)), "exact_duplicates_removed": int(dup_mask.sum()),
        "n_features": len(feats), "features": feats, "dropped_columns": dropped,
        "categorical_features": [c for c in dropped if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])],
        "missing_values_total": int(X.isna().sum().sum()), "inf_values_replaced": n_inf,
        "label_distribution": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "family_distribution": {str(k): int(v) for k, v in df["family"].value_counts().items()},
        "binary_distribution": {"benign": int((df["y"] == 0).sum()), "attack": int((df["y"] == 1).sum())},
        "unmapped_labels": sorted(df.loc[df["family"] == "UNMAPPED", "label"].astype(str).unique().tolist()),
        "n_groups": int(df[GROUP_COL].nunique()) if GROUP_COL in df else 0,
        "group_identifier": "source CSV file (no capture/session/device id in the flow-feature CSVs)",
    }
    df = df[~dup_mask].reset_index(drop=True)
    report["rows_after_dedup"] = int(len(df))
    return df, report


def split(df: pd.DataFrame, test_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    feats, _ = feature_columns(df)
    info: dict = {}
    n_groups = df[GROUP_COL].nunique() if GROUP_COL in df else 0
    if n_groups >= 5:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        tr_idx, te_idx = next(gss.split(df, df["y"], groups=df[GROUP_COL]))
        info["method"] = "group holdout by source file"
        info["residual_leakage_risk"] = ("Files are the only group identifier. If the official CSV parts were shuffled "
                                         "across captures, flows from one capture can sit on both sides of the split.")
    else:
        tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=test_fraction, random_state=seed, stratify=df["y"])
        info["method"] = "stratified random holdout (fewer than 5 groups)"
        info["residual_leakage_risk"] = "HIGH: no usable group identifier; near-duplicate flows may straddle the split."
    train, test = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    # leakage checks
    h_tr = set(pd.util.hash_pandas_object(train[feats], index=False).values)
    h_te = pd.util.hash_pandas_object(test[feats], index=False).values
    cross = int(np.isin(h_te, list(h_tr)).sum())
    info.update({
        "train_rows": len(train), "test_rows": len(test),
        "group_overlap": int(len(set(train[GROUP_COL]) & set(test[GROUP_COL]))) if GROUP_COL in df else None,
        "cross_split_identical_feature_rows": cross,
        "train_attack_fraction": float(train["y"].mean()), "test_attack_fraction": float(test["y"].mean()),
        "train_fingerprint": frame_fingerprint(train), "test_fingerprint": frame_fingerprint(test),
    })
    return train, test, info


def file_split(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Official file split: train on one file, test on another. Test rows whose features are identical to a training
    row are removed so no flow is scored that the model saw in training; the count is reported."""
    feats, _ = feature_columns(train)
    h_tr = set(pd.util.hash_pandas_object(train[feats], index=False).values)
    shared = np.isin(pd.util.hash_pandas_object(test[feats], index=False).values, list(h_tr))
    test = test[~shared].reset_index(drop=True)
    info = {"method": "official file split (training file / test file)",
            "residual_leakage_risk": "No capture or device identifiers, so flows from one capture may appear in both files; "
                                     "identical rows across files are removed from the test side and near-duplicates are counted.",
            "train_rows": len(train), "test_rows": len(test), "group_overlap": None,
            "cross_split_identical_feature_rows": 0, "identical_rows_removed_from_test": int(shared.sum()),
            "train_attack_fraction": float(train["y"].mean()), "test_attack_fraction": float(test["y"].mean()),
            "train_fingerprint": frame_fingerprint(train), "test_fingerprint": frame_fingerprint(test)}
    return train.reset_index(drop=True), test, info


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=1, default=str))
