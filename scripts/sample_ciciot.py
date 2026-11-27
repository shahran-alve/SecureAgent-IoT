"""Shrink a large CICIoT2023 CSV for upload.

Keeps a uniform random sample of at most N rows per label (default 10,000, the cap the
pipeline applies anyway), reading the file in chunks so memory stays small. Writes a
gzip-compressed CSV and prints the SHA-256 of the original file for the data manifest.

Usage:
    python sample_ciciot.py train.csv            # writes train_sampled.csv.gz
    python sample_ciciot.py train.csv --per-label 20000
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--per-label", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunksize", type=int, default=500_000)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    kept = None
    rows = 0
    for chunk in pd.read_csv(a.csv, chunksize=a.chunksize):
        label_col = "label" if "label" in chunk.columns else chunk.columns[-1]
        rows += len(chunk)
        # Random key per row; keeping the smallest keys per label gives a uniform sample.
        chunk["_key"] = rng.random(len(chunk))
        pool = chunk if kept is None else pd.concat([kept, chunk], ignore_index=True)
        kept = pool.sort_values("_key").groupby(label_col, sort=False).head(a.per_label)
        print(f"read {rows:,} rows, keeping {len(kept):,}", flush=True)

    out = a.csv.with_name(a.csv.stem + "_sampled.csv.gz")
    kept.drop(columns="_key").sort_index().to_csv(out, index=False, compression="gzip")
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"original: {a.csv.name}, {a.csv.stat().st_size:,} bytes, {rows:,} rows")
    print(f"original sha256: {sha256(a.csv)}")
    print(kept[label_col].value_counts().to_string())


if __name__ == "__main__":
    main()
