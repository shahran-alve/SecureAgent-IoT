# Data

## CICIoT2023 (not redistributed)

The detector is trained and evaluated on the CICIoT2023 CSV feature files (46 features plus a `label` column).
Download them from the official page, https://www.unb.ca/cic/datasets/iotdataset-2023.html, and place them as follows.

| Role | Path | Bytes | SHA-256 of the files used in the paper |
|---|---|---|---|
| training (per-label sample of the official training file) | `data/raw/CICIoT2023_train/ciciot2023_supplied_train_sampled.csv.gz` | 20,740,272 | `e9f5ffa53876e2c0118beb5844404e11a561dec27493ffeb72683b0b76f9b0d3` |
| test (official test file, 1,176,851 flows) | `data/raw/CICIoT2023/ciciot2023_supplied_test.csv` | 347,958,926 | `15aeb1f47a7bd3e3fc394d18d7b196b3640329abaae65d5220df4def6119bfc6` |
| validation (official validation file, 1,176,851 flows) | `data/raw/CICIoT2023_second/ciciot2023_supplied_validation.csv` | 347,921,865 | `b13775e87ef64c8dd66c910d4585323f99b1209f2508c75a0a0a8d584caf20d8` |

The official training file is too large to use whole. The training sample was drawn with
`scripts/sample_ciciot.py` (at most 10,000 flows per label, seed 42):

```bash
python scripts/sample_ciciot.py <official training CSV>     # writes <name>_sampled.csv.gz next to the input
```

Move the output to the training path above.

`scripts/prepare_data.py` then removes exact duplicates within each file and test rows identical to a training row,
and writes `data/processed/raw_file_checksums.csv` (name, bytes, SHA-256, rows, labels per raw file) and
`data/processed/data_report.json` (class distribution, duplicates removed, label mapping, leakage checks). The two
files from the paper's run are committed, so a new run can be compared with them.

Cite the dataset as: Neto et al., "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT
environment", Sensors 23(13):5941, 2023, doi:10.3390/s23135941. Check the terms on the download page before
redistributing any part of it.

## Generated data

- Incident scenarios are generated in memory by `secureagent_iot.simulator` from seeds. They are synthetic testbed
  conditions, not recorded attacks, and are never written under `data/`.
- `RUN_ALL.sh --demo` writes a toy table to `data/processed/DEMO_*` to check the wiring of the detection pipeline.
  Its outputs are prefixed `DEMO_` and marked SANITY CHECK ONLY — NOT RESEARCH RESULTS.
