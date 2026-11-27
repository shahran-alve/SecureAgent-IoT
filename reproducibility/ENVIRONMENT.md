# Environments

## Simulation, detection, and analysis

The results in `results/` were produced with Python 3.13.18 on Linux (x86_64), NumPy 2.5.3, pandas 3.0.5,
scikit-learn 1.9.1, XGBoost 3.4.1, Pydantic 2.13.5, Matplotlib 3.11.2, z3-solver 5.4.1, pyroute2 0.9.7, and
nftables 1.0.9.

`RUN_ALL.sh` was then run from an empty virtual environment created from `pyproject.toml`
(`pip install -e '.[dev,xgb]'`), which resolved to the versions in `requirements.lock.txt` (NumPy 2.5.5, pandas 3.1.1).
That run regenerated every result file (log: `reproduction.log`; 379 tests passed). Compared with
`results/SHA256SUMS`, 1,789 of 1,837 files were byte-identical. The other 48 differ only where a rerun is expected to
differ:

- wall-clock timings: the `decision_latency_us` column of the raw run files and the tables and figures built from it,
  `ext_scalability.csv`, and the latency columns of `ext_netns_enforcement.csv` and `ext_netemu_enforcement.csv`;
- values measured on the emulated network (`measured_share` in `ext_netemu_enforcement.csv`);
- the counterexample Z3 reports for a theorem that fails in an ablated configuration (`ext_formal_proofs.csv`); which
  theorems are proved and which fail is identical;
- PDF creation timestamps of the figures (the PNG versions match);
- floating-point rounding in the last digits of six summary tables (relative differences below 1e-9).

It needs no GPU; on a 2-core machine with 8 GB of memory it took 26 minutes, of which the emulated network experiment
is about 10.

## Real-LLM runs

Python 3.11.18, PyTorch 2.14.1 (CUDA 12.6), Transformers 5.24.0, float16, greedy decoding, on Tesla V100-PCIE-32GB
and V100S-PCIE-32GB GPUs. Full package list: `llm_requirements.lock.txt`. Each run records its model revision,
library versions, host, and job identifier in `results/raw/LLM_*environment*.json`. Model revisions as loaded:

- `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28`
- `google/gemma-4-12b-it@43c9d014b231474fa7212dbb48a762566c2b255d`
- `mistralai/Mistral-Nemo-Instruct-2407@04d8a90549d23fc6bd7f642064003592df51e9b3`
- `microsoft/Phi-4-mini-instruct@cfbefacb99257ffa30c83adab238a50856ac3083`
