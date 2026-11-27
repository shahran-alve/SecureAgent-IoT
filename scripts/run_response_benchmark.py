#!/usr/bin/env python
"""B0/B1/B2(/B3 + ablations) on identical scenario IDs. TESTBED-SYNTHETIC.

With --llm every finished (system, seed, scenario) row is appended to <out>/LLM_<model>__<condition>_response_runs.jsonl as
it is produced. Rerunning the same command resumes: finished rows are skipped, and the CSV and tables are
rebuilt from the JSONL (see secureagent_iot/llm_harness.py). With --llm --seed S all files get a _seed{S}
suffix so seeds can run as parallel jobs; combine them with scripts/merge_llm_shards.py."""
import argparse
import sys
from pathlib import Path

from secureagent_iot.config import load_config, set_global_seed
from secureagent_iot.evaluation import (response_comparisons, run_response_benchmark, save, summarize_cost,
                                        summarize_response, summarize_review_queue)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--seed", type=int, default=None, help="run a single seed instead of the config seed list")
    ap.add_argument("--detector", action="store_true",
                    help="draw p_attack from the trained Random Forest on held-out flows instead of the parametric detector model; "
                         "outputs are prefixed DET_")
    ap.add_argument("--demo", action="store_true", help="with --detector: use the DEMO_ toy model (SANITY CHECK ONLY)")
    ap.add_argument("--llm", action="store_true",
                    help="use a real model (LLM_PROVIDER, LLM_MODEL, LLM_PROMPT=base|defended|policy|policy_defended, + API key) as the proposing agent; outputs are prefixed LLM_<model>__<condition>_")
    ap.add_argument("--limit-cases", type=int, default=None, metavar="N",
                    help="SMOKE TEST, --llm only: run just the first N scenarios; every output file is prefixed SMOKE_")
    a = ap.parse_args(argv)
    if a.limit_cases is not None and (not a.llm or a.limit_cases < 1):
        ap.error("--limit-cases needs --llm and a positive N (it exists only to smoke-test a real model)")
    cfg = load_config(a.config)
    llm, prefix = None, ""
    if a.llm:
        from secureagent_iot.llm_adapter import LLMUnavailable, ToolCallingAgent
        try:
            llm = ToolCallingAgent(cfg=cfg)
        except LLMUnavailable as exc:
            print(f"BLOCKED: {exc}", file=sys.stderr)
            return 2
    seeds = [a.seed] if a.seed is not None else list(cfg["seeds"])
    set_global_seed(seeds[0])
    sampler = None
    if a.detector:
        import json
        import joblib
        import pandas as pd
        from secureagent_iot.simulator import DetectorSampler
        dp = "DEMO_" if a.demo else ""
        need = [Path(a.out) / "models" / f"{dp}random_forest.joblib", Path("data/processed") / f"{dp}test.csv",
                Path(a.out) / f"{dp}detection_meta.json"]
        if not all(p.exists() for p in need):
            print(f"BLOCKED: trained detector not found ({[str(p) for p in need]}). Run prepare_data.py and train_ids.py first.", file=sys.stderr)
            return 2
        meta = json.loads(need[2].read_text())
        if (meta["evidence_class"] == "toy-sanity") != a.demo:
            print("BLOCKED: detector evidence class does not match --demo flag.", file=sys.stderr)
            return 2
        if a.demo:
            print("*** SANITY CHECK ONLY - NOT RESEARCH RESULTS: toy detector driving the simulator ***")
        sampler = DetectorSampler(joblib.load(need[0]), pd.read_csv(need[1]), meta["features"], meta["evidence_class"])
        prefix = f"{dp}DET_" + prefix
    out = Path(a.out)
    tables = out.parent / "tables"
    if llm is not None:
        # Checkpointed and resumable; writes only LLM_-prefixed files (never the scripted results).
        from secureagent_iot.llm_harness import CheckpointMismatch, run_llm_benchmark
        try:
            info = run_llm_benchmark("response", cfg, seeds, llm, out, prefix_extra=prefix, sampler=sampler,
                                     limit_cases=a.limit_cases, shard_seed=a.seed)
        except CheckpointMismatch as exc:
            print(f"BLOCKED: {exc}", file=sys.stderr)
            return 2
        print(f"[{info['evidence_class']}] response benchmark: seeds={seeds} rows={info['rows']} "
              f"(resumed {info['resumed_rows']}) calls_this_run={info['model_calls_this_run']} -> {info['raw_csv']}")
        return 0
    df = run_response_benchmark(cfg, seeds, sampler=sampler)
    cls = "testbed-synthetic"
    if sampler is not None:
        cls = f"testbed-synthetic scenarios, detector probabilities from {sampler.evidence_class} model"
        df["evidence_class"] = cls
    save(df, out / f"{prefix}response_runs.csv")
    for name, t in (("response_summary", summarize_response(df, cfg)), ("response_cost", summarize_cost(df)),
                    ("response_paired", response_comparisons(df, cfg)), ("review_queue", summarize_review_queue(df, cfg))):
        if sampler is not None and len(t):
            t["evidence_class"] = cls
        save(t, tables / f"{prefix}{name}.csv")
    print(f"[{cls}] response benchmark: seeds={seeds} scenarios={df['scenario_id'].nunique()} "
          f"systems={df['system'].nunique()} rows={len(df)} -> {out / (prefix + 'response_runs.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
