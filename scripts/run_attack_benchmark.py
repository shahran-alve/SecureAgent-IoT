#!/usr/bin/env python
"""Offline prompt-injection benchmark (B1 vs B2 + ablations). TESTBED-SYNTHETIC.
Scripted agents by default; no LLM is called unless --llm is given.

With --llm every finished attempt is appended to <out>/LLM_<model>__<condition>_attack_runs.jsonl as it is produced.
Rerunning the same command resumes: finished attempts are skipped, and the CSV and tables are rebuilt
from the JSONL (see secureagent_iot/llm_harness.py). With --llm --seed S all files get a _seed{S} suffix so
seeds can run as parallel jobs; combine them with scripts/merge_llm_shards.py."""
import argparse
import sys
from pathlib import Path

import pandas as pd

from secureagent_iot.config import load_config, set_global_seed
from secureagent_iot.evaluation import (attack_case_catalogue, attack_paired, run_attack_benchmark, save,
                                        summarize_attacks, summarize_review_queue)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--llm", action="store_true",
                    help="use a real model (LLM_PROVIDER, LLM_MODEL, LLM_PROMPT=base|defended|policy|policy_defended, + API key) as the proposing agent; outputs are prefixed LLM_<model>__<condition>_")
    ap.add_argument("--limit-cases", type=int, default=None, metavar="N",
                    help="SMOKE TEST, --llm only: run just the first N cases of each attack variant; every output file is prefixed SMOKE_")
    a = ap.parse_args(argv)
    if a.limit_cases is not None and (not a.llm or a.limit_cases < 1):
        ap.error("--limit-cases needs --llm and a positive N (it exists only to smoke-test a real model)")
    cfg = load_config(a.config)
    llm = None
    if a.llm:
        from secureagent_iot.llm_adapter import LLMUnavailable, ToolCallingAgent
        try:
            llm = ToolCallingAgent(cfg=cfg)
        except LLMUnavailable as exc:
            print(f"BLOCKED: {exc}", file=sys.stderr)
            return 2
    seeds = [a.seed] if a.seed is not None else list(cfg["seeds"])
    set_global_seed(seeds[0])
    out = Path(a.out)
    tables = out.parent / "tables"

    if llm is not None:
        # Checkpointed and resumable; writes only LLM_-prefixed files (never the scripted results or the catalogue).
        from secureagent_iot.llm_harness import CheckpointMismatch, run_llm_benchmark
        try:
            info = run_llm_benchmark("attack", cfg, seeds, llm, out, limit_cases=a.limit_cases, shard_seed=a.seed)
        except CheckpointMismatch as exc:
            print(f"BLOCKED: {exc}", file=sys.stderr)
            return 2
        print(f"[LLM-backed] attack benchmark: model={llm.model} seeds={seeds} rows={info['rows']} "
              f"(resumed {info['resumed_rows']}) calls_this_run={info['model_calls_this_run']} -> {info['raw_csv']}")
        return 0

    df = run_attack_benchmark(cfg, seeds)
    save(df, out / "attack_runs.csv")
    for name, t in summarize_attacks(df, cfg).items():
        save(t, tables / f"{name}.csv")
    save(attack_paired(df, cfg), tables / "attack_paired.csv")
    save(summarize_review_queue(df[df["injected"]], cfg), tables / "review_queue_attacked.csv")
    save(attack_case_catalogue(), tables / "attack_catalogue.csv")

    sweep = []
    for c in cfg["agent"]["compliance_sweep"]:
        d = run_attack_benchmark(cfg, seeds, compliance=float(c), systems=["B1_unguarded", "B2_guarded"], controls=False)
        save(d, out / f"attack_runs_compliance_{c}.csv")
        v = d[d["reached_agent"]]
        e = v[~v["suppression_scored"]]
        for s, g in e.groupby("system", sort=False):
            gv = v[v["system"] == s]
            sup = gv[gv["suppression_scored"]]
            sweep.append({"evidence_class": "testbed-synthetic", "compliance": c, "system": s,
                          "valid_attempts": len(gv), "proposal_asr": gv["proposal_success"].mean(),
                          "exec_denominator": len(g), "exec_successes": int(g["exec_success"].sum()),
                          "exec_asr": g["exec_success"].mean(),
                          "suppression_rate": sup["suppression_success"].mean()})
    save(pd.DataFrame(sweep), tables / "attack_compliance_sweep.csv")
    inj = df[df["injected"]]
    print(f"[testbed-synthetic] attack benchmark: seeds={seeds} attempts/system={len(inj) // inj['system'].nunique()} "
          f"variants={inj['attack_id'].nunique()} -> {out / 'attack_runs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
