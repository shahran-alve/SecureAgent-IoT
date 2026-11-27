#!/usr/bin/env python
"""Merge the per-seed shards of a real-LLM run into the final LLM_<model>__<condition>_* raw CSVs and tables.

Each `--llm --seed S` job writes LLM_<model>__<condition>_{response,attack}_runs_seed{S}.jsonl. This script checks
that the shards of ONE (model, prompt condition) add up to exactly one complete run and only then writes
LLM_<model>__<condition>_{response,attack}_runs.csv, the LLM_<model>__<condition>_* tables and that pair's entry
in LLM_environment.json. Prompt conditions are never merged together: run it once per condition.

It REFUSES (exit 2, nothing written) if any seed's shard is missing, if any key is duplicated, absent or
unexpected, if a shard belongs to another prompt condition, or if the shards were produced with different model
revisions, dtypes, prompt templates, system prompts or configs.

    for c in base defended policy policy_defended; do
      python scripts/merge_llm_shards.py --model Qwen/Qwen2.5-7B-Instruct --condition $c
    done
"""
import argparse
import sys
from pathlib import Path

from secureagent_iot.config import load_config
from secureagent_iot.llm_adapter import PROMPT_CONDITIONS
from secureagent_iot.llm_harness import MergeRefused, merge_shards


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model id exactly as given in LLM_MODEL, e.g. Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--condition", required=True, choices=list(PROMPT_CONDITIONS),
                    help="prompt condition (LLM_PROMPT) of the shards to merge; one condition per invocation")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results/raw", help="directory holding the shards; merged files are written next to them")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds that must all be present (default: the config seed list)")
    ap.add_argument("--benchmark", choices=["both", "response", "attack"], default="both")
    ap.add_argument("--limit-cases", type=int, default=None, metavar="N",
                    help="merge SMOKE_ shards produced with --limit-cases N (output stays SMOKE_-prefixed)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    try:
        seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else [int(x) for x in cfg["seeds"]]
    except ValueError:
        ap.error("--seeds must be comma-separated integers")
    kinds = ("response", "attack") if a.benchmark == "both" else (a.benchmark,)
    try:
        info = merge_shards(cfg, seeds, a.model, a.condition, Path(a.out), kinds, a.limit_cases)
    except (MergeRefused, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    for kind in kinds:
        print(f"merged {kind} [{a.condition}]: seeds={seeds} rows={info[kind + '_rows']} -> {info['files'][kind]}")
    print(f"environment -> {info['environment']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
