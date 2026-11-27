#!/usr/bin/env python
"""Exhaustive proposal-space search: the attacker writes the proposal directly
(a fully hijacked "puppet" agent). TESTBED-SYNTHETIC; no agent or LLM involved."""
import argparse
from pathlib import Path

from secureagent_iot.config import load_config, set_global_seed
from secureagent_iot.evaluation import run_envelope_search, save, summarize_envelope


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    seeds = [a.seed] if a.seed is not None else list(cfg["seeds"])
    set_global_seed(seeds[0])
    summary, env = run_envelope_search(cfg, seeds)
    out = Path(a.out)
    save(env, out / "envelope_authorized_actions.csv")
    save(summary, out.parent / "tables" / "envelope_summary.csv")
    save(summarize_envelope(env), out.parent / "tables" / "envelope_actions.csv")
    b2 = summary.set_index("system").loc["B2_guarded"]
    print(f"[testbed-synthetic] proposal-space search: seeds={seeds} proposals/config={int(b2.proposals)} "
          f"B2 authorized={int(b2.authorized)} of which spec-violating={int(b2.authorized_violating)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
