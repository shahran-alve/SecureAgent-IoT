"""`secureagent-iot` command line: thin wrapper over the scripts."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMANDS = {
    "prepare-data": "prepare_data.py", "train-ids": "train_ids.py", "response": "run_response_benchmark.py",
    "attack": "run_attack_benchmark.py", "figures": "make_figures.py", "envelope": "run_envelope_search.py",
}


def _run_script(name: str, argv: list[str]) -> int:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return int(mod.main(argv) or 0)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="secureagent-iot", description="SecureAgent-IoT (simulation-only research artifact)")
    ap.add_argument("command", choices=list(COMMANDS) + ["verify-audit", "scenarios"])
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.command in COMMANDS:
        return _run_script(COMMANDS[a.command], a.rest)
    if a.command == "verify-audit":
        from .audit import AuditLog
        if not a.rest:
            print("usage: secureagent-iot verify-audit <audit.jsonl>", file=sys.stderr)
            return 2
        ok, bad = AuditLog(a.rest[0]).verify()
        print(json.dumps({"path": a.rest[0], "chain_intact": ok, "first_bad_record": bad}))
        return 0 if ok else 1
    from .simulator import build_templates
    for t in build_templates():
        print(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
