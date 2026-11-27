#!/usr/bin/env python
"""Write or check PROMPTS_FROZEN.json: the sha256 (and text) of each prompt condition's system prompt and of the
injection style wrappers, for the default config.

    python scripts/freeze_prompts.py            # write PROMPTS_FROZEN.json
    python scripts/freeze_prompts.py --check    # exit 1 if the code no longer produces the frozen prompts

Every real-LLM job runs --check before it starts, so a prompt edited after the freeze stops the run instead of
silently mixing two wordings.
"""
import argparse
import json
import sys
from pathlib import Path

from secureagent_iot.config import load_config
from secureagent_iot.llm_harness import frozen_prompts

PATH = Path(__file__).resolve().parents[1] / "PROMPTS_FROZEN.json"


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        out.update(flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {prefix + k: v})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--path", default=str(PATH))
    a = ap.parse_args(argv)
    now = frozen_prompts(load_config(a.config))
    path = Path(a.path)
    if not a.check:
        path.write_text(json.dumps(now, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path}")
        for c, forms in now["conditions"].items():
            print(f"  {c:16s} local_json {forms['local_json']['sha256']}  api_tool {forms['api_tool']['sha256']}")
        print(f"  styles           {now['styles']['styles_sha256']}")
        return 0
    if not path.exists():
        print(f"PROMPTS NOT FROZEN: {path} does not exist", file=sys.stderr)
        return 1
    a_, b_ = flat(json.loads(path.read_text(encoding="utf-8"))), flat(now)
    diff = sorted(k for k in set(a_) | set(b_) if a_.get(k) != b_.get(k))
    if diff:
        print(f"PROMPTS CHANGED since the freeze ({path.name}); differing entries: {diff[:12]}", file=sys.stderr)
        return 1
    print(f"prompts match {path.name} (template {now['prompt_template']}, {len(now['conditions'])} conditions, "
          f"styles {now['styles']['styles_sha256'][:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
