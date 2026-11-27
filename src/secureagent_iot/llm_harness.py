"""Checkpointed, resumable runner for the real-LLM (`--llm`) benchmarks.

A real model is slow (seconds per call) and its jobs get killed, so nothing here
keeps results only in memory:

* every finished attempt is appended to a JSONL checkpoint as soon as it exists
  (one line per row, flushed and fsynced), keyed by
  (system, seed, attack_id, scenario_id, style, injected) for the attack
  benchmark and (system, seed, scenario_id) for the response benchmark;
* on restart the finished keys are skipped. Resume is at WHOLE-ATTEMPT
  granularity: an attempt is one `run_attempt` / `run_clean_once` call, which
  can make several model calls (memory variants call the model twice, a TTL
  denial adds a retry). A half-finished attempt leaves no row and is redone
  from its first model call;
* the final CSVs and tables are always built from the JSONL, never from
  in-memory state, so an interrupted-and-resumed run and an uninterrupted run
  go through exactly the same code path;
* a prompt cache inside the agent (on unless LLM_CACHE=0, one process only) skips regenerating a prompt it has
  already answered; the reused answer is still logged as a call with cache_hit=true, so nothing below changes;
* every model call is logged to a second JSONL (`*_calls.jsonl`): attempt key,
  prompt sha256, full prompt, raw model text, parsed arguments, whether JSON
  extraction and strict schema validation succeeded, tokens, latency. Calls of
  an attempt that never finished are removed from that log on resume, so it
  holds exactly the calls behind the rows in the checkpoint.

`limit_cases=N` (script flag --limit-cases) keeps only the first N cases per
attack variant / the first N scenarios, for smoke tests. Every file such a run
writes is prefixed SMOKE_, and the limit is part of the checkpoint identity, so
a limited run can never be mistaken for, or resumed into, a result.

Prompt conditions: every run measures exactly one prompt condition (LLM_PROMPT:
base, defended, policy, policy_defended). The condition is part of the
checkpoint identity and of every file name, `LLM_<model>__<condition>_...`, so
two conditions can never write the same file, resume into each other or be
merged together.

Injection styles: the attack benchmark runs every case in all four styles of
attacks.STYLES (naive, authority, delimiter, in_policy) and ONE no-injection
control per (system, seed, attack_id, scenario_id), whose style is "none".

Seed sharding: with a single `--seed S` every file gets a `_seed{S}` suffix, so
jobs for different seeds (or models) can run at the same time without ever
writing the same file. `merge_shards` (scripts/merge_llm_shards.py) then builds
the final, un-suffixed LLM_<model>__<condition>_* CSVs and tables from the per-seed JSONLs,
and refuses if a seed is missing or any key is duplicated, absent or unexpected.

The scripted (non-LLM) benchmarks do not use this module.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from .agents import build_systems
from .attacks import CONTROL_STYLE, SECTION_TAG, STYLE_SETS, STYLES, apply_style, build_cases, run_attempt
from .evaluation import (EVIDENCE_CLASS, _attack_rates, attack_paired, response_comparisons, run_clean_once, save,
                         summarize_attacks, summarize_cost, summarize_response, summarize_review_queue)
from .llm_adapter import DEFENCE_SENTENCE, PROMPT_CONDITIONS, PROMPT_TEMPLATE_VERSION, policy_statement, system_prompt
from .schemas import ProposalRejected, parse_proposal
from .simulator import build_registry, build_templates

KEY_FIELDS: dict[str, tuple[str, ...]] = {
    "attack": ("system", "seed", "attack_id", "scenario_id", "style", "injected"),
    "response": ("system", "seed", "scenario_id"),
}
# Systems built by agents.build_systems(cfg, llm=...) and run in LLM mode. B0 makes no model call.
LLM_SYSTEMS: dict[str, list[str]] = {
    "attack": ["B1_unguarded", "B2_guarded"],
    "response": ["B0_rules", "B1_unguarded", "B2_guarded"],
}
# Columns that hold wall-clock measurements: the only columns allowed to differ between two runs of the same key.
WALL_CLOCK_COLUMNS = ("decision_latency_us", "llm_latency_s")


class CheckpointMismatch(RuntimeError):
    """The checkpoint on disk was written by a different run configuration, or is corrupt."""


def model_slug(model: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in model.split("/")[-1])


def llm_evidence_class(provider: str, model: str, condition: str) -> str:
    return f"testbed-synthetic scenarios, LLM-backed agent ({provider}:{model}, prompt={condition})"


def condition_of(llm) -> str:
    condition = getattr(llm, "condition", None)
    if not condition:
        raise ValueError("the LLM agent has no prompt condition (LLM_PROMPT); the harness never assumes one")
    return str(condition)


def env_key(model: str, condition: str) -> str:
    """Key of one (model, prompt condition) entry in an LLM_environment.json file."""
    return f"{model}::{condition}"


def config_sha256(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def jsonable(o: Any, nonfinite: Any = None) -> Any:
    """Total conversion to strict-JSON-safe data: numpy scalars become Python scalars, tuples/sets become
    lists, NaN/Infinity become `nonfinite` (None by default), anything unknown becomes str()."""
    if o is None or isinstance(o, (bool, str, int)):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else (nonfinite(o) if callable(nonfinite) else nonfinite)
    if isinstance(o, dict):
        return {str(k): jsonable(v, nonfinite) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [jsonable(v, nonfinite) for v in o]
    item = getattr(o, "item", None)   # numpy scalar
    if callable(item):
        try:
            return jsonable(item(), nonfinite)
        except Exception:
            pass
    return str(o)


def key_of(row: dict, key_fields: tuple[str, ...]) -> tuple:
    return tuple(row[f] for f in key_fields)


def read_jsonl(path: str | Path) -> tuple[list[dict], int]:
    """All complete, valid lines of a JSONL file and the byte offset where they end. A torn final line (the
    process was killed mid-write) is not an error: it is simply left out. A bad line anywhere else is."""
    data = Path(path).read_bytes()
    good = data.rfind(b"\n") + 1
    out = []
    for n, line in enumerate(data[:good].split(b"\n")[:-1], 1):
        try:
            out.append(json.loads(line))
        except ValueError as exc:
            raise CheckpointMismatch(f"{path}: line {n} is not valid JSON ({exc}); refusing to guess") from exc
    return out, good


class JsonlCheckpoint:
    """Append-only row store. Line 1 is {"_meta": {...}} describing the run; every other line is one finished row."""

    def __init__(self, path: str | Path, key_fields: tuple[str, ...], meta: dict):
        self.path, self.key_fields, self.meta = Path(path), tuple(key_fields), jsonable(meta)
        self.rows: list[dict] = []
        self.done: set[tuple] = set()
        self.resumed = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines, good = read_jsonl(self.path) if self.path.exists() else ([], 0)
        if lines:
            stored = lines[0].get("_meta") if isinstance(lines[0], dict) else None
            if stored != self.meta:
                diff = sorted(k for k in set(stored or {}) | set(self.meta) if (stored or {}).get(k) != self.meta.get(k))
                raise CheckpointMismatch(
                    f"{self.path} was written by a different run (differs in: {diff}). "
                    f"Move it away to start over; it is never overwritten automatically.")
            for row in lines[1:]:
                self._remember(row)
            self.resumed = len(self.rows)
        if self.path.exists() and good != self.path.stat().st_size:
            os.truncate(self.path, good)     # drop the torn final line before appending after it
        self._fh = open(self.path, "a", encoding="utf-8")
        try:    # two jobs on one checkpoint would interleave rows; refuse the second one
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._fh.close()
            raise CheckpointMismatch(f"{self.path} is being written by another process (same model and seed submitted twice?)") from exc
        except OSError:
            pass    # the filesystem does not support locks; per-seed file names are the main protection anyway
        if not lines:
            self._write({"_meta": self.meta})

    def _remember(self, row: dict) -> None:
        k = key_of(row, self.key_fields)
        if k in self.done:
            raise CheckpointMismatch(f"{self.path}: duplicate key {k}")
        self.done.add(k)
        self.rows.append(row)

    def _write(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, allow_nan=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def append(self, row: dict) -> dict:
        """Persist one finished row. What is kept in memory is the JSON round trip of the row, so a resumed
        run and an uninterrupted run hold identical values."""
        stored = json.loads(json.dumps(jsonable(row), allow_nan=False))
        self._remember(stored)
        self._write(stored)
        return stored

    def close(self) -> None:
        self._fh.close()


def load_checkpoint(path: str | Path, key_fields: tuple[str, ...]) -> tuple[dict, list[dict]]:
    """Read-only load: (meta, rows). Raises on a missing header or a duplicate key."""
    lines, _ = read_jsonl(path)
    if not lines or not isinstance(lines[0], dict) or "_meta" not in lines[0]:
        raise CheckpointMismatch(f"{path}: no _meta header line")
    seen: set[tuple] = set()
    for row in lines[1:]:
        k = key_of(row, key_fields)
        if k in seen:
            raise CheckpointMismatch(f"{path}: duplicate key {k}")
        seen.add(k)
    return lines[0]["_meta"], lines[1:]


# ---------------------------------------------------------------------------
# Per-call log
# ---------------------------------------------------------------------------
def _nonfinite_name(x: float) -> str:
    return "NaN" if x != x else ("Infinity" if x > 0 else "-Infinity")


class CallLog:
    """One JSON line per model call. Used as `ToolCallingAgent.call_sink`.

    On open, entries whose attempt key is not in `done` (the attempt was killed before its row was written and
    will be redone) are dropped, as is a torn final line, by rewriting the file atomically."""

    def __init__(self, path: str | Path, key_fields: tuple[str, ...], registry, done: set[tuple]):
        self.path, self.key_fields, self.registry = Path(path), tuple(key_fields), registry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.kept = self.pruned = self.written = 0
        if self.path.exists():
            entries, good = read_jsonl(self.path)
            keep = [e for e in entries if isinstance(e.get("key"), dict) and self._key(e) in done]
            self.kept, self.pruned = len(keep), len(entries) - len(keep)
            if self.pruned or good != self.path.stat().st_size:
                tmp = self.path.with_name(self.path.name + ".tmp")
                with open(tmp, "w", encoding="utf-8") as fh:
                    for e in keep:
                        fh.write(json.dumps(e, allow_nan=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, self.path)
        self._fh = open(self.path, "a", encoding="utf-8")

    def _key(self, entry: dict) -> tuple | None:
        try:
            return key_of(entry["key"], self.key_fields)
        except (KeyError, TypeError):
            return None

    def __call__(self, rec: dict) -> None:
        rec = dict(rec)
        proposal = rec.pop("proposal", None)
        if isinstance(proposal, dict):
            proposal = {k: v for k, v in proposal.items() if k != "memory_note"}   # stripped before validation, as in agents.py
        try:
            parse_proposal(proposal, self.registry)
            rec["schema_valid"], rec["schema_error"] = True, None
        except ProposalRejected as exc:
            rec["schema_valid"], rec["schema_error"] = False, f"{exc.code}: {exc.detail}"[:300]
        except Exception as exc:   # logging must never end a run; an unexpected validator error is recorded as such
            rec["schema_valid"], rec["schema_error"] = False, f"VALIDATOR_RAISED {type(exc).__name__}: {exc}"[:300]
        rec["logged_at_unix"] = round(time.time(), 3)
        self._fh.write(json.dumps(jsonable(rec, _nonfinite_name), allow_nan=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.written += 1

    def close(self) -> None:
        self._fh.close()


def call_log_stats(path: str | Path) -> dict:
    """Counts over a per-call log: how many outputs failed JSON extraction / strict schema validation, and timing."""
    entries, _ = read_jsonl(path)
    n = len(entries)
    lat = sorted(float(e["latency_s"]) for e in entries)
    no_json = sum(e.get("json_extracted") is False for e in entries)
    bad_schema = sum(not e.get("schema_valid") for e in entries)
    gen = [float(e["latency_s"]) for e in entries if not e.get("cache_hit")]
    return {"calls": n, "cache_hits": n - len(gen), "generated": len(gen),
            "latency_s_mean_generated": sum(gen) / len(gen) if gen else float("nan"),
            "json_extraction_failed": no_json, "schema_invalid": bad_schema,
            "schema_invalid_with_json": bad_schema - no_json,
            "json_extraction_failed_share": no_json / n if n else float("nan"),
            "schema_invalid_share": bad_schema / n if n else float("nan"),
            "latency_s_mean": sum(lat) / n if n else float("nan"), "latency_s_median": lat[n // 2] if n else float("nan"),
            "latency_s_max": lat[-1] if n else float("nan"),
            "input_tokens": sum(int(e["input_tokens"]) for e in entries),
            "output_tokens": sum(int(e["output_tokens"]) for e in entries)}


# ---------------------------------------------------------------------------
# Plans: the exact, ordered list of attempts a run consists of
# ---------------------------------------------------------------------------
SMOKE_PREFIX = "SMOKE_"


def check_limit(limit_cases: int | None) -> int | None:
    if limit_cases is None:
        return None
    if isinstance(limit_cases, bool) or not isinstance(limit_cases, int) or limit_cases < 1:
        raise ValueError(f"limit_cases must be a positive integer, got {limit_cases!r}")
    return limit_cases


# Arms of one attack case for one system: the four injection styles, then ONE no-injection control. The control
# does not depend on the style, so it is run once, not once per style.
ATTACK_ARMS: list[tuple[str, bool]] = [(style, True) for style in STYLES] + [(CONTROL_STYLE, False)]


def active_styles() -> tuple[str, ...]:
    """Injection styles of this run: the frozen run-1 set unless LLM_STYLE_SET=heldout (held-out styles)."""
    name = (os.environ.get("LLM_STYLE_SET") or "main").strip()
    if name not in STYLE_SETS:
        raise ValueError(f"LLM_STYLE_SET={name!r}; expected one of {sorted(STYLE_SETS)}")
    return STYLE_SETS[name]


def attack_arms() -> list[tuple[str, bool]]:
    return [(style, True) for style in active_styles()] + [(CONTROL_STYLE, False)]


def styles_sha256(styles: tuple[str, ...] | None = None) -> str:
    """Fingerprint of the style wrappers, part of the attack checkpoint identity: rewording a style invalidates
    checkpoints written with the old wording instead of silently mixing the two."""
    styles = STYLES if styles is None else styles
    probe = [[style, ep, apply_style("{T}", style, ep)] for style in styles for ep in sorted(SECTION_TAG)]
    return hashlib.sha256(json.dumps(probe).encode()).hexdigest()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def frozen_prompts(cfg: dict) -> dict:
    """Every prompt text a real-LLM run depends on, with its sha256: the system prompt of each condition (in the
    form local models get and in the form the tool-calling APIs get) and the injection style wrappers.
    PROMPTS_FROZEN.json is this dict for the default config. `system_prompt_sha256` in a shard's header equals
    conditions[<condition>]["local_json"]["sha256"] (or ["api_tool"] for an API provider), and `styles_sha256`
    in an attack shard's header equals styles["styles_sha256"]."""
    return {
        "prompt_template": PROMPT_TEMPLATE_VERSION,
        "config_sha256": config_sha256(cfg),
        "policy_statement": {"sha256": _sha(policy_statement(cfg)), "text": policy_statement(cfg)},
        "defence_sentence": {"sha256": _sha(DEFENCE_SENTENCE), "text": DEFENCE_SENTENCE},
        "conditions": {c: {form: {"sha256": _sha(system_prompt(c, cfg, out)), "text": system_prompt(c, cfg, out)}
                           for form, out in (("local_json", "json"), ("api_tool", "tool"))} for c in PROMPT_CONDITIONS},
        "styles": {"styles_sha256": styles_sha256(), "placeholder": "{T}",
                   "wrappers": {style: {ep: {"sha256": _sha(apply_style("{T}", style, ep)), "text": apply_style("{T}", style, ep)}
                                        for ep in sorted(SECTION_TAG)} for style in STYLES}},
    }


def attack_plan(cfg: dict, seeds: list[int], limit_cases: int | None = None):
    """(registry, templates, [(key, variant, template)]): the scripted benchmark's case order, and per case and
    system the arms in ATTACK_ARMS. limit_cases=N keeps the first N cases of each attack variant."""
    registry, templates = build_registry(), build_templates()
    cases = build_cases(templates, registry, cfg["spec"])
    if check_limit(limit_cases) is not None:
        taken: dict[str, int] = {}
        kept = []
        for v, t in cases:
            if taken.get(v.attack_id, 0) < limit_cases:
                taken[v.attack_id] = taken.get(v.attack_id, 0) + 1
                kept.append((v, t))
        cases = kept
    plan = [({"system": name, "seed": int(seed), "attack_id": v.attack_id, "scenario_id": t.scenario_id,
              "style": style, "injected": inject}, v, t)
            for seed in seeds for v, t in cases for name in LLM_SYSTEMS["attack"] for style, inject in attack_arms()]
    return registry, templates, plan


def response_plan(cfg: dict, seeds: list[int], limit_cases: int | None = None):
    """(registry, [(key, template)]) in the scripted benchmark's iteration order.
    limit_cases=N keeps the first N scenarios."""
    registry, templates = build_registry(), build_templates()
    if check_limit(limit_cases) is not None:
        templates = templates[:limit_cases]
    plan = [({"system": name, "seed": int(seed), "scenario_id": t.scenario_id}, t)
            for seed in seeds for t in templates for name in LLM_SYSTEMS["response"]]
    return registry, plan


def plan_keys(kind: str, cfg: dict, seeds: list[int], limit_cases: int | None = None) -> list[tuple]:
    plan = attack_plan(cfg, seeds, limit_cases)[2] if kind == "attack" else response_plan(cfg, seeds, limit_cases)[1]
    keys = [key_of(p[0], KEY_FIELDS[kind]) for p in plan]
    if len(set(keys)) != len(keys):
        raise AssertionError(f"{kind} plan has duplicate keys")
    return keys


def ordered_rows(rows: list[dict], kind: str, expected: list[tuple], where: str = "") -> list[dict]:
    """Rows in plan order. Refuses if any expected key is absent, duplicated, or any unexpected key is present."""
    by_key: dict[tuple, dict] = {}
    for r in rows:
        k = key_of(r, KEY_FIELDS[kind])
        if k in by_key:
            raise CheckpointMismatch(f"{where}: duplicate key {k}")
        by_key[k] = r
    missing = [k for k in expected if k not in by_key]
    extra = set(by_key) - set(expected)
    if missing or extra:
        raise CheckpointMismatch(f"{where}: {len(missing)} expected keys absent (first: {missing[:3]}), "
                                 f"{len(extra)} unexpected keys present (first: {sorted(extra, key=str)[:3]})")
    return [by_key[k] for k in expected]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
class _Progress:
    def __init__(self, label: str, total: int, done: int, every_s: float = 60.0):
        self.label, self.total, self.start_done, self.done = label, total, done, done
        self.t0 = self.last = time.monotonic()
        self.every_s = every_s

    def tick(self, llm) -> None:
        self.done += 1
        now = time.monotonic()
        if now - self.last >= self.every_s or self.done == self.total:
            self.last = now
            n = self.done - self.start_done
            rate = (now - self.t0) / max(n, 1)
            print(f"[{self.label}] {self.done}/{self.total} attempts ({n} this run, {rate:.1f} s/attempt, "
                  f"{len(llm.calls)} model calls, ETA {(self.total - self.done) * rate / 60:.1f} min)", flush=True)


def _begin(llm, kind: str, key: dict) -> None:
    begin = getattr(llm, "begin_attempt", None)
    if callable(begin):
        begin({"benchmark": kind, **key})


def run_attack_llm(cfg: dict, seeds: list[int], llm, ckpt: JsonlCheckpoint, limit_cases: int | None = None) -> list[dict]:
    registry, templates, plan = attack_plan(cfg, seeds, limit_cases)
    sysmap = build_systems(cfg, llm=llm)
    condition = condition_of(llm)
    cls = llm_evidence_class(llm.provider, llm.model, condition)
    prog = _Progress("attack", len(plan), sum(key_of(k, KEY_FIELDS["attack"]) in ckpt.done for k, _, _ in plan))
    for key, v, t in plan:
        if key_of(key, KEY_FIELDS["attack"]) in ckpt.done:
            continue
        _begin(llm, "attack", key)
        r = run_attempt(sysmap[key["system"]], v, t, key["seed"], cfg, registry, templates, inject=key["injected"],
                        style=key["style"] if key["injected"] else "naive")
        r["style"] = key["style"]
        r["compliance"] = "measured (real model)"
        r["evidence_class"] = cls
        r["prompt_condition"] = condition
        ckpt.append(r)
        prog.tick(llm)
    return ordered_rows(ckpt.rows, "attack", [key_of(k, KEY_FIELDS["attack"]) for k, _, _ in plan], str(ckpt.path))


def run_response_llm(cfg: dict, seeds: list[int], llm, ckpt: JsonlCheckpoint, sampler=None,
                     limit_cases: int | None = None) -> list[dict]:
    registry, plan = response_plan(cfg, seeds, limit_cases)
    sysmap = build_systems(cfg, llm=llm)
    if list(sysmap) != LLM_SYSTEMS["response"]:
        raise AssertionError(f"LLM system list changed: {list(sysmap)}")
    condition = condition_of(llm)
    cls = llm_evidence_class(llm.provider, llm.model, condition)
    prog = _Progress("response", len(plan), sum(key_of(k, KEY_FIELDS["response"]) in ckpt.done for k, _ in plan))
    for key, t in plan:
        if key_of(key, KEY_FIELDS["response"]) in ckpt.done:
            continue
        _begin(llm, "response", key)
        r = run_clean_once(sysmap[key["system"]], t, key["seed"], cfg, registry, sampler)
        r["evidence_class"] = cls
        r["prompt_condition"] = condition
        ckpt.append(r)
        prog.tick(llm)
    return ordered_rows(ckpt.rows, "response", [key_of(k, KEY_FIELDS["response"]) for k, _ in plan], str(ckpt.path))


# ---------------------------------------------------------------------------
# Outputs (always built from checkpoint rows)
# ---------------------------------------------------------------------------
def build_outputs(kind: str, rows: list[dict], cfg: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """(raw run frame, {table name: frame}) for one benchmark, from checkpoint rows in plan order."""
    df = pd.DataFrame(rows)
    cls = rows[0]["evidence_class"] if rows else ""
    with warnings.catch_warnings():
        # A --limit-cases run can leave a table cell with no rows behind it; numpy then warns "Mean of empty slice"
        # once per cell. The cell is NaN either way, so keep the job log readable.
        warnings.simplefilter("ignore", RuntimeWarning)
        if kind == "attack":
            tables = dict(summarize_attacks(df, cfg))
            for name, keys in (("attack_by_style", ["system", "injected", "style"]),
                               ("attack_by_style_category", ["system", "injected", "style", "category"])):
                tables[name] = pd.DataFrame([{"evidence_class": EVIDENCE_CLASS, **dict(zip(keys, k)), **_attack_rates(d, cfg, (name, *k))}
                                             for k, d in df.groupby(keys, sort=False)])
            tables["attack_paired"] = attack_paired(df, cfg)
            tables["review_queue_attacked"] = summarize_review_queue(df[df["injected"]], cfg)
        else:
            tables = {"response_summary": summarize_response(df, cfg), "response_cost": summarize_cost(df),
                      "response_paired": response_comparisons(df, cfg), "review_queue": summarize_review_queue(df, cfg)}
    for t in tables.values():
        if len(t):
            t["evidence_class"] = cls
    return df, tables


def write_outputs(kind: str, rows: list[dict], cfg: dict, out: Path, prefix: str, suffix: str = "") -> Path:
    df, tables = build_outputs(kind, rows, cfg)
    raw_csv = save(df, out / f"{prefix}{kind}_runs{suffix}.csv")
    for name, t in tables.items():
        save(t, out.parent / "tables" / f"{prefix}{name}{suffix}.csv")
    return raw_csv


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(jsonable(obj), indent=1, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


ENV_NOTE = "Environment of real-LLM runs, written by secureagent_iot.llm_harness (one entry per model and prompt condition)."


def record_environment(path: Path, llm, kind: str, run: dict) -> dict:
    """Record what the model calls ran on: dtype, model revision hash, torch and transformers versions, GPU,
    peak GPU memory, prompt condition and system prompt. The file is
    {"models": {"<model id>::<condition>": {..., "runs": {<benchmark>: {...}}}}}; other entries are kept.
    Called at the start of a run (so versions are on disk even if the job dies) and at the end."""
    env_fn = getattr(llm, "environment", None)
    env = dict(env_fn()) if callable(env_fn) else {"provider": getattr(llm, "provider", None), "model": getattr(llm, "model", None)}
    doc: dict = {}
    if path.exists():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            doc = {}
    if not isinstance(doc.get("models"), dict):
        doc = {"models": {}}
    doc["note"] = ENV_NOTE
    entry = env_key(llm.model, condition_of(llm))
    prev = doc["models"].get(entry) or {}
    runs = dict(prev.get("runs") or {})
    # Per-process facts are repeated under the run, because the top-level entry is rewritten by whichever benchmark
    # process wrote last (response and attack are separate processes, each with its own peak GPU memory).
    per_process = {k: env.get(k) for k in ("hostname", "slurm_job_id", "slurm_array_task_id", "gpu_peak_memory_allocated_gb",
                                           "gpu_peak_memory_reserved_gb", "dtype", "revision") if k in env}
    runs[kind] = {**run, **per_process, "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    doc["models"][entry] = {**env, "runs": runs}
    _atomic_write_json(path, doc)
    return env


def file_prefix(model: str, condition: str, limit_cases: int | None = None, prefix_extra: str = "") -> str:
    """Every file of one (model, prompt condition): [SMOKE_]LLM_<model>__<condition>_"""
    if not condition or not condition.replace("_", "").isalnum():
        raise ValueError(f"bad prompt condition {condition!r}")
    return f"{SMOKE_PREFIX if check_limit(limit_cases) is not None else ''}{prefix_extra}LLM_{model_slug(model)}__{condition}_"


def shard_suffix(shard_seed: int | None) -> str:
    return "" if shard_seed is None else f"_seed{int(shard_seed)}"


def run_llm_benchmark(kind: str, cfg: dict, seeds: list[int], llm, out: str | Path, *, prefix_extra: str = "",
                      sampler=None, limit_cases: int | None = None, shard_seed: int | None = None) -> dict:
    """Run (or resume) one LLM benchmark end to end and write its CSVs/tables. Returns a small summary.
    With limit_cases every output file is prefixed SMOKE_. With shard_seed=S (the script was given --seed S)
    every output file is suffixed _seed{S}; merge the shards with scripts/merge_llm_shards.py."""
    if kind not in KEY_FIELDS:
        raise ValueError(kind)
    if shard_seed is not None and [int(x) for x in seeds] != [int(shard_seed)]:
        raise ValueError(f"shard_seed={shard_seed} but seeds={seeds}")
    out = Path(out)
    suffix = shard_suffix(shard_seed)
    prefix = file_prefix(llm.model, condition_of(llm), limit_cases, prefix_extra)
    prefix_extra = (SMOKE_PREFIX if limit_cases is not None else "") + prefix_extra
    meta = {"benchmark": kind, "key_fields": list(KEY_FIELDS[kind]), "seeds": [int(s) for s in seeds],
            "limit_cases": limit_cases,
            "config_sha256": config_sha256(cfg), "detector_sampler": getattr(sampler, "evidence_class", None),
            **llm.identity()}
    if kind == "attack":
        meta.update({"styles": list(active_styles()), "styles_sha256": styles_sha256(active_styles())})
    ckpt_path = out / f"{prefix}{kind}_runs{suffix}.jsonl"
    ckpt = JsonlCheckpoint(ckpt_path, KEY_FIELDS[kind], meta)
    calls_path = out / f"{prefix}{kind}_calls{suffix}.jsonl"
    calls = CallLog(calls_path, KEY_FIELDS[kind], build_registry(), ckpt.done)
    if ckpt.resumed or calls.pruned:
        print(f"[{kind}] resuming from {ckpt_path}: {ckpt.resumed} finished attempts are skipped; "
              f"{calls.pruned} logged calls of unfinished attempts dropped from {calls_path.name}", flush=True)
    # One environment file per benchmark and seed shard, so no two processes ever share it (not even the response
    # and attack runs of one seed); un-sharded runs write the common file directly.
    env_path = out / (f"{prefix}{kind}_environment{suffix}.json" if suffix else f"{prefix_extra}LLM_environment.json")
    run_info = {"seeds": [int(x) for x in seeds], "limit_cases": limit_cases, "checkpoint": ckpt_path.name, "finished": False,
                "rows": len(ckpt.rows), "resumed_rows": ckpt.resumed}
    record_environment(env_path, llm, kind, run_info)
    calls_before = len(llm.calls)
    llm.call_sink = calls
    try:
        rows = (run_attack_llm(cfg, seeds, llm, ckpt, limit_cases) if kind == "attack"
                else run_response_llm(cfg, seeds, llm, ckpt, sampler, limit_cases))
    finally:
        llm.call_sink = None
        calls.close()
        ckpt.close()
    raw_csv = write_outputs(kind, rows, cfg, out, prefix, suffix)
    stats = call_log_stats(calls_path)
    env = record_environment(env_path, llm, kind, {**run_info, "finished": True, "rows": len(rows),
                                                   "model_calls_logged": stats["calls"], "cache_hits_logged": stats["cache_hits"],
                                                   "model_calls_this_process": len(llm.calls) - calls_before})
    print(f"[{kind}] model calls logged: {stats['calls']} ({stats['generated']} generated, {stats['cache_hits']} from the prompt "
          f"cache); JSON extraction failed: {stats['json_extraction_failed']}; schema-invalid: {stats['schema_invalid']}; "
          f"mean latency {stats['latency_s_mean_generated']:.2f} s per generated call", flush=True)
    print(f"[{kind}] environment -> {env_path}: dtype={env.get('dtype')} revision={env.get('revision')} "
          f"torch={env.get('torch')} transformers={env.get('transformers')} "
          f"gpu_peak_allocated_gb={env.get('gpu_peak_memory_allocated_gb')}", flush=True)
    return {"kind": kind, "rows": len(rows), "resumed_rows": ckpt.resumed, "model_calls_this_run": len(llm.calls) - calls_before,
            "checkpoint": str(ckpt_path), "call_log": str(calls_path), "call_stats": stats, "raw_csv": str(raw_csv),
            "environment": str(env_path), "evidence_class": rows[0]["evidence_class"] if rows else ""}


# ---------------------------------------------------------------------------
# Merging per-seed shards
# ---------------------------------------------------------------------------
class MergeRefused(RuntimeError):
    """The shards do not add up to exactly one complete run; nothing was written."""


# Checkpoint identity fields that must agree across every seed shard of one model (seeds differ by design).
SHARD_IDENTITY_EXCLUDES = ("seeds",)


def collect_shards(kind: str, cfg: dict, seeds: list[int], model: str, condition: str, out: str | Path,
                   limit_cases: int | None = None, prefix_extra: str = "") -> list[dict]:
    """Validate every per-seed checkpoint of one benchmark, model and prompt condition, and return all rows in plan
    order. Raises MergeRefused listing every problem found: a missing seed shard, a duplicated key, an absent key,
    an unexpected key, a shard of another prompt condition, or shards produced with different model revision /
    dtype / prompt template / system prompt / config."""
    out = Path(out)
    prefix = file_prefix(model, condition, limit_cases, prefix_extra)
    seeds = [int(x) for x in seeds]
    problems: list[str] = []
    if len(set(seeds)) != len(seeds):
        raise MergeRefused(f"seed list has duplicates: {seeds}")
    want_cfg = config_sha256(cfg)
    rows: list[dict] = []
    identities: dict[int, dict] = {}
    for seed in seeds:
        path = out / f"{prefix}{kind}_runs{shard_suffix(seed)}.jsonl"
        if not path.exists():
            problems.append(f"seed {seed}: shard missing ({path})")
            continue
        try:
            meta, shard_rows = load_checkpoint(path, KEY_FIELDS[kind])
        except (CheckpointMismatch, KeyError) as exc:
            problems.append(f"seed {seed}: {exc}")
            continue
        for field, want in (("benchmark", kind), ("seeds", [seed]), ("limit_cases", limit_cases), ("model", model),
                            ("prompt_condition", condition), ("config_sha256", want_cfg)):
            if meta.get(field) != want:
                problems.append(f"seed {seed}: {path.name} has {field}={meta.get(field)!r}, expected {want!r}")
        identities[seed] = {k: v for k, v in meta.items() if k not in SHARD_IDENTITY_EXCLUDES}
        strays = sorted({str(r.get("prompt_condition")) for r in shard_rows} - {condition})
        if strays:
            problems.append(f"seed {seed}: {path.name} holds rows of prompt condition(s) {strays}, expected only {condition!r}")
        try:
            rows += ordered_rows(shard_rows, kind, plan_keys(kind, cfg, [seed], limit_cases), where=path.name)
        except CheckpointMismatch as exc:
            problems.append(f"seed {seed}: {exc}")
    if identities:
        first_seed = next(iter(identities))
        for seed, ident in identities.items():
            diff = sorted(k for k in set(ident) | set(identities[first_seed]) if ident.get(k) != identities[first_seed].get(k))
            if diff:
                problems.append(f"seed {seed} and seed {first_seed} were produced differently (differs in: {diff}): "
                                f"{ {k: ident.get(k) for k in diff} } vs { {k: identities[first_seed].get(k) for k in diff} }")
    if not problems:   # belt and braces: the concatenation itself must be exactly the multi-seed plan
        try:
            rows = ordered_rows(rows, kind, plan_keys(kind, cfg, seeds, limit_cases), where=f"merged {kind}")
        except CheckpointMismatch as exc:
            problems.append(str(exc))
    if problems:
        raise MergeRefused(f"{kind}: " + "; ".join(problems))
    return rows


def merge_shards(cfg: dict, seeds: list[int], model: str, condition: str, out: str | Path,
                 kinds: tuple[str, ...] = ("response", "attack"), limit_cases: int | None = None, prefix_extra: str = "") -> dict:
    """Merge all seed shards of ONE (model, prompt condition) into the final LLM_<model>__<condition>_* raw CSVs
    and tables (SMOKE_-prefixed when limit_cases is given). Conditions are never merged together. Everything is
    validated first; if anything is refused, nothing is written."""
    out = Path(out)
    collected, problems = {}, []
    for kind in kinds:
        try:
            collected[kind] = collect_shards(kind, cfg, seeds, model, condition, out, limit_cases, prefix_extra)
        except MergeRefused as exc:
            problems.append(str(exc))
    if problems:
        raise MergeRefused(" | ".join(problems))
    prefix = file_prefix(model, condition, limit_cases, prefix_extra)
    entry = env_key(model, condition)
    info: dict = {"model": model, "condition": condition, "seeds": [int(x) for x in seeds], "files": {}}
    for kind, rows in collected.items():
        info["files"][kind] = str(write_outputs(kind, rows, cfg, out, prefix))
        info[f"{kind}_rows"] = len(rows)
    # Environment: one entry per model in the common file, with every shard's record kept under "shards".
    shards: dict = {}
    for seed in seeds:
        for kind in collected:
            path = out / f"{prefix}{kind}_environment{shard_suffix(seed)}.json"
            if path.exists():
                try:
                    rec = (json.loads(path.read_text(encoding="utf-8")).get("models") or {}).get(entry)
                except ValueError:
                    rec = None
                shards[f"seed{int(seed)}_{kind}"] = rec
    env_path = out / f"{SMOKE_PREFIX if limit_cases is not None else ''}{prefix_extra}LLM_environment.json"
    doc: dict = {}
    if env_path.exists():
        try:
            doc = json.loads(env_path.read_text(encoding="utf-8"))
        except ValueError:
            doc = {}
    if not isinstance(doc.get("models"), dict):
        doc = {"models": {}}
    doc["note"] = ENV_NOTE
    prev = doc["models"].get(entry) or {}
    merged_kinds = list(collected)
    if prev.get("merged_from_seed_shards") == [int(x) for x in seeds]:
        # --benchmark response and --benchmark attack merged in two invocations: keep the other benchmark's records
        shards = {**(prev.get("shards") or {}), **shards}
        merged_kinds = [k for k in KEY_FIELDS if k in set(prev.get("benchmarks_merged") or []) | set(collected)]
    common = {}
    recs = [r for r in shards.values() if isinstance(r, dict)]
    if recs:   # top level: only what every shard agrees on (dtype, revision, library versions, GPU model, ...)
        common = {k: v for k, v in recs[0].items() if k != "runs" and all(r.get(k) == v for r in recs)}
    doc["models"][entry] = {**common, "merged_from_seed_shards": [int(x) for x in seeds], "benchmarks_merged": merged_kinds,
                            "merged_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shards": shards}
    _atomic_write_json(env_path, doc)
    info["environment"] = str(env_path)
    return info
