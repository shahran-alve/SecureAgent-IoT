#!/usr/bin/env python
"""Model preflight for the real-LLM run: which candidate models can be run, and run correctly, on one GPU here.

For each candidate id (CANDIDATES, in order; the first is the pilot model and is required):

  1. Look the repository up on the Hugging Face Hub: does it exist, its licence, whether it is gated (and, if so,
     whether this account can read it), its parameter count, and its chat template. Ids are taken as given; a
     candidate that does not resolve is skipped with the reason, and no other id is tried in its place.
  2. Skip it if it is gated without access, has more than MAX_PARAMS parameters, or has no chat template.
  3. Otherwise download it, and run seed 42 under the policy_defended prompt as a SMOKE_ run (30 response
     scenarios, 1 case per attack variant), twice: prompt cache on, prompt cache off. Each run is the ordinary
     benchmark script in its own process, so what is measured is what the full run will do.
  4. It PASSES if peak GPU memory is under 28 GB, JSON extraction fails in at most 5% of calls, no output reaches
     the token limit, and the cached and uncached rows are identical.

Selection: the pilot plus the first two other passing candidates, at most one per vendor (the part of the id before
the slash). If the pilot fails, nothing is selected.

Writes preflight/<vendor>__<name>.json per candidate (every measurement, pass or fail and why) as soon as that
candidate is done, and preflight/selected.json at the end, always, even if a candidate crashed the preflight.
Run it through scripts/llm_preflight.sbatch on a compute node; never on the head node.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "preflight"
CONDITION, SEED, RESPONSE_LIMIT, ATTACK_LIMIT = "policy_defended", 42, 30, 1
MAX_PARAMS = 13e9
MAX_PEAK_GB, MAX_JSON_FAIL, MIN_FREE_GB = 28.0, 0.05, 30
RUN_TIMEOUT_S = 3 * 3600        # per benchmark process
SMALL_FILE_PATTERNS = ["*.json", "*.txt", "*.model", "*.jinja"]
# The weights transformers loads: model.safetensors or model-00001-of-0000N.safetensors at the repository root.
# Some repositories also carry a second full copy in another layout; that copy is not downloaded.
STANDARD_WEIGHTS = re.compile(r"model(-\d+-of-\d+)?\.safetensors")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def slug_of(model: str) -> str:
    return model.replace("/", "__")


def vendor_of(model: str) -> str:
    return model.split("/")[0].lower()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def say(msg: str) -> None:
    print(f"[preflight {now()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1-2. Hub lookup
# ---------------------------------------------------------------------------
def hub_lookup(model: str) -> tuple[dict, str | None]:
    """(what the Hub says about the repository, reason to skip it or None)."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    from secureagent_iot.llm_adapter import reasoning_switches

    hub: dict = {"repo_id": model, "token_in_environment": bool(os.environ.get("HF_TOKEN"))}
    try:
        info = HfApi().model_info(model, files_metadata=True)
    except RepositoryNotFoundError as exc:
        hub["exists"] = False
        return hub, (f"repository not found on the Hub: '{model}' does not resolve (or is private/gated and invisible to "
                     f"this account). {type(exc).__name__}")
    except Exception as exc:
        hub["exists"] = None
        return hub, f"Hub lookup failed: {type(exc).__name__}: {str(exc)[:300]}"
    siblings = [(s.rfilename, s.size) for s in (info.siblings or [])]
    licence = getattr(info.card_data, "license", None) if info.card_data else None
    licence_tags = [t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("license:")]
    params = info.safetensors.total if getattr(info, "safetensors", None) else None
    hub.update({
        "exists": True, "resolved_id": info.id, "revision": info.sha, "private": info.private, "gated": info.gated,
        "licence": licence or (licence_tags[0] if licence_tags else None), "licence_name": getattr(info.card_data, "license_name", None) if info.card_data else None,
        "pipeline_tag": info.pipeline_tag, "library": info.library_name,
        "parameters": params, "parameters_source": "Hub safetensors metadata" if params else None,
        "parameters_by_dtype": dict(info.safetensors.parameters) if getattr(info, "safetensors", None) else None,
        "weight_files": [n for n, _ in siblings if n.endswith((".safetensors", ".bin", ".gguf"))][:40],
        "n_files": len(siblings),
    })
    standard = [(n, sz or 0) for n, sz in siblings if STANDARD_WEIGHTS.fullmatch(n)]
    chosen = standard or [(n, sz or 0) for n, sz in siblings if n.endswith(".safetensors")]
    hub["download_weight_files"] = [n for n, _ in chosen]
    hub["safetensors_bytes"] = sum(sz for _, sz in chosen)
    hub["weights_not_downloaded"] = [n for n, _ in siblings if n.endswith(".safetensors") and n not in hub["download_weight_files"]]
    names = {n for n, _ in siblings}

    def fetch(name: str) -> str | None:
        return Path(hf_hub_download(model, name, revision=info.sha)).read_text(encoding="utf-8")

    # gating: metadata is public, files are not; try to read one small file
    try:
        config = json.loads(fetch("config.json"))
        hub["access"] = True
    except GatedRepoError as exc:
        hub["access"] = False
        return hub, (f"gated ({info.gated}) and this account has no access" + ("" if hub["token_in_environment"] else
                                                                                  " (no HF_TOKEN in the environment)") + f": {type(exc).__name__}")
    except EntryNotFoundError:
        config, hub["access"] = {}, True
        hub["note_config"] = "no config.json at the repository root"
    except HfHubHTTPError as exc:
        hub["access"] = False
        return hub, f"cannot read files of the repository: {type(exc).__name__}: {str(exc)[:200]}"
    hub["architectures"], hub["model_type"], hub["config_dtype"] = config.get("architectures"), config.get("model_type"), config.get("torch_dtype") or config.get("dtype")

    # parameter count fallback: weight bytes / bytes per parameter of the stored dtype
    if not hub["parameters"] and hub["safetensors_bytes"]:
        per = 2 if str(hub["config_dtype"]) in ("float16", "bfloat16") else 4
        hub["parameters"] = int(hub["safetensors_bytes"] / per)
        hub["parameters_source"] = f"estimated: safetensors bytes / {per} (config dtype {hub['config_dtype']})"
    # chat template: a separate chat_template.jinja (newer repositories) or tokenizer_config.json
    template, source = None, None
    try:
        if "chat_template.jinja" in names:
            template, source = fetch("chat_template.jinja"), "chat_template.jinja"
        if not template and "tokenizer_config.json" in names:
            template, source = json.loads(fetch("tokenizer_config.json")).get("chat_template"), "tokenizer_config.json"
        if not template and "chat_template.json" in names:
            template, source = json.loads(fetch("chat_template.json")).get("chat_template"), "chat_template.json"
    except Exception as exc:
        hub["chat_template_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    if isinstance(template, list):          # several named templates
        template = next((t.get("template") for t in template if t.get("name") == "default"), None) or (template[0].get("template") if template else None)
    hub["has_chat_template"] = bool(template)
    if template:
        hub.update({"chat_template_source": source, "chat_template_chars": len(template),
                    "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                    "chat_template_mentions_system_role": "system" in template,
                    "reasoning_switches": reasoning_switches(template)})
    if not hub["parameters"]:
        return hub, "parameter count not available from the Hub (no safetensors metadata and no safetensors weight files)"
    if hub["parameters"] > MAX_PARAMS:
        return hub, f"over 13B parameters ({hub['parameters'] / 1e9:.2f}B, {hub['parameters_source']})"
    if not template:
        return hub, "no chat template (neither chat_template.jinja nor tokenizer_config.json has one)"
    if not hub["safetensors_bytes"]:
        return hub, "no safetensors weights in the repository (only safetensors weights are downloaded)"
    return hub, None


# ---------------------------------------------------------------------------
# 3. Download and the two smoke runs
# ---------------------------------------------------------------------------
def download(model: str, revision: str, need_bytes: int, weight_files: list[str]) -> tuple[dict, str | None]:
    from huggingface_hub import snapshot_download
    hf_home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    hf_home.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(hf_home).free / 1e9
    rec: dict = {"hf_home": str(hf_home), "free_gb_before": round(free_gb, 1), "weights_gb": round(need_bytes / 1e9, 2)}
    if free_gb < max(MIN_FREE_GB, need_bytes / 1e9 + 10):
        return rec, f"not enough free disk for the download ({free_gb:.0f} GB free, weights {need_bytes / 1e9:.0f} GB, minimum {MIN_FREE_GB} GB)"
    t0 = time.time()
    try:
        # No revision argument on purpose: resolving "main" is what records refs/main in the cache, and the
        # benchmark processes load "main" offline. The revision actually downloaded is recorded and compared.
        path = Path(snapshot_download(model, allow_patterns=SMALL_FILE_PATTERNS + list(weight_files)))
    except Exception as exc:
        return rec, f"download failed: {type(exc).__name__}: {str(exc)[:300]}"
    rec.update({"snapshot": str(path), "revision": path.name, "same_revision_as_hub_lookup": path.name == revision,
                "seconds": round(time.time() - t0, 1),
                "downloaded_gb": round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9, 2)})
    return rec, None


def run_benchmarks(model: str, cache: str, out: Path) -> tuple[dict, str | None]:
    """Both benchmark scripts for one cache setting, each in its own process. (record, failure reason or None)."""
    env = {**os.environ, "LLM_PROVIDER": "hf", "LLM_MODEL": model, "LLM_PROMPT": CONDITION, "LLM_CACHE": cache,
           "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
    env.pop("LLM_REVISION", None)
    rec: dict = {"LLM_CACHE": cache, "out": str(out.relative_to(ROOT)), "processes": {}}
    for bench, limit in (("response", RESPONSE_LIMIT), ("attack", ATTACK_LIMIT)):
        log = out.parent / f"{bench}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "scripts" / f"run_{bench}_benchmark.py"), "--llm", "--seed", str(SEED),
               "--limit-cases", str(limit), "--out", str(out)]
        t0 = time.time()
        try:
            with open(log, "w", encoding="utf-8") as fh:
                code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=RUN_TIMEOUT_S).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
        proc = {"exit": code, "seconds": round(time.time() - t0, 1), "log": str(log.relative_to(ROOT)), "limit_cases": limit}
        rec["processes"][bench] = proc
        if code != 0:
            tail = log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n").splitlines()[-25:]
            proc["log_tail"] = [ln[:300] for ln in tail if ln.strip() and "Loading weights" not in ln]
            last = next((ln for ln in reversed(proc["log_tail"]) if ln.strip()), "")
            return rec, f"{bench} benchmark with LLM_CACHE={cache} exited with {code}: {last[:300]}"
    return rec, None


def measure(model: str, out: Path) -> dict:
    """Everything recorded by one cache setting's two processes."""
    from secureagent_iot.llm_harness import KEY_FIELDS, WALL_CLOCK_COLUMNS, env_key, file_prefix, load_checkpoint, read_jsonl
    prefix = file_prefix(model, CONDITION, 1)       # any limit gives the SMOKE_ prefix
    m: dict = {}
    for bench in ("response", "attack"):
        meta, rows = load_checkpoint(out / f"{prefix}{bench}_runs_seed{SEED}.jsonl", KEY_FIELDS[bench])
        calls = read_jsonl(out / f"{prefix}{bench}_calls_seed{SEED}.jsonl")[0]
        env = json.loads((out / f"{prefix}{bench}_environment_seed{SEED}.json").read_text())["models"][env_key(model, CONDITION)]
        limit = int(env.get("max_new_tokens") or meta.get("max_new_tokens") or 0)
        gen = [c for c in calls if not c.get("cache_hit")]
        lat = [c["latency_s"] for c in gen]
        m[bench] = {
            "rows": len(rows), "finished": bool(env["runs"][bench].get("finished")),
            "calls": len(calls), "generated": len(gen), "cache_hits": len(calls) - len(gen),
            "json_extraction_failed": sum(c.get("json_extracted") is False for c in calls),
            "schema_invalid": sum(not c.get("schema_valid") for c in calls),
            "schema_errors": sorted({str(c.get("schema_error"))[:80] for c in calls if not c.get("schema_valid")})[:8],
            "max_new_tokens": limit, "outputs_at_token_limit": sum(limit and c["output_tokens"] >= limit for c in calls),
            "output_tokens_mean": round(sum(c["output_tokens"] for c in calls) / max(len(calls), 1), 1),
            "output_tokens_max": max((c["output_tokens"] for c in calls), default=0),
            "input_tokens_mean": round(sum(c["input_tokens"] for c in calls) / max(len(calls), 1), 1),
            "ttl_retries": sum(bool(c.get("is_retry")) for c in calls),
            "seconds_per_generated_call": round(sum(lat) / len(lat), 3) if lat else None,
            "gpu_peak_allocated_gb": env.get("gpu_peak_memory_allocated_gb"), "gpu_peak_reserved_gb": env.get("gpu_peak_memory_reserved_gb"),
            "gpu": env.get("gpu_name"), "hostname": env.get("hostname"), "dtype": env.get("dtype"), "revision": env.get("revision"),
            "reasoning_mode": env.get("reasoning_mode"), "prompt_cache": env.get("prompt_cache"),
            "system_prompt_sha256": meta.get("system_prompt_sha256"), "chat_template_kwargs": meta.get("chat_template_kwargs"),
            "example_output": next((c["raw_text"][:600] for c in calls), None),
            "example_failed_output": next((c["raw_text"][:600] for c in calls if c.get("json_extracted") is False), None),
        }
        m[bench]["_rows"] = [{k: v for k, v in r.items() if k not in WALL_CLOCK_COLUMNS} for r in rows]
        m[bench]["_calls"] = [{k: v for k, v in c.items() if k not in ("cache_hit", "latency_s", "logged_at_unix")} for c in calls]
    return m


def judge(on: dict, off: dict) -> tuple[dict, dict, list[str]]:
    """Pass criteria from the measurements of the cache-on and cache-off runs: (criteria, summary, fail reasons)."""
    measured = {"cache_on": on, "cache_off": off}
    peak = max(float(measured[lab][b].get("gpu_peak_reserved_gb") or measured[lab][b].get("gpu_peak_allocated_gb") or 0)
               for lab in measured for b in ("response", "attack"))
    calls = sum(off[b]["calls"] for b in off)
    json_fail = sum(off[b]["json_extraction_failed"] for b in off)
    at_limit = sum(off[b]["outputs_at_token_limit"] for b in off)
    rows_same = all(on[b]["_rows"] == off[b]["_rows"] for b in ("response", "attack"))
    logs_same = all(on[b]["_calls"] == off[b]["_calls"] for b in ("response", "attack"))
    criteria = {
        "peak_gpu_memory_gb": {"value": round(peak, 3), "limit": f"< {MAX_PEAK_GB}", "ok": 0 < peak < MAX_PEAK_GB,
                               "measured_as": "largest torch.cuda.max_memory_reserved over the four processes"},
        "json_failure_rate": {"value": round(json_fail / calls, 4) if calls else None, "failed": json_fail, "calls": calls,
                              "limit": f"<= {MAX_JSON_FAIL}", "ok": bool(calls) and json_fail / calls <= MAX_JSON_FAIL,
                              "measured_on": "the cache-off run, both benchmarks"},
        "outputs_at_token_limit": {"value": at_limit, "limit": "0", "ok": at_limit == 0, "measured_on": "the cache-off run, both benchmarks"},
        "cached_and_uncached_rows_identical": {"value": rows_same, "ok": rows_same, "rows": {b: off[b]["rows"] for b in off},
                                               "call_logs_identical_apart_from_cache_hit_and_latency": logs_same},
    }
    summary = {
        "schema_invalid_rate": round(sum(off[b]["schema_invalid"] for b in off) / calls, 4) if calls else None,
        "seconds_per_generated_call": {b: off[b]["seconds_per_generated_call"] for b in off},
        "cache_hit_rate_with_cache_on": round(sum(on[b]["cache_hits"] for b in on) / max(sum(on[b]["calls"] for b in on), 1), 4),
        "calls_per_run": calls, "ttl_retries": sum(off[b]["ttl_retries"] for b in off),
        "dtype": off["response"]["dtype"], "gpu": off["response"]["gpu"], "revision": off["response"]["revision"],
        "reasoning_mode": off["response"]["reasoning_mode"],
    }
    reasons = [f"{name} = {c['value']} (limit {c.get('limit', 'must be true')})" for name, c in criteria.items() if not c["ok"]]
    return criteria, summary, reasons


def evaluate(model: str) -> dict:
    rec: dict = {"model": model, "vendor": vendor_of(model), "started_utc": now(), "condition": CONDITION, "seed": SEED,
                 "response_limit": RESPONSE_LIMIT, "attack_limit": ATTACK_LIMIT, "skipped": False, "skip_reason": None,
                 "passed": False, "fail_reasons": [], "criteria": {}, "hub": {}, "runs": {}}
    say(f"=== {model}: Hub lookup")
    rec["hub"], skip = hub_lookup(model)
    if skip:
        rec.update({"skipped": True, "skip_reason": skip})
        say(f"SKIP {model}: {skip}")
        return rec
    hub = rec["hub"]
    say(f"{model}: licence {hub['licence']}, gated {hub['gated']}, {hub['parameters'] / 1e9:.2f}B parameters ({hub['parameters_source']}), "
        f"chat template from {hub['chat_template_source']}, reasoning switches {hub['reasoning_switches']['detected'] or 'none'}")
    rec["download"], fail = download(model, hub["revision"], hub["safetensors_bytes"], hub["download_weight_files"])
    if fail:
        rec["fail_reasons"].append(fail)
        say(f"FAIL {model}: {fail}")
        return rec
    say(f"{model}: downloaded {rec['download']['downloaded_gb']} GB in {rec['download']['seconds']} s (revision {rec['download']['revision'][:12]})")
    base = OUT / "runs" / slug_of(model)
    if base.exists():
        shutil.rmtree(base)             # a preflight is always a fresh measurement, never a resume of an older one
    measured: dict = {}
    for label, cache in (("cache_on", "1"), ("cache_off", "0")):
        say(f"{model}: smoke run with LLM_CACHE={cache}")
        run, fail = run_benchmarks(model, cache, base / label / "raw")
        rec["runs"][label] = run
        if fail:
            rec["fail_reasons"].append(fail)
            say(f"FAIL {model}: {fail}")
            return rec
        measured[label] = measure(model, base / label / "raw")
        for bench in ("response", "attack"):
            run[bench] = {k: v for k, v in measured[label][bench].items() if not k.startswith("_")}
    rec["criteria"], rec["summary"], reasons = judge(measured["cache_on"], measured["cache_off"])
    rec["fail_reasons"] += reasons
    rec["passed"] = not rec["fail_reasons"]
    c = rec["criteria"]
    say(f"{'PASS' if rec['passed'] else 'FAIL'} {model}: peak {c['peak_gpu_memory_gb']['value']} GB, JSON failures "
        f"{c['json_failure_rate']['failed']}/{c['json_failure_rate']['calls']}, at token limit {c['outputs_at_token_limit']['value']}, "
        f"rows identical {c['cached_and_uncached_rows_identical']['value']}" + ("" if rec["passed"] else f" -> {rec['fail_reasons']}"))
    return rec


def select(records: list[dict]) -> tuple[list[str], dict[str, str]]:
    """(selected ids, why each candidate was or was not selected): the pilot plus the first two other passing
    candidates, at most one per vendor; nothing if the pilot failed."""
    why: dict[str, str] = {}
    if not records:
        return [], why
    pilot = records[0]
    if not pilot.get("passed"):
        reason = pilot.get("skip_reason") or "; ".join(pilot.get("fail_reasons") or ["did not complete"])
        for r in records:
            why[r["model"]] = f"nothing selected: the pilot model {pilot['model']} did not pass ({reason})"
        return [], why
    selected, vendors = [pilot["model"]], {pilot["vendor"]}
    why[pilot["model"]] = "selected: pilot model, passed"
    for r in records[1:]:
        if r.get("skipped"):
            why[r["model"]] = f"not selected: skipped ({r['skip_reason']})"
        elif not r.get("passed"):
            why[r["model"]] = "not selected: failed (" + "; ".join(r.get("fail_reasons") or ["did not complete"]) + ")"
        elif r["vendor"] in vendors:
            why[r["model"]] = f"not selected: passed, but a model of vendor '{r['vendor']}' is already selected"
        elif len(selected) >= 3:
            why[r["model"]] = "not selected: passed, but the pilot and two other passing candidates were already selected"
        else:
            selected.append(r["model"])
            vendors.add(r["vendor"])
            why[r["model"]] = "selected: passed"
    return selected, why


def main() -> int:
    candidates = os.environ.get("CANDIDATES", "").split()
    if not candidates:
        print("BLOCKED: CANDIDATES is empty (source scripts/llm_candidates.sh)", file=sys.stderr)
        return 2
    OUT.mkdir(exist_ok=True)
    stale = OUT / "selected.json"
    if stale.exists():                      # never leave an older selection in place while a new preflight is running
        os.replace(stale, OUT / f"selected.previous.{time.strftime('%Y%m%dT%H%M%S')}.json")
    say(f"candidates: {candidates}; HF_TOKEN in environment: {bool(os.environ.get('HF_TOKEN'))}; job {os.environ.get('SLURM_JOB_ID')}")
    records: list[dict] = []
    try:
        for model in candidates:
            try:
                rec = evaluate(model)
            except Exception as exc:        # one candidate must not take the others down with it
                rec = {"model": model, "vendor": vendor_of(model), "skipped": False, "skip_reason": None, "passed": False,
                       "fail_reasons": [f"preflight crashed on this candidate: {type(exc).__name__}: {str(exc)[:300]}"],
                       "traceback": traceback.format_exc()[-3000:]}
                say(f"FAIL {model}: {rec['fail_reasons'][0]}")
            rec["finished_utc"] = now()
            records.append(rec)
            write_json(OUT / f"{slug_of(model)}.json", rec)
    finally:
        selected, why = select(records)
        for model in candidates[len(records):]:
            why[model] = "not selected: the preflight ended before this candidate was evaluated"
        write_json(OUT / "selected.json", {
            "selected": selected, "pilot": candidates[0], "candidates_in_order": candidates,
            "complete": len(records) == len(candidates), "why": why,
            "rule": "pilot plus the first two other passing candidates, at most one per vendor; nothing if the pilot fails",
            "pass_criteria": {"peak_gpu_memory_gb": f"< {MAX_PEAK_GB}", "json_failure_rate": f"<= {MAX_JSON_FAIL}",
                              "outputs_at_token_limit": 0, "cached_and_uncached_rows_identical": True},
            "skip_rules": ["repository does not resolve", "gated without access", f"over {MAX_PARAMS / 1e9:g}B parameters", "no chat template"],
            "smoke": {"condition": CONDITION, "seed": SEED, "response_scenarios": RESPONSE_LIMIT, "attack_cases_per_variant": ATTACK_LIMIT},
            "written_utc": now(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "code_commit": os.environ.get("SAIOT_CODE_COMMIT"),
        })
        say(f"selected: {selected}")
        for model, reason in why.items():
            say(f"  {model}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
