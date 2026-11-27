"""A scripted FAKE text backend for harness tests. It is not an LLM, needs no network, GPU or weights, and
produces no reported result. It stands in for `llm_adapter.HFLocalModel`: it receives the user prompt and
returns raw TEXT, so the adapter's real JSON extraction, the strict schema and the policy all run on it."""
from __future__ import annotations

import hashlib
import json
import re

from secureagent_iot.agents import parse_directives

BEHAVIOURS = ("valid", "dur_text", "dur_null", "dur_float", "device_list", "action_list", "extra_key", "no_json", "obey")
_INCIDENT = re.compile(r"<incident id='([^']*)' device='([^']*)' p_attack='([^']*)'[^>]*/>")


class SimulatedKill(BaseException):
    """Raised by the fake in place of SIGKILL. BaseException so that no `except Exception` can swallow it."""


def _section(prompt: str, tag: str) -> str:
    """Text of every <tag ...>...</tag> section, joined (the delimiter style closes and reopens a section)."""
    return " ".join(m.group(1) for m in re.finditer(rf"<{tag}[^>]*>(.*?)</{tag}>", prompt, re.S)).strip()


class FakeBackend:
    """select="turn": emit the nine BEHAVIOURS in turn, one per call (stateful).
    select="prompt": pick the behaviour from sha256(prompt), so output is a pure function of the prompt
    (needed to compare an interrupted, a sharded and an uninterrupted run).
    only=<behaviour>: always that behaviour.  kill_at=N: raise SimulatedKill on the N-th call (1-based)."""
    model_id = "fake/Fake-7B-Instruct"
    model_version = "fake/Fake-7B-Instruct@test-double"

    def __init__(self, select: str = "turn", only: str | None = None, kill_at: int | None = None):
        assert select in ("turn", "prompt") and (only is None or only in BEHAVIOURS)
        self.select, self.only, self.kill_at = select, only, kill_at
        self.n_calls = 0
        self.emitted: list[str] = []
        self.systems: set[str] = set()

    def environment(self) -> dict:
        return {"backend": "FAKE test double (no model was run)", "dtype": "none", "revision": "test-double",
                "torch": "n/a", "transformers": "n/a"}

    def identity(self) -> dict:
        return {"model_version": self.model_version, "dtype": "none"}

    def _behaviour(self, prompt: str) -> str:
        if self.only:
            return self.only
        if self.select == "turn":
            return BEHAVIOURS[(self.n_calls - 1) % len(BEHAVIOURS)]
        return BEHAVIOURS[int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % len(BEHAVIOURS)]

    def render(self, system: str, prompt: str) -> str:
        return f"<fake-system>{system}</fake-system>\n{prompt}"

    def generation_settings(self) -> dict:
        return {"model_version": self.model_version, "max_new_tokens": 256}

    def generate(self, system: str, prompt: str) -> dict:
        """`system` is the system prompt of the run's prompt condition, `prompt` the user message. The fake's
        behaviour depends on the user message only, so the same incident gets the same answer in every condition."""
        self.n_calls += 1
        self.systems.add(system)
        if self.kill_at is not None and self.n_calls == self.kill_at:
            raise SimulatedKill(f"simulated kill at model call {self.n_calls}")
        b = self._behaviour(prompt)
        self.emitted.append(b)
        text = self.respond(b, prompt)
        return {"text": text, "prompt": self.render(system, prompt), "input_tokens": len(prompt.split()),
                "output_tokens": len(text.split())}

    @staticmethod
    def respond(b: str, prompt: str) -> str:
        m = _INCIDENT.search(prompt)
        _, dev, p = m.groups()
        p = float(p)
        refs = [o["evidence_id"] for o in json.loads(_section(prompt, "observations") or "[]")]
        # On a very confident alert it asks for a long TTL, so the guarded system denies it (TTL cap) and asks
        # again: that retry is a second model call inside one attempt. After feedback it complies.
        long_ttl = p >= 0.9 and "<policy_feedback>" not in prompt
        action, dur = ("restrict_egress", 900 if long_ttl else 300) if p >= 0.8 else (("request_review", 0) if p >= 0.5 else ("notify", 0))
        out = {"device_id": dev, "action_type": action, "duration_seconds": dur, "evidence_refs": refs,
               "reason": f"fake:{b}"}
        if b == "valid":
            out["memory_note"] = f"saw activity on {dev}"
        elif b == "dur_text":
            out.update(action_type="restrict_egress", duration_seconds="300 seconds")
        elif b == "dur_null":
            out.update(action_type="restrict_egress", duration_seconds=None)
        elif b == "dur_float":
            out.update(action_type="restrict_egress", duration_seconds=300.5)
        elif b == "device_list":
            out["device_id"] = [dev]
        elif b == "action_list":
            out["action_type"] = ["restrict_egress", "notify"]
        elif b == "extra_key":
            out["approved"] = True
        elif b == "no_json":
            return "I'm sorry, but I can't decide this one without more context."
        elif b == "obey":
            notes = json.loads(_section(prompt, "agent_notes_from_earlier_incidents") or "[]")
            ds = []
            for src, text in [("memory", n) for n in notes] + [("untrusted_text", _section(prompt, "untrusted_telemetry")),
                                                               ("retrieved_text", _section(prompt, "retrieved_reference")),
                                                               ("fake_trusted", _section(prompt, "trusted_instruction"))]:
                for d in parse_directives(text, src):
                    if d.kind.startswith("standing_") and src != "memory":
                        out["memory_note"] = text          # plant the standing instruction for a later incident
                    else:
                        ds.append(d)
            for d in ds:
                if d.kind == "isolate":
                    out.update(action_type="restrict_egress", device_id=d.device, duration_seconds=d.duration or 300,
                               evidence_refs=refs + list(d.evidence_ids))
                elif d.kind == "release":
                    out.update(action_type="release_restriction", device_id=d.device, duration_seconds=0)
                elif d.kind == "unknown_tool":
                    out.update(action_type=d.tool, device_id=d.device or dev, duration_seconds=0)
                elif d.kind == "standing_notify":
                    out.update(action_type="notify", duration_seconds=0)
                elif d.kind == "standing_release" and d.device == dev:
                    out.update(action_type="release_restriction", duration_seconds=0)
                else:
                    continue
                break
        return "Here is my decision:\n```json\n" + json.dumps(out) + "\n```"
