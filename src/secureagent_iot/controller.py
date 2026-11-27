"""Simulated network controller.

An in-memory store of reversible, expiring egress-restriction rules. It has
no shell, socket, firewall or device access: the only side effect is a Python
dict (optionally dumped to a local JSON file).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .schemas import ACTION_TYPES, ActionOutcome, Asset, ProposedAction


class SimClock:
    """Simulated time in seconds. Never reads the wall clock."""

    def __init__(self, start: float = 0.0):
        self.now = float(start)

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("time cannot go backwards")
        self.now += float(seconds)


@dataclass
class Rule:
    device_id: str
    incident_id: str
    created_at: float
    expires_at: float
    duration_seconds: int


class ControllerError(Exception):
    pass


class SimulatedNetworkController:
    def __init__(
        self,
        registry: Mapping[str, Asset],
        clock: SimClock,
        rng: np.random.Generator | None = None,
        hard_max_ttl: int | None = None,
    ):
        self.registry = registry
        self.clock = clock
        self.rng = rng or np.random.default_rng(0)
        self.hard_max_ttl = hard_max_ttl
        self.rules: dict[str, Rule] = {}
        self.events: list[dict] = []
        self.review_queue: list[dict] = []
        self.notifications: list[dict] = []

    # ---- internals -------------------------------------------------------
    def _latency_ms(self) -> float:
        # SIMULATED action latency (lognormal, median ~20 ms). Not a measurement.
        return float(self.rng.lognormal(mean=3.0, sigma=0.3))

    def _log(self, kind: str, **kw) -> None:
        self.events.append({"t": self.clock.now, "kind": kind, **kw})

    def _refused(self, why: str, **kw) -> ActionOutcome:
        self._log("refused", why=why, **kw)
        return ActionOutcome(executed=False, success=False, service_disruption=False, elapsed_ms=0.0)

    def expire(self) -> list[str]:
        """Remove rules whose TTL has elapsed. Called on every read/write."""
        gone = [d for d, r in self.rules.items() if r.expires_at <= self.clock.now]
        for d in gone:
            r = self.rules.pop(d)
            self._log("expired", device_id=d, incident_id=r.incident_id)
        return gone

    # ---- queries ---------------------------------------------------------
    def is_restricted(self, device_id: str) -> bool:
        self.expire()
        return device_id in self.rules

    def restricted_seconds(self, device_id: str, window: float) -> float:
        """Restriction time GRANTED to a device inside the rolling window (trusted state for the policy)."""
        t0 = self.clock.now - window
        return float(sum(e["duration"] for e in self.events if e["kind"] == "restrict_egress"
                         and e.get("device_id") == device_id and e["t"] >= t0))

    def fleet_restricted_seconds(self, window: float) -> float:
        """Restriction time granted to ANY device inside the rolling window."""
        t0 = self.clock.now - window
        return float(sum(e["duration"] for e in self.events if e["kind"] == "restrict_egress" and e["t"] >= t0))

    # ---- the four allowlisted actions -------------------------------------
    def notify(self, device_id: str, incident_id: str) -> ActionOutcome:
        if device_id not in self.registry:
            return self._refused("unregistered_device", device_id=device_id)
        self.notifications.append({"t": self.clock.now, "device_id": device_id, "incident_id": incident_id})
        self._log("notify", device_id=device_id, incident_id=incident_id)
        return ActionOutcome(executed=True, success=True, service_disruption=False, elapsed_ms=self._latency_ms())

    def request_review(self, device_id: str, incident_id: str, risk_score: float | None = None) -> ActionOutcome:
        if device_id not in self.registry:
            return self._refused("unregistered_device", device_id=device_id)
        self.review_queue.append(
            {"t": self.clock.now, "device_id": device_id, "incident_id": incident_id, "risk_score": risk_score}
        )
        self._log("request_review", device_id=device_id, incident_id=incident_id)
        return ActionOutcome(executed=True, success=True, service_disruption=False, elapsed_ms=self._latency_ms())

    def restrict_egress(self, device_id: str, duration_seconds: int, incident_id: str) -> ActionOutcome:
        self.expire()
        if device_id not in self.registry:
            return self._refused("unregistered_device", device_id=device_id)
        if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, int) or duration_seconds <= 0:
            return self._refused("invalid_duration", device_id=device_id)
        if self.hard_max_ttl is not None and duration_seconds > self.hard_max_ttl:
            return self._refused("ttl_exceeds_hard_max", device_id=device_id)
        now = self.clock.now
        self.rules[device_id] = Rule(device_id, incident_id, now, now + duration_seconds, duration_seconds)
        self._log("restrict_egress", device_id=device_id, incident_id=incident_id, duration=duration_seconds)
        disruption = self.registry[device_id].criticality == "high"
        return ActionOutcome(executed=True, success=True, service_disruption=disruption, elapsed_ms=self._latency_ms())

    def release_restriction(self, device_id: str, incident_id: str) -> ActionOutcome:
        self.expire()
        if device_id not in self.registry:
            return self._refused("unregistered_device", device_id=device_id)
        had = self.rules.pop(device_id, None) is not None
        self._log("release_restriction", device_id=device_id, incident_id=incident_id, had_rule=had)
        return ActionOutcome(
            executed=True, success=had, service_disruption=False, elapsed_ms=self._latency_ms(), rollback_success=had
        )

    def rollback(self, device_id: str) -> bool:
        """Operator rollback of an active rule. Returns True if a rule was removed."""
        had = self.rules.pop(device_id, None) is not None
        self._log("rollback", device_id=device_id, had_rule=had)
        return had

    # ---- dispatch ----------------------------------------------------------
    def execute(self, action: ProposedAction) -> ActionOutcome:
        """Dispatch a *typed* action. Only the four allowlisted names map to
        methods; there is no getattr on agent-supplied strings."""
        return self.execute_raw(action.action_type, action.device_id, action.duration_seconds, action.incident_id)

    def execute_raw(self, action_type: str, device_id: str, duration_seconds: int, incident_id: str) -> ActionOutcome:
        if action_type not in ACTION_TYPES:
            return self._refused("unknown_action", action_type=str(action_type)[:64])
        if action_type == "notify":
            return self.notify(device_id, incident_id)
        if action_type == "request_review":
            return self.request_review(device_id, incident_id)
        if action_type == "restrict_egress":
            return self.restrict_egress(device_id, duration_seconds, incident_id)
        return self.release_restriction(device_id, incident_id)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"now": self.clock.now, "rules": [asdict(r) for r in self.rules.values()], "events": self.events}, indent=1)
        )
