"""Independent, deterministic, default-deny policy engine.

`evaluate` is a function of the action and trusted state; its only side effect
is consuming a single-use approval. The engine sees only (a) a schema-validated ProposedAction and (b) trusted
state that the attacker cannot write: the asset registry, the recorded
incident (alert + observation IDs), the approval store and the controller's
rule state. It never reads untrusted telemetry text, agent reasoning or
retrieved tool descriptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from .schemas import Asset, PolicyDecision, ProposedAction

IMPACT = {"low": 0.2, "medium": 0.5, "high": 1.0}


@dataclass
class IncidentRecord:
    incident_id: str
    device_id: str
    p_attack: float
    evidence_ids: frozenset[str]
    spread: float = 0.0          # fraction of observations indicating lateral activity, in [0,1]
    resolved: bool = False
    evidence_kinds: dict = field(default_factory=dict)    # evidence_id -> kind of observation
    evidence_sources: dict = field(default_factory=dict)  # evidence_id -> sensor/collector that produced it
    evidence_time: dict = field(default_factory=dict)     # evidence_id -> time the observation was recorded
    indicator: str = ""          # external indicator (e.g. a destination) seen on this device by a trusted flow collector

    def independent(self, refs, now: float | None = None, max_age: float | None = None) -> int:
        """Independent, fresh observations among `refs`. Two observations count as two only if they differ
        in kind AND come from different sources, so one compromised collector reporting several kinds counts
        once. Observations older than `max_age` do not count."""
        refs = set(refs) & self.evidence_ids
        if now is not None and max_age is not None:
            refs = {r for r in refs if now - self.evidence_time.get(r, now) <= max_age}
        kinds = {self.evidence_kinds.get(r, r) for r in refs}
        sources = {self.evidence_sources.get(r, r) for r in refs}
        return min(len(kinds), len(sources))


@dataclass
class TrustedState:
    registry: Mapping[str, Asset]
    incidents: dict[str, IncidentRecord] = field(default_factory=dict)
    approvals: set[tuple[str, str, str]] = field(default_factory=set)  # (incident_id, device_id, action_type)
    is_restricted: Callable[[str], bool] = lambda device_id: False
    now: Callable[[], float] = lambda: 0.0
    restricted_seconds: Callable[[str, float], float] = lambda device_id, window: 0.0
    fleet_restricted_seconds: Callable[[float], float] = lambda window: 0.0
    # number of OTHER registered devices on which trusted collectors saw `indicator` inside the window
    corroborating_devices: Callable[[str, str, float], int] = lambda indicator, device_id, window: 0
    approval_time: dict = field(default_factory=dict)
    consumed: set = field(default_factory=set)
    # ledger of restrictions granted under the corroboration exemption: (time granted, duration)
    exempt_log: list = field(default_factory=list)

    def exempt_seconds(self, window: float) -> float:
        t0 = self.now() - window
        return float(sum(d for t, d in self.exempt_log if t >= t0))

    def record_incident(self, rec: IncidentRecord) -> None:
        self.incidents[rec.incident_id] = rec

    def approve(self, incident_id: str, device_id: str, action_type: str) -> None:
        """Simulated human approval, bound to one incident, device and action. Only test/harness code calls this."""
        key = (incident_id, device_id, action_type)
        self.approvals.add(key)
        self.approval_time[key] = self.now()

    def approval_valid(self, key, max_age: float) -> bool:
        return key in self.approvals and (self.now() - self.approval_time.get(key, self.now())) <= max_age

    def consume(self, key) -> None:
        """Approvals are single-use: once an action has been authorized on one, it cannot be replayed."""
        if key in self.approvals:
            self.approvals.discard(key)
            self.consumed.add(key)

    def all_approvals(self) -> set:
        return self.approvals | self.consumed


@dataclass(frozen=True)
class PolicyFlags:
    """Ablation switches. All True = the proposed system."""
    provenance: bool = True
    criticality: bool = True
    ttl_cap: bool = True
    escalation_floor: bool = True
    budget: bool = True
    thresholds: bool = True       # detector-confidence and evidence-count thresholds (off only in the generic baseline)


class PolicyEngine:
    def __init__(self, cfg: dict, state: TrustedState, flags: PolicyFlags | None = None):
        p = cfg["policy"]
        self.version = str(p["version"])
        self.max_ttl = int(p["max_ttl_seconds"])
        self.p_threshold = float(p["p_attack_threshold"])
        self.review_floor = float(p["review_floor"])
        self.min_evidence = int(p["min_evidence"])
        self.known_agents = frozenset(p["known_agents"])
        self.weights = dict(p["risk_weights"])
        self.approval_max_age = float(p.get("approval_max_age_seconds", 900))
        self.device_budget = float(p.get("device_budget_seconds", 900))
        self.budget_window = float(p.get("budget_window_seconds", 3600))
        self.budget_by_crit = {k: float(v) for k, v in (p.get("device_budget_by_criticality") or {}).items()}
        self.evidence_max_age = float(p.get("evidence_max_age_seconds", float("inf")))
        self.fleet_budget = float(p.get("fleet_budget_seconds") or 0)   # 0 = no fleet-wide cap
        # corroboration-aware fleet budget: a restriction whose indicator was independently seen on at least this many
        # other devices does not draw on the fleet budget (0 = off). The per-device budget still applies.
        self.corroboration_min = int(p.get("fleet_corroboration_devices") or 0)
        # bounded exemption: corroboration-exempt restrictions draw on their own budget B_c (0 = unbounded exemption)
        self.corroboration_budget = float(p.get("corroboration_budget_seconds") or 0)
        # hard ceilings: apply to every restriction, approved or corroborated included; nothing overrides them (0 = off)
        self.device_hard_cap = float(p.get("device_hard_cap_seconds") or 0)
        self.fleet_hard_cap = float(p.get("fleet_hard_cap_seconds") or 0)
        self.state = state
        self.flags = flags or PolicyFlags()

    # ---- risk ranking (ordering only; never authorizes) -------------------
    def risk_score(self, inc: IncidentRecord, asset: Asset) -> float:
        t = min(max(inc.p_attack, 0.0), 1.0)
        u = 1.0 - abs(2.0 * t - 1.0)
        i = IMPACT[asset.criticality]
        s = min(max(inc.spread, 0.0), 1.0)
        w = self.weights
        total = w["threat"] + w["uncertainty"] + w["impact"] + w["spread"]
        return (w["threat"] * t + w["uncertainty"] * u + w["impact"] * i + w["spread"] * s) / total

    def _d(self, allowed: bool, approval: bool, code: str, risk: float | None = None) -> PolicyDecision:
        return PolicyDecision(allowed=allowed, requires_approval=approval, reason_code=code,
                              policy_version=self.version, risk_score=risk)

    # ---- decision -----------------------------------------------------------
    def evaluate(self, action: ProposedAction) -> PolicyDecision:
        st, fl = self.state, self.flags

        # 1. identity and registration (default deny)
        if action.agent_id not in self.known_agents:
            return self._d(False, False, "DENY_UNKNOWN_AGENT")
        asset = st.registry.get(action.device_id)
        if asset is None:
            return self._d(False, False, "DENY_UNREGISTERED_DEVICE")
        inc = st.incidents.get(action.incident_id)
        if inc is None:
            return self._d(False, False, "DENY_UNKNOWN_INCIDENT")
        incident_asset = st.registry.get(inc.device_id, asset)
        risk = self.risk_score(inc, incident_asset)

        # 2. provenance: device binding + evidence refs must resolve to recorded observations
        refs = set(action.evidence_refs)
        if fl.provenance:
            if action.device_id != inc.device_id:
                return self._d(False, False, "DENY_DEVICE_MISMATCH", risk)
            if not refs <= inc.evidence_ids:
                return self._d(False, False, "DENY_UNRESOLVED_EVIDENCE", risk)
            n_evidence = inc.independent(refs, st.now(), self.evidence_max_age)   # independent, fresh sources
        else:
            n_evidence = len(refs)

        # 3. asset allowlist / privilege scope
        if action.action_type not in asset.allowed_actions:
            return self._d(False, False, "DENY_ACTION_NOT_ALLOWLISTED", risk)

        key = (action.incident_id, action.device_id, action.action_type)
        approved = st.approval_valid(key, self.approval_max_age)   # bound to incident+device+action, not expired

        if action.action_type == "request_review":
            return self._d(True, False, "ALLOW_REVIEW", risk)

        if action.action_type == "notify":
            if fl.escalation_floor and inc.p_attack >= self.p_threshold and inc.independent(inc.evidence_ids, st.now(), self.evidence_max_age) >= self.min_evidence:
                # agent under-responded to a well-evidenced, high-confidence alert
                return self._d(False, True, "REVIEW_ESCALATION_FLOOR", risk)
            return self._d(True, False, "ALLOW_NOTIFY", risk)

        if action.action_type == "release_restriction":
            if not st.is_restricted(action.device_id):     # nothing to release (found by SMT proof)
                return self._d(False, False, "DENY_NOTHING_TO_RELEASE", risk)
            if inc.resolved or approved:
                if approved and not inc.resolved:
                    st.consume(key)
                return self._d(True, False, "ALLOW_RELEASE", risk)
            return self._d(False, True, "REVIEW_RELEASE_UNRESOLVED", risk)

        # ---- restrict_egress ---------------------------------------------
        if action.duration_seconds <= 0:
            return self._d(False, False, "DENY_INVALID_DURATION", risk)
        if fl.ttl_cap and action.duration_seconds > self.max_ttl:
            return self._d(False, False, "DENY_TTL_EXCEEDS_CAP", risk)
        # 4. confidence and evidence thresholds (trusted alert, not agent belief). They apply to every
        #    restriction: a human approval is an additional requirement, never a substitute (found by SMT proof).
        if fl.thresholds:
            if inc.p_attack < self.review_floor:
                return self._d(False, False, "DENY_BELOW_THRESHOLD", risk)
            if inc.p_attack < self.p_threshold:
                return self._d(False, True, "REVIEW_UNCERTAIN", risk)
            if n_evidence < self.min_evidence:
                return self._d(False, True, "REVIEW_INSUFFICIENT_EVIDENCE", risk)
        # 4b. hard ceilings: no approval and no corroboration can push a device or the fleet past them
        if self.device_hard_cap > 0 and st.restricted_seconds(action.device_id, self.budget_window) + action.duration_seconds > self.device_hard_cap:
            return self._d(False, False, "DENY_HARD_CEILING", risk)
        if self.fleet_hard_cap > 0 and st.fleet_restricted_seconds(self.budget_window) + action.duration_seconds > self.fleet_hard_cap:
            return self._d(False, False, "DENY_HARD_CEILING", risk)
        # 5. high criticality => human approval, never autonomous
        if fl.criticality and asset.criticality == "high":
            if approved:
                st.consume(key)
                return self._d(True, False, "ALLOW_APPROVED_HIGH_IMPACT", risk)
            return self._d(False, True, "REVIEW_HIGH_CRITICALITY", risk)
        # 6. cumulative budget: repeated short restrictions must not add up to an outage without a human
        budget = self.budget_by_crit.get(asset.criticality, self.device_budget)
        if fl.budget and st.restricted_seconds(action.device_id, self.budget_window) + action.duration_seconds > budget:
            if approved:
                st.consume(key)
                return self._d(True, False, "ALLOW_APPROVED_OVER_BUDGET", risk)
            return self._d(False, True, "REVIEW_DEVICE_BUDGET", risk)
        # 7. optional fleet-wide budget: spreading restrictions over many devices must not evade the per-device cap
        if fl.budget and self.fleet_budget > 0 and st.fleet_restricted_seconds(self.budget_window) + action.duration_seconds > self.fleet_budget:
            if (self.corroboration_min > 0 and inc.indicator
                    and st.corroborating_devices(inc.indicator, action.device_id, self.budget_window) >= self.corroboration_min
                    and (self.corroboration_budget <= 0
                         or st.exempt_seconds(self.budget_window) + action.duration_seconds <= self.corroboration_budget)):
                st.exempt_log.append((st.now(), float(action.duration_seconds)))
                return self._d(True, False, "ALLOW_CORROBORATED", risk)
            if approved:
                st.consume(key)
                return self._d(True, False, "ALLOW_APPROVED_OVER_BUDGET", risk)
            return self._d(False, True, "REVIEW_FLEET_BUDGET", risk)
        return self._d(True, False, "ALLOW_RESTRICT", risk)
