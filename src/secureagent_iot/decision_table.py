"""Policy decision table: one row per decision branch, shared by the tests and the manuscript appendix.

Each row is (id, plain-language condition, trusted-state setup, proposal fields, expected (allowed, requires_approval, reason_code))."""
from .policy import IncidentRecord, PolicyEngine, TrustedState
from .schemas import ProposedAction

EV = frozenset({"ev-1", "ev-2", "ev-3"})


def engine(cfg, registry, device="cam_01", p=0.95, evidence=EV, restricted=(), resolved=False, approve=None, advance=0,
           approve_other_incident=None, kinds=None, other_incident_evidence=None, used_seconds=0, sources=None,
           evidence_age=0, budget_by_crit=None, fleet_used=0, fleet_budget=None, indicator="", corroborating=0,
           corroboration_min=None, exempt_used=0, corroboration_budget=None, hard_dev=None, hard_fleet=None):
    clock = {"t": 0.0}
    st = TrustedState(registry=registry, is_restricted=lambda d: d in restricted, now=lambda: clock["t"],
                      restricted_seconds=lambda d, w: float(used_seconds), fleet_restricted_seconds=lambda w: float(fleet_used),
                      corroborating_devices=lambda i, d, w: int(corroborating))
    st.record_incident(IncidentRecord("inc-1", device, p, evidence, resolved=resolved, evidence_kinds=kinds or {},
                                      evidence_sources=sources or {}, evidence_time={e: -float(evidence_age) for e in evidence},
                                      indicator=indicator))
    if other_incident_evidence:
        st.record_incident(IncidentRecord("inc-2", "plug_02", 0.95, frozenset(other_incident_evidence)))
    if exempt_used:
        st.exempt_log.append((0.0, float(exempt_used)))
    if approve:
        st.approve("inc-1", *approve)
    if approve_other_incident:
        st.approve("inc-9", *approve_other_incident)
    clock["t"] += advance
    if budget_by_crit is not None:
        cfg = {**cfg, "policy": {**cfg["policy"], "device_budget_by_criticality": budget_by_crit}}
    if fleet_budget is not None:
        cfg = {**cfg, "policy": {**cfg["policy"], "fleet_budget_seconds": fleet_budget}}
    if corroboration_min is not None:
        cfg = {**cfg, "policy": {**cfg["policy"], "fleet_corroboration_devices": corroboration_min}}
    for key, val in (("corroboration_budget_seconds", corroboration_budget), ("device_hard_cap_seconds", hard_dev),
                     ("fleet_hard_cap_seconds", hard_fleet)):
        if val is not None:
            cfg = {**cfg, "policy": {**cfg["policy"], key: val}}
    return PolicyEngine(cfg, st)


def act(**kw):
    base = {"incident_id": "inc-1", "device_id": "cam_01", "action_type": "restrict_egress", "duration_seconds": 300,
            "evidence_refs": ["ev-1", "ev-2"], "reason": "r", "agent_id": "responder"}
    return ProposedAction(**{**base, **kw})



R, N, Q, L = "restrict_egress", "notify", "request_review", "release_restriction"
CASES = [
    # id, engine kwargs, action kwargs, (allowed, requires_approval, reason_code)
    ("unknown_agent", {}, {"agent_id": "mallory"}, (False, False, "DENY_UNKNOWN_AGENT")),
    ("unregistered_device", {}, {"device_id": "cam_99"}, (False, False, "DENY_UNREGISTERED_DEVICE")),
    ("unknown_incident", {}, {"incident_id": "inc-404"}, (False, False, "DENY_UNKNOWN_INCIDENT")),
    ("device_mismatch", {}, {"device_id": "plug_02"}, (False, False, "DENY_DEVICE_MISMATCH")),
    ("unresolved_evidence", {}, {"evidence_refs": ["ev-1", "ev-x"]}, (False, False, "DENY_UNRESOLVED_EVIDENCE")),
    ("not_allowlisted", {"device": "gateway_01"}, {"device_id": "gateway_01"}, (False, False, "DENY_ACTION_NOT_ALLOWLISTED")),
    ("review_allowed", {}, {"action_type": Q, "duration_seconds": 0}, (True, False, "ALLOW_REVIEW")),
    ("notify_floor", {}, {"action_type": N, "duration_seconds": 0}, (False, True, "REVIEW_ESCALATION_FLOOR")),
    ("notify_floor_exact_threshold", {"p": 0.80, "evidence": frozenset({"ev-1", "ev-2"})}, {"action_type": N, "duration_seconds": 0},
     (False, True, "REVIEW_ESCALATION_FLOOR")),
    ("notify_low_p", {"p": 0.79}, {"action_type": N, "duration_seconds": 0}, (True, False, "ALLOW_NOTIFY")),
    ("notify_thin_evidence", {"evidence": frozenset({"ev-1"})}, {"action_type": N, "duration_seconds": 0, "evidence_refs": ["ev-1"]},
     (True, False, "ALLOW_NOTIFY")),
    ("release_resolved", {"resolved": True, "restricted": ("cam_01",)}, {"action_type": L, "duration_seconds": 0}, (True, False, "ALLOW_RELEASE")),
    ("release_approved", {"approve": ("cam_01", L), "restricted": ("cam_01",)}, {"action_type": L, "duration_seconds": 0}, (True, False, "ALLOW_RELEASE")),
    ("release_unresolved", {"restricted": ("cam_01",)}, {"action_type": L, "duration_seconds": 0}, (False, True, "REVIEW_RELEASE_UNRESOLVED")),
    ("release_nothing", {}, {"action_type": L, "duration_seconds": 0}, (False, False, "DENY_NOTHING_TO_RELEASE")),
    ("duration_zero", {}, {"duration_seconds": 0}, (False, False, "DENY_INVALID_DURATION")),
    ("duration_one", {}, {"duration_seconds": 1}, (True, False, "ALLOW_RESTRICT")),
    ("ttl_at_cap", {}, {"duration_seconds": 300}, (True, False, "ALLOW_RESTRICT")),
    ("ttl_over_cap", {}, {"duration_seconds": 301}, (False, False, "DENY_TTL_EXCEEDS_CAP")),
    ("high_crit_no_approval", {"device": "actuator_01"}, {"device_id": "actuator_01"}, (False, True, "REVIEW_HIGH_CRITICALITY")),
    ("high_crit_approved", {"device": "actuator_01", "approve": ("actuator_01", R)}, {"device_id": "actuator_01"},
     (True, False, "ALLOW_APPROVED_HIGH_IMPACT")),
    ("medium_crit_autonomous", {"device": "hub_01"}, {"device_id": "hub_01"}, (True, False, "ALLOW_RESTRICT")),
    ("below_floor", {"p": 0.49}, {}, (False, False, "DENY_BELOW_THRESHOLD")),
    ("at_floor", {"p": 0.50}, {}, (False, True, "REVIEW_UNCERTAIN")),
    ("just_below_threshold", {"p": 0.7999}, {}, (False, True, "REVIEW_UNCERTAIN")),
    ("at_threshold", {"p": 0.80}, {}, (True, False, "ALLOW_RESTRICT")),
    ("one_evidence", {}, {"evidence_refs": ["ev-1"]}, (False, True, "REVIEW_INSUFFICIENT_EVIDENCE")),
    ("two_evidence", {}, {"evidence_refs": ["ev-1", "ev-3"]}, (True, False, "ALLOW_RESTRICT")),
]



EXTRA = [
    ("stale_approval", {"device": "actuator_01", "approve": ("actuator_01", R), "advance": 901}, {"device_id": "actuator_01"}, (False, True, "REVIEW_HIGH_CRITICALITY")),
    ("approval_for_other_device", {"device": "actuator_01", "approve": ("actuator_02", R)}, {"device_id": "actuator_01"}, (False, True, "REVIEW_HIGH_CRITICALITY")),
    ("approval_for_other_action", {"device": "actuator_01", "approve": ("actuator_01", L)}, {"device_id": "actuator_01"}, (False, True, "REVIEW_HIGH_CRITICALITY")),
    ("approval_for_other_incident", {"device": "actuator_01", "approve_other_incident": ("actuator_01", R)}, {"device_id": "actuator_01"}, (False, True, "REVIEW_HIGH_CRITICALITY")),
    ("duplicate_evidence_refs", {}, {"evidence_refs": ["ev-1", "ev-1"]}, (False, True, "REVIEW_INSUFFICIENT_EVIDENCE")),
    ("same_kind_evidence", {"kinds": {"ev-1": "flow_anomaly", "ev-2": "flow_anomaly", "ev-3": "dns_anomaly"}}, {"evidence_refs": ["ev-1", "ev-2"]}, (False, True, "REVIEW_INSUFFICIENT_EVIDENCE")),
    ("cross_incident_evidence", {"other_incident_evidence": ["ev-o1", "ev-o2"]}, {"evidence_refs": ["ev-o1", "ev-o2"]}, (False, False, "DENY_UNRESOLVED_EVIDENCE")),
    ("budget_exceeded", {"used_seconds": 900}, {"duration_seconds": 300}, (False, True, "REVIEW_DEVICE_BUDGET")),
    ("same_source_evidence", {"sources": {"ev-1": "local-collector", "ev-2": "local-collector", "ev-3": "edge-ids"}},
     {"evidence_refs": ["ev-1", "ev-2"]}, (False, True, "REVIEW_INSUFFICIENT_EVIDENCE")),
    ("different_kind_and_source", {"kinds": {"ev-1": "flow_anomaly", "ev-2": "dns_anomaly"}, "sources": {"ev-1": "netflow", "ev-2": "dns-log"}},
     {"evidence_refs": ["ev-1", "ev-2"]}, (True, False, "ALLOW_RESTRICT")),
    ("stale_evidence", {"evidence_age": 1801}, {}, (False, True, "REVIEW_INSUFFICIENT_EVIDENCE")),
    ("evidence_at_age_limit", {"evidence_age": 1800}, {}, (True, False, "ALLOW_RESTRICT")),
    ("stale_evidence_no_floor", {"evidence_age": 1801}, {"action_type": N, "duration_seconds": 0}, (True, False, "ALLOW_NOTIFY")),
    ("low_crit_budget_override", {"device": "plug_01", "used_seconds": 1500, "budget_by_crit": {"low": 1800, "medium": 900}},
     {"device_id": "plug_01", "duration_seconds": 300}, (True, False, "ALLOW_RESTRICT")),
    ("medium_crit_budget_override", {"device": "hub_01", "used_seconds": 700, "budget_by_crit": {"low": 1800, "medium": 900}},
     {"device_id": "hub_01", "duration_seconds": 300}, (False, True, "REVIEW_DEVICE_BUDGET")),
    ("release_resolved_nothing_restricted", {"resolved": True}, {"action_type": L, "duration_seconds": 0}, (False, False, "DENY_NOTHING_TO_RELEASE")),
    ("approved_high_crit_low_p", {"device": "actuator_01", "approve": ("actuator_01", R), "p": 0.30}, {"device_id": "actuator_01"},
     (False, False, "DENY_BELOW_THRESHOLD")),
    ("approved_high_crit_thin_evidence", {"device": "actuator_01", "approve": ("actuator_01", R)}, {"device_id": "actuator_01", "evidence_refs": ["ev-1"]},
     (False, True, "REVIEW_INSUFFICIENT_EVIDENCE")),
    ("fleet_budget_exceeded", {"fleet_used": 1600, "fleet_budget": 1800}, {"duration_seconds": 300}, (False, True, "REVIEW_FLEET_BUDGET")),
    ("fleet_budget_exactly_met", {"fleet_used": 1500, "fleet_budget": 1800}, {"duration_seconds": 300}, (True, False, "ALLOW_RESTRICT")),
    ("fleet_budget_exceeded_but_approved", {"device": "hub_01", "fleet_used": 1600, "fleet_budget": 1800, "approve": ("hub_01", R)},
     {"device_id": "hub_01", "duration_seconds": 300}, (True, False, "ALLOW_APPROVED_OVER_BUDGET")),
    ("fleet_budget_off", {"fleet_used": 100000, "fleet_budget": 0}, {"duration_seconds": 300}, (True, False, "ALLOW_RESTRICT")),
    ("fleet_over_corroborated", {"fleet_used": 1600, "fleet_budget": 1800, "indicator": "c2:x", "corroborating": 2, "corroboration_min": 2},
     {"duration_seconds": 300}, (True, False, "ALLOW_CORROBORATED")),
    ("fleet_over_too_few_corroborating", {"fleet_used": 1600, "fleet_budget": 1800, "indicator": "c2:x", "corroborating": 1, "corroboration_min": 2},
     {"duration_seconds": 300}, (False, True, "REVIEW_FLEET_BUDGET")),
    ("fleet_over_no_indicator", {"fleet_used": 1600, "fleet_budget": 1800, "indicator": "", "corroborating": 5, "corroboration_min": 2},
     {"duration_seconds": 300}, (False, True, "REVIEW_FLEET_BUDGET")),
    ("fleet_over_corroboration_off", {"fleet_used": 1600, "fleet_budget": 1800, "indicator": "c2:x", "corroborating": 5, "corroboration_min": 0},
     {"duration_seconds": 300}, (False, True, "REVIEW_FLEET_BUDGET")),
    ("corroborated_but_device_budget_spent", {"used_seconds": 800, "fleet_used": 1600, "fleet_budget": 1800, "indicator": "c2:x",
     "corroborating": 3, "corroboration_min": 2}, {"duration_seconds": 300}, (False, True, "REVIEW_DEVICE_BUDGET")),
    ("corroborated_within_exemption_budget", {"fleet_used": 1600, "fleet_budget": 1800, "indicator": "c2:x", "corroborating": 2,
     "corroboration_min": 2, "corroboration_budget": 1200, "exempt_used": 900}, {"duration_seconds": 300}, (True, False, "ALLOW_CORROBORATED")),
    ("corroborated_but_exemption_budget_spent", {"fleet_used": 1600, "fleet_budget": 1800, "indicator": "c2:x", "corroborating": 2,
     "corroboration_min": 2, "corroboration_budget": 1200, "exempt_used": 1000}, {"duration_seconds": 300}, (False, True, "REVIEW_FLEET_BUDGET")),
    ("device_hard_ceiling_blocks_approval", {"used_seconds": 1000, "approve": ("cam_01", R), "hard_dev": 1200},
     {"duration_seconds": 300}, (False, False, "DENY_HARD_CEILING")),
    ("device_hard_ceiling_exactly_met", {"used_seconds": 900, "approve": ("cam_01", R), "hard_dev": 1200},
     {"duration_seconds": 300}, (True, False, "ALLOW_APPROVED_OVER_BUDGET")),
    ("fleet_hard_ceiling_blocks_corroboration", {"fleet_used": 2300, "fleet_budget": 1800, "indicator": "c2:x", "corroborating": 3,
     "corroboration_min": 2, "hard_fleet": 2400}, {"duration_seconds": 300}, (False, False, "DENY_HARD_CEILING")),
    ("hard_ceiling_blocks_approved_high_crit", {"device": "pump_01", "used_seconds": 1000, "approve": ("pump_01", R), "hard_dev": 1200},
     {"device_id": "pump_01", "duration_seconds": 300}, (False, False, "DENY_HARD_CEILING")),
    ("budget_exactly_met", {"used_seconds": 600}, {"duration_seconds": 300}, (True, False, "ALLOW_RESTRICT")),
    ("budget_exceeded_but_approved", {"used_seconds": 900, "approve": ("cam_01", R)}, {"duration_seconds": 300}, (True, False, "ALLOW_APPROVED_OVER_BUDGET")),
]
CASES = CASES + EXTRA

DESCRIPTION = {
    "unknown_agent": "Proposal from an agent identity that is not registered", "unregistered_device": "Device not in the registry",
    "unknown_incident": "Incident not recorded", "device_mismatch": "Device differs from the recorded incident device",
    "unresolved_evidence": "An evidence reference is not recorded for this incident", "not_allowlisted": "Action not in the asset's allowlist",
    "review_allowed": "Request for human review", "notify_floor": "Notify on a strong, well-evidenced alert",
    "notify_floor_exact_threshold": "Notify with p exactly at the threshold and two observations", "notify_low_p": "Notify with p below the threshold",
    "notify_thin_evidence": "Notify with one recorded observation", "release_resolved": "Release, incident resolved",
    "release_approved": "Release, valid approval", "release_unresolved": "Release of an active restriction, unresolved, no approval",
    "release_nothing": "Release when nothing is restricted", "duration_zero": "Restrict with duration 0",
    "duration_one": "Restrict for 1 s", "ttl_at_cap": "Restrict for exactly 300 s", "ttl_over_cap": "Restrict for 301 s",
    "high_crit_no_approval": "Restrict a high-criticality device, no approval", "high_crit_approved": "Restrict a high-criticality device, valid approval",
    "medium_crit_autonomous": "Restrict a medium-criticality device", "below_floor": "Restrict with p = 0.49",
    "at_floor": "Restrict with p = 0.50", "just_below_threshold": "Restrict with p = 0.7999", "at_threshold": "Restrict with p = 0.80",
    "one_evidence": "Restrict citing one observation", "two_evidence": "Restrict citing two observations of different kinds",
    "stale_approval": "High-criticality restrict, approval older than 900 s", "approval_for_other_device": "Approval recorded for a different device",
    "approval_for_other_action": "Approval recorded for a different action", "approval_for_other_incident": "Approval recorded for a different incident",
    "duplicate_evidence_refs": "Restrict citing the same observation twice", "same_kind_evidence": "Restrict citing two observations of the same kind",
    "cross_incident_evidence": "Restrict citing observations recorded for another incident", "budget_exceeded": "Restrict when the device has used its 900 s budget",
    "same_source_evidence": "Two cited observations of different kinds from the same collector",
    "different_kind_and_source": "Two cited observations differing in kind and in source",
    "stale_evidence": "Restrict citing observations older than 1800 s", "evidence_at_age_limit": "Restrict citing observations exactly 1800 s old",
    "stale_evidence_no_floor": "Notify when all evidence is older than 1800 s",
    "low_crit_budget_override": "Low-criticality device, 1500 s used, per-criticality budget 1800 s",
    "medium_crit_budget_override": "Medium-criticality device, 700 s used, per-criticality budget 900 s",
    "release_resolved_nothing_restricted": "Release on a resolved incident when nothing is restricted",
    "approved_high_crit_low_p": "High-criticality restrict with valid approval but p = 0.30",
    "approved_high_crit_thin_evidence": "High-criticality restrict with valid approval citing one observation",
    "fleet_budget_exceeded": "Fleet has used 1600 s of an 1800 s fleet budget; restrict 300 s",
    "fleet_budget_exactly_met": "Fleet has used 1500 s of an 1800 s fleet budget; restrict 300 s",
    "fleet_budget_exceeded_but_approved": "Over the fleet budget with a valid approval",
    "fleet_budget_off": "Fleet budget disabled (0), large fleet use",
    "corroborated_within_exemption_budget": "Corroborated over the fleet budget; 900 of a 1200 s exemption budget used",
    "corroborated_but_exemption_budget_spent": "Corroborated over the fleet budget; 1000 of a 1200 s exemption budget used",
    "device_hard_ceiling_blocks_approval": "Approved restriction; device at 1000 s of a 1200 s hard ceiling",
    "device_hard_ceiling_exactly_met": "Approved restriction; device at 900 s of a 1200 s hard ceiling (over the 900 s budget)",
    "fleet_hard_ceiling_blocks_corroboration": "Corroborated restriction; fleet at 2300 s of a 2400 s hard ceiling",
    "hard_ceiling_blocks_approved_high_crit": "Approved high-criticality restriction; device at 1000 s of a 1200 s hard ceiling",
    "fleet_over_corroborated": "Over the fleet budget; indicator seen on 2 other devices, exemption needs 2",
    "fleet_over_too_few_corroborating": "Over the fleet budget; indicator seen on 1 other device, exemption needs 2",
    "fleet_over_no_indicator": "Over the fleet budget; no indicator recorded for the incident",
    "fleet_over_corroboration_off": "Over the fleet budget; indicator on 5 other devices but exemption off",
    "corroborated_but_device_budget_spent": "Corroborated, but the device has used 800 of its 900 s budget; restrict 300 s",
    "budget_exactly_met": "Restrict that exactly reaches the 900 s budget", "budget_exceeded_but_approved": "Over-budget restrict with valid approval",
}


def evaluate_table(cfg, registry):
    """Run every row through the real engine. Returns rows for the manuscript appendix."""
    rows = []
    for name, ekw, akw, expected in CASES:
        d = engine(cfg, registry, **ekw).evaluate(act(**akw))
        outcome = "allow" if d.allowed else ("review" if d.requires_approval else "deny")
        rows.append({"case": name, "condition": DESCRIPTION[name], "decision": outcome, "reason_code": d.reason_code,
                     "matches_expected": (d.allowed, d.requires_approval, d.reason_code) == expected})
    return rows
