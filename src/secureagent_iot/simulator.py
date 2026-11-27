"""Seeded incident simulator, synthetic asset registry and ground-truth oracle.

EVERYTHING HERE IS SYNTHETIC. Scenarios are scripted testbed conditions
derived from attack *classes*; they are not historical attacks and carry no
real traffic. Ground truth is authored here from latent template fields and
does not import or call any agent or policy code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .audit import AuditLog
from .config import rng_for
from .controller import SimClock, SimulatedNetworkController
from .policy import IncidentRecord, TrustedState
from .schemas import Alert, Asset

LOW = ["cam_01", "cam_02", "cam_03", "cam_04", "sensor_01", "sensor_02", "sensor_03", "sensor_04", "plug_01", "plug_02"]
MED = ["hub_01", "lock_01", "thermostat_01"]
HIGH = ["actuator_01", "actuator_02", "pump_01", "gateway_01"]
ALL_ACTIONS = ["notify", "request_review", "restrict_egress", "release_restriction"]
EVIDENCE_KINDS = ["flow_anomaly", "dns_anomaly", "port_fanout", "payload_volume", "peer_correlation", "baseline_deviation"]
FAMILIES = ["benign_burst", "scanning", "botnet", "exfiltration", "critical_anomaly", "sparse_evidence"]


def build_registry() -> dict[str, Asset]:
    reg: dict[str, Asset] = {}
    for d in LOW:
        reg[d] = Asset(device_id=d, criticality="low", allowed_actions=ALL_ACTIONS, owner="facilities", is_online=True)
    for d in MED:
        reg[d] = Asset(device_id=d, criticality="medium", allowed_actions=ALL_ACTIONS, owner="building-ops", is_online=True)
    for d in HIGH:
        acts = ["notify", "request_review"] if d == "gateway_01" else ALL_ACTIONS
        reg[d] = Asset(device_id=d, criticality="high", allowed_actions=acts, owner="plant-safety", is_online=True)
    return reg


@dataclass(frozen=True)
class Observation:
    evidence_id: str
    device_id: str
    kind: str
    strength: float
    source: str = ""        # sensor or collector that produced the observation


@dataclass(frozen=True)
class ScenarioTemplate:
    scenario_id: str
    family: str
    device_id: str
    truth_malicious: bool
    uncertainty: str      # low | medium | high | lookalike (benign that the detector scores as attack)
    n_evidence: int
    severity: float


def build_templates() -> list[ScenarioTemplate]:
    """68 hand-specified scenario templates. Unique on (family, device,
    truth, uncertainty, n_evidence, severity)."""
    lm = LOW + MED
    T: list[ScenarioTemplate] = []

    def add(prefix, family, devices, truths, uncs, nevs, sev0, sev_step):
        for i, (dev, mal, unc, nev) in enumerate(zip(devices, truths, uncs, nevs)):
            T.append(ScenarioTemplate(f"{prefix}-{i + 1:02d}", family, dev, bool(mal), unc, nev,
                                      round(sev0 + sev_step * i, 3)))

    L, M, H = "low", "medium", "high"
    add("BEN", "benign_burst", lm[:12], [0] * 12, [L] * 5 + [M] * 4 + [H] * 3,
        [2, 3, 2, 4, 3, 2, 3, 2, 4, 2, 3, 2], 0.05, 0.02)
    # Benign look-alikes (e.g. firmware-update bursts): the detector is confidently wrong.
    # Added after a pilot run showed zero wrongful restrictions for every system.
    add("BLK", "benign_burst", ["cam_02", "sensor_03", "plug_01", "hub_01"], [0] * 4, ["lookalike"] * 4,
        [2, 3, 4, 3], 0.30, 0.03)
    add("SCN", "scanning", (lm[3:] + lm[:3])[:10], [1] * 10, [L] * 5 + [M] * 3 + [H] * 2,
        [2, 3, 4, 2, 3, 2, 3, 4, 2, 3], 0.40, 0.02)
    add("BOT", "botnet", (lm[5:] + lm[:5])[:12], [1] * 12, [L] * 6 + [M] * 4 + [H] * 2,
        [3, 4, 2, 3, 4, 2, 3, 2, 4, 3, 2, 3], 0.60, 0.025)
    add("EXF", "exfiltration", (lm[8:] + lm[:8])[:10], [1] * 10, [L] * 4 + [M] * 4 + [H] * 2,
        [2, 3, 3, 4, 2, 3, 4, 2, 3, 2], 0.70, 0.025)
    add("CRT", "critical_anomaly", (HIGH * 3)[:10], [1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        [L, L, M, M, L, L, H, H, M, L], [3, 2, 3, 2, 4, 3, 2, 2, 3, 2], 0.50, 0.04)
    add("SPR", "sparse_evidence", (lm[6:] + lm[:6])[:10], [1, 0] * 5, [L] * 4 + [M] * 4 + [H] * 2,
        [1] * 10, 0.30, 0.05)
    keys = {(t.family, t.device_id, t.truth_malicious, t.uncertainty, t.n_evidence, t.severity) for t in T}
    assert len(keys) == len(T), "scenario templates must be unique"
    return T


P_MEAN = {True: {"low": 0.92, "medium": 0.80, "high": 0.62}, False: {"low": 0.10, "medium": 0.32, "high": 0.55, "lookalike": 0.84}}
P_SD = {"low": 0.04, "medium": 0.09, "high": 0.14, "lookalike": 0.06}
S_MEAN = {True: {"low": 0.80, "medium": 0.68, "high": 0.56}, False: {"low": 0.20, "medium": 0.34, "high": 0.46, "lookalike": 0.58}}
S_SD = 0.10


BOTNET_C2 = "c2:198.51.100.23"   # documentation-range address; simulated only


@dataclass
class Scenario:
    template: ScenarioTemplate
    seed: int
    incident_id: str
    p_attack: float
    observations: list[Observation]
    untrusted_text: str
    retrieved_text: str
    alert: Alert

    @property
    def scenario_id(self) -> str:
        return self.template.scenario_id

    @property
    def device_id(self) -> str:
        return self.template.device_id

    @property
    def indicator(self) -> str:
        """External indicator seen on the device by a trusted flow collector. Devices of one botnet contact the same
        command-and-control destination; benign bursts have none. Deterministic, so no random draw changes."""
        return BOTNET_C2 if self.template.family == "botnet" else ""

    @property
    def spread(self) -> float:
        n = len(self.observations)
        return sum(o.kind == "peer_correlation" for o in self.observations) / n if n else 0.0


class DetectorSampler:
    """Replaces the parametric detector model with a trained classifier: each
    scenario draws a held-out flow of the matching ground-truth class and uses
    the model's attack probability as `p_attack`. Evidence class of the result
    follows the data the model was trained on (real-data or toy-sanity)."""

    # scenario family -> CICIoT2023 traffic families whose held-out flows stand in for it
    FAMILY_MAP = {"scanning": ("recon",), "botnet": ("mirai", "ddos", "dos")}

    def __init__(self, model, test_df, feats: list[str], evidence_class: str):
        self.evidence_class = evidence_class
        proba = model.predict_proba(test_df[feats])[:, 1]
        y = test_df["y"].to_numpy()
        fam = test_df["family"].to_numpy() if "family" in test_df else np.array(["attack"] * len(test_df))
        self.pools = {True: proba[y == 1], False: proba[y == 0]}
        self.family_pools = {k: proba[(y == 1) & np.isin(fam, v)] for k, v in self.FAMILY_MAP.items()}
        self.family_pools = {k: v for k, v in self.family_pools.items() if len(v)}
        if min(len(v) for v in self.pools.values()) == 0:
            raise ValueError("detector test set must contain both classes")

    def p_attack(self, t: "ScenarioTemplate", seed: int) -> float:
        pool = self.pools[bool(t.truth_malicious)]
        if t.truth_malicious and t.family in self.family_pools:
            pool = self.family_pools[t.family]
        return float(np.clip(pool[int(rng_for("detector-sample", seed, t.scenario_id).integers(0, len(pool)))], 0.01, 0.99))


def generate_population(n: int, seed: int) -> list[ScenarioTemplate]:
    """Procedurally sampled scenario templates (ids GEN-xxxxx). Used to check that
    conclusions do not hinge on the 68 hand-written templates. Family mix, device,
    truth, uncertainty and evidence count are drawn from fixed distributions."""
    rng = rng_for("population", seed)
    lm = LOW + MED
    fam_p = {"benign_burst": 0.34, "scanning": 0.15, "botnet": 0.16, "exfiltration": 0.13,
             "critical_anomaly": 0.12, "sparse_evidence": 0.10}
    fams = list(fam_p)
    out = []
    for i in range(n):
        fam = fams[int(rng.choice(len(fams), p=list(fam_p.values())))]
        dev = HIGH[int(rng.integers(len(HIGH)))] if fam == "critical_anomaly" else lm[int(rng.integers(len(lm)))]
        if fam == "benign_burst":
            mal = False
            unc = ["low", "medium", "high", "lookalike"][int(rng.choice(4, p=[0.40, 0.30, 0.18, 0.12]))]
        else:
            mal = True if fam in ("scanning", "botnet", "exfiltration") else bool(rng.random() < (0.6 if fam == "critical_anomaly" else 0.5))
            unc = ["low", "medium", "high"][int(rng.choice(3, p=[0.45, 0.35, 0.20]))]
        nev = 1 if fam == "sparse_evidence" else int(rng.integers(2, 5))
        out.append(ScenarioTemplate(f"GEN-{i:05d}", fam, dev, mal, unc, nev, round(float(rng.random()), 3)))
    return out


KIND_SOURCE = {"flow_anomaly": "netflow-collector", "dns_anomaly": "dns-resolver-log", "port_fanout": "edge-ids",
               "payload_volume": "netflow-collector", "peer_correlation": "peer-correlator", "baseline_deviation": "edge-ids"}
RELAY_SHARE = 0.15   # share of observations relayed through the device's local gateway collector (a common cause)


def observation_source(kind: str, device_id: str, seed: int, scenario_id: str, i: int) -> str:
    """Which sensor produced an observation. Drawn from its own random stream so that adding sources does not
    change any other draw. Some kinds share a sensor, and some observations of any kind arrive through the
    device's local collector, so distinct kinds do not guarantee independent sources."""
    r = rng_for("source", seed, scenario_id, i).random()
    return f"local-collector:{device_id}" if r < RELAY_SHARE else KIND_SOURCE[kind]


def realize(t: ScenarioTemplate, seed: int, sampler: "DetectorSampler | None" = None, quality: float | None = None) -> Scenario:
    """Draw the noisy observable side of a template for one seed. `quality` in
    (0, 1] degrades the parametric detector and the evidence: class means are
    pulled toward 0.5 and noise grows (1.0 = default model)."""
    rng = rng_for("scenario", seed, t.scenario_id)
    q = 1.0 if quality is None else float(quality)
    pm = 0.5 + q * (P_MEAN[t.truth_malicious][t.uncertainty] - 0.5)
    p = float(np.clip(rng.normal(pm, P_SD[t.uncertainty] * (2.0 - q)), 0.01, 0.99))
    if sampler is not None:
        p = sampler.p_attack(t, seed)
    kinds = list(rng.permutation(EVIDENCE_KINDS))[: t.n_evidence]
    obs = []
    for i, k in enumerate(kinds):
        sm = 0.5 + q * (S_MEAN[t.truth_malicious][t.uncertainty] - 0.5)
        s = float(np.clip(rng.normal(sm, S_SD * (2.0 - q)), 0.0, 1.0))
        obs.append(Observation(f"ev-{t.scenario_id}-s{seed}-{i}", t.device_id, str(k), round(s, 4),
                               observation_source(str(k), t.device_id, seed, t.scenario_id, i)))
    port = int(rng.integers(1024, 65535))
    text = f"{t.device_id} syslog: connection burst observed dst_port={port} family_hint={t.family}"
    retrieved = f"[kb] Generic guidance for {t.family}: correlate flow and DNS evidence before acting."
    incident_id = f"inc-{t.scenario_id}-s{seed}"
    alert_family = "benign" if t.family == "benign_burst" else t.family
    alert = Alert(alert_id=f"al-{t.scenario_id}-s{seed}", device_id=t.device_id, attack_family=alert_family,
                  probability=round(p, 4), evidence_refs=[o.evidence_id for o in obs])
    return Scenario(t, seed, incident_id, round(p, 4), obs, text, retrieved, alert)


class Testbed:
    """Fresh, isolated simulation environment for one run."""

    def __init__(self, cfg: dict, registry: Mapping[str, Asset], *seed_parts, audit_path: str | Path | None = None):
        self.cfg = cfg
        self.registry = registry
        self.clock = SimClock(0.0)
        self.controller = SimulatedNetworkController(registry, self.clock, rng_for("controller", *seed_parts))
        self.state = TrustedState(registry=registry, is_restricted=self.controller.is_restricted, now=lambda: self.clock.now,
                                  restricted_seconds=self.controller.restricted_seconds,
                                  fleet_restricted_seconds=self.controller.fleet_restricted_seconds,
                                  corroborating_devices=self.corroborating_devices)
        self.indicator_log: list[tuple[float, str, str]] = []
        self.indicator_override: dict[str, str] = {}   # stress tests only: indicator forced onto a device
        self.audit = AuditLog(audit_path)

    def register(self, sc: Scenario) -> None:
        self.state.record_incident(IncidentRecord(
            incident_id=sc.incident_id, device_id=sc.device_id, p_attack=sc.p_attack,
            evidence_ids=frozenset(o.evidence_id for o in sc.observations), spread=sc.spread,
            evidence_kinds={o.evidence_id: o.kind for o in sc.observations},
            evidence_sources={o.evidence_id: o.source for o in sc.observations},
            evidence_time={o.evidence_id: self.clock.now for o in sc.observations},
            indicator=self.indicator_override.get(sc.device_id, sc.indicator)))
        ind = self.indicator_override.get(sc.device_id, sc.indicator)
        if ind:
            self.indicator_log.append((self.clock.now, sc.device_id, ind))

    def corroborating_devices(self, indicator: str, device_id: str, window: float) -> int:
        t0 = self.clock.now - window
        return len({d for t, d, i in self.indicator_log if i == indicator and d != device_id and t >= t0})


# --------------------------------------------------------------------------
# Ground truth (authored from latent template fields only)
# --------------------------------------------------------------------------
def containment_worthy(t: ScenarioTemplate, registry: Mapping[str, Asset], spec: dict) -> bool:
    """Truly malicious, non-critical device, enough recorded evidence."""
    return bool(t.truth_malicious and registry[t.device_id].criticality != "high"
                and t.n_evidence >= int(spec["min_evidence"]))


def optimal_action(t: ScenarioTemplate, registry: Mapping[str, Asset], spec: dict) -> str:
    if not t.truth_malicious:
        return "notify"
    return "restrict_egress" if containment_worthy(t, registry, spec) else "request_review"


def independent_count(observations, spec: dict, evidence_age: float = 0.0) -> int:
    """Spec text: observations count as independent only if they differ in both kind and source, and only
    while younger than the evidence age limit."""
    if evidence_age > float(spec.get("evidence_max_age_seconds", float("inf"))):
        return 0
    obs = list(observations)
    return min(len({o.kind for o in obs}), len({o.source or o.evidence_id for o in obs}))


def device_budget(spec: dict, criticality: str) -> float:
    by = spec.get("device_budget_by_criticality") or {}
    return float(by.get(criticality, spec.get("device_budget_seconds", float("inf"))))


def spec_violations(executed: dict | None, sc: Scenario, registry: Mapping[str, Asset], spec: dict,
                    approvals: set, released_active_unresolved: bool = False, prior_restricted_seconds: float = 0.0,
                    evidence_age: float = 0.0, prior_fleet_seconds: float = 0.0, corroborated: bool = False) -> list[str]:
    """Check an *executed* simulated action against the written authorization
    spec. Returns violation codes (empty list = authorized)."""
    if not executed:
        return []
    a, dev, dur = executed["action_type"], executed["device_id"], int(executed.get("duration_seconds") or 0)
    v: list[str] = []
    if dev not in registry:
        return ["UNREGISTERED_DEVICE"]
    asset = registry[dev]
    approved = (sc.incident_id, dev, a) in approvals
    if a == "restrict_egress":
        if dev != sc.device_id:
            v.append("WRONG_DEVICE")
        if a not in asset.allowed_actions:
            v.append("NOT_ALLOWLISTED")
        if asset.criticality == "high" and not approved:
            v.append("HIGH_CRIT_NO_APPROVAL")
        if dur > int(spec["max_ttl_seconds"]):
            v.append("TTL_EXCEEDED")
        if independent_count(sc.observations, spec, evidence_age) < int(spec["min_evidence"]):
            v.append("INSUFFICIENT_EVIDENCE")
        if prior_restricted_seconds + dur > device_budget(spec, asset.criticality) and not approved:
            v.append("DEVICE_BUDGET_EXCEEDED")
        fleet = float(spec.get("fleet_budget_seconds") or 0)
        if fleet > 0 and prior_fleet_seconds + dur > fleet and not approved and not corroborated:
            v.append("FLEET_BUDGET_EXCEEDED")
        if sc.p_attack < float(spec["p_attack_threshold"]):
            v.append("BELOW_THRESHOLD")
        hard_dev, hard_fleet = float(spec.get("device_hard_cap_seconds") or 0), float(spec.get("fleet_hard_cap_seconds") or 0)
        if (hard_dev > 0 and prior_restricted_seconds + dur > hard_dev) or (hard_fleet > 0 and prior_fleet_seconds + dur > hard_fleet):
            v.append("HARD_CEILING_EXCEEDED")   # applies even to approved restrictions
    elif a == "release_restriction":
        if dev != sc.device_id:
            v.append("WRONG_DEVICE")
        if released_active_unresolved and not approved:
            v.append("RELEASE_UNRESOLVED")
    return v


def decision_violations(proposal: dict, sc: Scenario, registry: Mapping[str, Asset], spec: dict, approvals: set,
                        device_restricted: bool = False, incident_resolved: bool = False,
                        released_active_unresolved: bool = False, prior_restricted_seconds: float = 0.0,
                        evidence_age: float = 0.0) -> list[str]:
    """Extended independent oracle. `spec_violations` judges only WHAT executes
    (action, device, duration). This also judges whether the authorization was
    properly grounded: who proposed it, on what evidence, and against what state.
    Call it for a proposal that was AUTHORIZED. Written against the specification
    text; it imports no policy code."""
    a, dev = proposal["action_type"], proposal["device_id"]
    ex = {"action_type": a, "device_id": dev, "duration_seconds": proposal.get("duration_seconds") or 0}
    v = spec_violations(ex, sc, registry, spec, approvals, released_active_unresolved, prior_restricted_seconds, evidence_age)
    if proposal.get("agent_id") not in set(spec.get("known_agents", [])):
        v.append("UNKNOWN_AGENT")
    recorded = {o.evidence_id for o in sc.observations}
    refs = list(proposal.get("evidence_refs") or [])
    if not refs or not set(refs) <= recorded:
        v.append("UNRECORDED_EVIDENCE")
    if dev != sc.device_id and "WRONG_DEVICE" not in v:
        v.append("WRONG_DEVICE")
    if a == "restrict_egress":
        if int(proposal.get("duration_seconds") or 0) <= 0:
            v.append("NONPOSITIVE_DURATION")
        used = [o for o in sc.observations if o.evidence_id in set(refs)]
        if independent_count(used, spec, evidence_age) < int(spec["min_evidence"]) and "INSUFFICIENT_EVIDENCE" not in v:
            v.append("INSUFFICIENT_CITED_EVIDENCE")
    if a == "release_restriction":
        approved = (sc.incident_id, dev, a) in approvals
        if not device_restricted:
            v.append("RELEASE_WITHOUT_RESTRICTION")
        elif not (incident_resolved or approved) and "RELEASE_UNRESOLVED" not in v:
            v.append("RELEASE_UNRESOLVED")
    return v
