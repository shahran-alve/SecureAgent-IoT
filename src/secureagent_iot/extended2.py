"""Second batch of extended evaluations. TESTBED-SYNTHETIC, scripted agents, no LLM.

E8  design-pattern comparison: re-implementations of prior response/defense
    PATTERNS inside this testbed (not the original systems of any cited paper)
E9  exposure: how long truly malicious incidents wait, per pattern
E10 campaign risk: probability of at least one prohibited execution over N attempts
E11 seed convergence (30 seeds)
E12 outcome breakdown by scenario family       E13 policy decision reason codes
E14 queue mitigations versus number of compromised devices
E15 audit-log tamper detection                 E16 stress on trust assumptions
"""
from __future__ import annotations

import copy
import heapq
import json

import numpy as np
import pandas as pd

from .agents import AgenticSystem, B0Rules, _blank, _exec_dict
from .attacks import build_cases, run_attempt
from .audit import AuditLog, GENESIS
from .config import rng_for
from .evaluation import EVIDENCE_CLASS, cluster_bootstrap_ci, run_clean_once, run_response_benchmark
from .schemas import Asset
from .simulator import HIGH, build_registry, build_templates

SHIFT_S = 8 * 3600


# ------------------------------------------------------------------ pattern systems
class AdvisoryOnly:
    """Pattern: the automation only advises; a human takes every action."""
    name = "P1_advisory_only"

    def __init__(self, cfg):
        self.memory = []

    def handle(self, sc, tb, rng=None, untrusted_text=None, retrieved_text=None):
        res = _blank(self.name)
        out = tb.controller.request_review(sc.device_id, sc.incident_id)
        res.update(proposal=_exec_dict("request_review", sc.device_id, 0), decisions=["ADVISORY"], escalated=True,
                   executed=_exec_dict("request_review", sc.device_id, 0), action_latency_ms_sim=out.elapsed_ms)
        return res


class UnboundedRules:
    """Pattern: fixed thresholds trigger blocking with no expiry, no criticality
    gate and no approval step (blocks last until someone lifts them)."""
    name = "P2_unbounded_rules"

    def __init__(self, cfg):
        self.thr = float(cfg["rules_baseline"]["restrict_threshold"])
        self.memory = []

    def handle(self, sc, tb, rng=None, untrusted_text=None, retrieved_text=None):
        res = _blank(self.name)
        act, dur = ("restrict_egress", 86_400) if sc.p_attack >= self.thr else ("notify", 0)
        out = tb.controller.execute_raw(act, sc.device_id, dur, sc.incident_id)
        res.update(proposal=_exec_dict(act, sc.device_id, dur), decisions=["RULE_UNBOUNDED"], action_latency_ms_sim=out.elapsed_ms)
        if out.executed:
            res["executed"] = _exec_dict(act, sc.device_id, dur)
        return res


class InputFilter:
    """Pattern: a probabilistic injection detector in front of the agent. With
    probability `recall` it recognises injected text and strips it; otherwise the
    agent sees it. False positives on clean text are assumed to be zero (generous)."""

    def __init__(self, inner, recall: float, name: str):
        self.inner, self.recall, self.name = inner, float(recall), name
        inner.name = name

    @property
    def memory(self):
        return self.inner.memory

    @memory.setter
    def memory(self, v):
        self.inner.memory = v

    def handle(self, sc, tb, rng, untrusted_text=None, retrieved_text=None):
        if (untrusted_text is not None or retrieved_text is not None) and \
                rng_for("filter", self.name, sc.incident_id, self.recall).random() < self.recall:
            untrusted_text = retrieved_text = None
        return self.inner.handle(sc, tb, rng, untrusted_text=untrusted_text, retrieved_text=retrieved_text)


PATTERN_LABEL = {
    "P1_advisory_only": "Advisory only (human executes)", "P2_unbounded_rules": "Autonomous rules, unbounded blocks",
    "B0_rules": "Autonomous rules, bounded (B0)", "B1_unguarded": "Autonomous agent, unguarded (B1)",
    "P4_agent_filter": "Agent + injection filter (recall 0.90)", "B1v_schema_only": "Agent + schema validation (B1v)",
    "B2_guarded": "Agent + policy engine (B2, ours)", "P7_filter_policy": "Agent + filter + policy engine",
}


def pattern_systems(cfg, recall: float = 0.90) -> dict:
    return {
        "P1_advisory_only": AdvisoryOnly(cfg), "P2_unbounded_rules": UnboundedRules(cfg), "B0_rules": B0Rules(cfg),
        "B1_unguarded": AgenticSystem("B1_unguarded", cfg, guarded=False),
        "P4_agent_filter": InputFilter(AgenticSystem("x", cfg, guarded=False), recall, "P4_agent_filter"),
        "B1v_schema_only": AgenticSystem("B1v_schema_only", cfg, guarded=False, validate_schema=True),
        "B2_guarded": AgenticSystem("B2_guarded", cfg, guarded=True),
        "P7_filter_policy": InputFilter(AgenticSystem("x", cfg, guarded=True), recall, "P7_filter_policy"),
    }


def _attack_frame(cfg, seeds, sysmap, registry=None, variants=None) -> pd.DataFrame:
    reg_true = build_registry()
    registry = registry or reg_true
    templates = build_templates()
    cases = build_cases(templates, reg_true, cfg["spec"])
    if variants:
        cases = [(v, t) for v, t in cases if v.attack_id in variants]
    rows = []
    for seed in seeds:
        for v, t in cases:
            for name, system in sysmap.items():
                for inject in (True, False):
                    rows.append(run_attempt(system, v, t, seed, cfg, registry, templates, inject=inject))
    return pd.DataFrame(rows)


def _wrongful_seconds_clean(d: pd.DataFrame) -> pd.Series:
    bad = (d["executed_action"] == "restrict_egress") & ((~d["truth_malicious"]) | (d["criticality"] == "high"))
    return np.where(bad, np.minimum(d["executed_duration"].fillna(0), SHIFT_S), 0.0)


def _wrongful_seconds_attack(d: pd.DataFrame) -> pd.Series:
    bad = (d["executed_action"] == "restrict_egress") & (d["benign_isolated"] | d["high_crit_isolated_no_approval"])
    return np.where(bad, np.minimum(d["executed_duration"].fillna(0), SHIFT_S), 0.0)


# ------------------------------------------------------------------ E8 + E9
def design_pattern_comparison(cfg, seeds):
    sysmap = pattern_systems(cfg)
    clean = run_response_benchmark(cfg, seeds, sysmap=sysmap)
    atk = _attack_frame(cfg, seeds, sysmap)
    rows = []
    for name in sysmap:
        c = clean[clean.system == name]
        a = atk[(atk.system == name) & atk.injected]
        ctl = atk[(atk.system == name) & ~atk.injected]
        e, sup = a[~a.suppression_scored], a[a.suppression_scored]
        cw = c[c.containment_worthy]
        lo, hi = cluster_bootstrap_ci(e.assign(_c=e.attack_id + "|" + e.scenario_id).groupby("_c")["any_violation"].mean().values,
                                      int(cfg["bootstrap"]["n_resamples"]), rng_for("pat", name))
        rows.append({
            "evidence_class": EVIDENCE_CLASS, "system": name, "pattern": PATTERN_LABEL[name],
            "safe_containment": cw["safe_contained"].mean(), "autonomous_containment": cw["contained"].mean(),
            "false_isolation": c.loc[~c.truth_malicious, "false_isolation"].mean(), "unauthorized_clean": c["spec_violation"].mean(),
            "review_rate_clean": c["escalated"].mean(), "missed_malicious": c.loc[c.truth_malicious, "missed"].mean(),
            "wrongful_isolation_s_per_100_clean": 100 * _wrongful_seconds_clean(c).mean(),
            "violation_rate_attacked": e["any_violation"].mean(), "violation_ci_lo": lo, "violation_ci_hi": hi,
            "violation_rate_control": ctl.loc[~ctl.suppression_scored, "any_violation"].mean(),
            "attack_goal_executed": e["goal_executed"].mean(),
            "wrongful_isolation_s_per_100_attacked": 100 * _wrongful_seconds_attack(e).mean(),
            "suppression_rate": sup["suppression_success"].mean(), "review_rate_attacked": a["escalated"].mean(),
            "n_clean": len(c), "n_attacked": len(e)})
    return pd.DataFrame(rows), clean, atk


def _simulate_shift(events, discipline, analysts, service_min, svc_rng, horizon):
    """events: list of (t, malicious, p, device). Returns (waits of reviewed malicious tickets, malicious never reviewed, backlog, tickets)."""
    servers = [0.0] * analysts
    heapq.heapify(servers)
    queue, open_ticket, k, seq, n_tickets = [], {}, 0, 0, 0
    w_mal, pending, left = [], 0, 0

    def admit(ev):
        nonlocal seq, n_tickets
        t, mal, p, dev = ev
        if discipline.endswith("coalesce") and dev in open_ticket:
            it = open_ticket[dev]
            it[3].append((t, mal))
            if -p < it[0]:
                it[0] = -p
                heapq.heapify(queue)
            return
        item = [t if discipline == "fifo" else -p, seq, t, [(t, mal)], dev]
        seq += 1; n_tickets += 1
        heapq.heappush(queue, item)
        if discipline.endswith("coalesce"):
            open_ticket[dev] = item

    while k < len(events) or queue:
        now = heapq.heappop(servers)
        if not queue:
            now = max(now, events[k][0])
        while k < len(events) and events[k][0] <= now:
            admit(events[k]); k += 1
        if now > horizon:
            pending = sum(m for it in queue for _, m in it[3]) + sum(e[1] for e in events[k:])
            left = len(queue) + len(events) - k
            break
        item = heapq.heappop(queue)
        open_ticket.pop(item[4], None)
        w_mal += [now - t for t, m in item[3] if m]
        heapq.heappush(servers, now + svc_rng.exponential(service_min))
    return w_mal, pending, left, n_tickets


def exposure_by_pattern(clean: pd.DataFrame, atk: pd.DataFrame, reps=300, hours=8.0, rate=30.0, analysts=2, service_min=6.0,
                        attack_fraction=0.3, fleet=200, attacker_devices=5, discipline="priority"):
    """Time until a truly malicious incident gets a response: 0 if contained
    autonomously, the queue wait if escalated, never (censored at shift end) if dropped."""
    rows, dist = [], []
    horizon = hours * 60.0
    for name in clean.system.unique():
        c = clean[clean.system == name][["escalated", "truth_malicious", "p_attack", "contained"]].to_numpy()
        a_ = atk[(atk.system == name) & atk.injected]
        a = a_.assign(contained=(a_.executed_action == "restrict_egress") & a_.truth_malicious & ~a_.benign_isolated)[
            ["escalated", "truth_malicious", "p_attack", "contained"]].to_numpy()
        allw, never, auto, total, tick = [], 0, 0, 0, []
        for r in range(reps):
            rng = rng_for("expo", attack_fraction, r)
            n = rng.poisson(rate * hours)
            t_arr = np.sort(rng.random(n) * horizon)
            is_att = rng.random(n) < attack_fraction
            events = []
            for i in range(n):
                pool = a if is_att[i] else c
                esc, mal, p, cont = pool[int(rng.integers(len(pool)))]
                dev = int(rng.integers(attacker_devices)) if is_att[i] else int(attacker_devices + rng.integers(fleet - attacker_devices))
                total += bool(mal)
                if mal and cont:
                    auto += 1; allw.append(0.0)
                elif esc:
                    events.append((t_arr[i], bool(mal), float(p), dev))
                elif mal:
                    never += 1
            w, pend, _, nt = _simulate_shift(events, discipline, analysts, service_min, rng_for("expo-svc", attack_fraction, r), horizon)
            allw += w; never += pend; tick.append(nt)
        w = np.array(allw)
        within = lambda m: float((w <= m).sum() / total)
        rows.append({"evidence_class": EVIDENCE_CLASS, "system": name, "pattern": PATTERN_LABEL.get(name, name),
                     "attack_fraction": attack_fraction, "malicious_incidents": total, "handled_autonomously": auto / total,
                     "never_handled_in_shift": never / total, "responded_within_1min": within(1), "responded_within_15min": within(15),
                     "responded_within_60min": within(60), "tickets_per_shift": float(np.mean(tick))})
        for m in (0, 1, 2, 5, 10, 15, 30, 60, 120, 240, 480):
            dist.append({"system": name, "minutes": m, "share_responded": within(m)})
    return pd.DataFrame(rows), pd.DataFrame(dist)


# ------------------------------------------------------------------ filter recall sweep (feeds E10)
def filter_recall_sweep(cfg, seeds, recalls=(0.0, 0.5, 0.8, 0.9, 0.95, 0.99)):
    rows = []
    for r in recalls:
        sysmap = {"P4_agent_filter": InputFilter(AgenticSystem("x", cfg, guarded=False), r, "P4_agent_filter"),
                  "P7_filter_policy": InputFilter(AgenticSystem("x", cfg, guarded=True), r, "P7_filter_policy")}
        a = _attack_frame(cfg, seeds, sysmap)
        for name in sysmap:
            d = a[(a.system == name) & a.injected]
            e, sup = d[~d.suppression_scored], d[d.suppression_scored]
            rows.append({"evidence_class": EVIDENCE_CLASS, "recall": r, "system": name, "exec_asr": e.exec_success.mean(),
                         "goal_executed": e.goal_executed.mean(), "suppression_rate": sup.suppression_success.mean(),
                         "review_rate_attacked": d.escalated.mean(), "n": len(e)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E10
REPORTED_PRIOR = [
    # label, per-attempt attack success rate as REPORTED in the cited paper (different agents, tasks and attacks)
    ("InjecAgent: GPT-4 ReAct agent, no defense", 0.24, "injecagent"),
    ("AgentDojo: agent + secondary attack detector", 0.08, "agentdojo"),
    ("Progent on AgentDojo (v1)", 0.022, "progent"),
    ("MetaSecAlign, static attacks", 0.02, "nasr2025"),
    ("MetaSecAlign, adaptive attacks", 0.96, "nasr2025"),
]


def campaign_risk(per_attempt: dict[str, float], ns=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)) -> pd.DataFrame:
    rows = []
    for label, a in per_attempt.items():
        for n in ns:
            rows.append({"label": label, "per_attempt_rate": a, "attempts": n, "p_at_least_one": 1.0 - (1.0 - a) ** n})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E11
def seed_convergence(cfg, n_seeds=30):
    seeds = list(range(1000, 1000 + n_seeds))
    df = run_response_benchmark(cfg, seeds, systems=["B0_rules", "B1_unguarded", "B2_guarded"])
    rows = []
    for system, d in df.groupby("system", sort=False):
        for metric, col, mask in (("safe_containment", "safe_contained", d.containment_worthy), ("unauthorized_action", "spec_violation", None),
                                  ("review_rate", "escalated", None), ("false_isolation", "false_isolation", ~d.truth_malicious)):
            g = d if mask is None else d[mask]
            per_seed = g.groupby("seed")[col].mean().reindex(seeds).to_numpy(dtype=float)
            for k in range(2, n_seeds + 1):
                x = per_seed[:k]
                rows.append({"evidence_class": EVIDENCE_CLASS, "system": system, "metric": metric, "n_seeds": k,
                             "running_mean": x.mean(), "seed_sd": x.std(ddof=1), "half_width_95": 1.96 * x.std(ddof=1) / np.sqrt(k)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E12, E13
def family_breakdown(runs: pd.DataFrame) -> pd.DataFrame:
    d = runs[runs.system.isin(["B0_rules", "B1_unguarded", "B2_guarded", "B3_single_guarded"])]
    g = d.groupby(["system", "family"], sort=False).agg(appropriate=("appropriate", "mean"), unauthorized=("spec_violation", "mean"),
                                                         review=("escalated", "mean"), restrict=("executed_action", lambda s: (s == "restrict_egress").mean()),
                                                         utility=("utility", "mean"), n=("utility", "size")).reset_index()
    g.insert(0, "evidence_class", EVIDENCE_CLASS)
    return g


def reason_codes(runs: pd.DataFrame, atk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, d in (("clean", runs[runs.system == "B2_guarded"]), ("attacked", atk[(atk.system == "B2_guarded") & atk.injected])):
        final = d["policy_decisions"].astype(str).str.split(">").str[-1].str.replace(r"DENY_SCHEMA_INVALID|DENY_UNREGISTERED_DEVICE", "DENY_SCHEMA_OR_REGISTRY", regex=True)
        vc = final.value_counts(normalize=True)
        for code, share in vc.items():
            rows.append({"evidence_class": EVIDENCE_CLASS, "condition": cond, "reason_code": code, "share": share, "n": len(d),
                         "outcome": "allow" if code.startswith("ALLOW") else ("review" if code.startswith("REVIEW") else "deny")})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E14
def coalescing_vs_compromised(clean, atk, device_counts=(1, 5, 20, 50, 100, 200), attack_fraction=0.5, reps=300, fleet=200,
                              hours=8.0, rate=30.0, analysts=2, service_min=6.0, sla=15.0):
    c = clean[clean.system == "B2_guarded"][["escalated", "truth_malicious", "p_attack"]].to_numpy()
    a = atk[(atk.system == "B2_guarded") & atk.injected][["escalated", "truth_malicious", "p_attack"]].to_numpy()
    rows, horizon = [], hours * 60.0
    for k in device_counts:
        for disc in ("fifo", "priority", "priority+coalesce"):
            miss, tick = [], []
            for r in range(reps):
                rng = rng_for("coal", k, r)
                n = rng.poisson(rate * hours)
                t_arr = np.sort(rng.random(n) * horizon)
                is_att = rng.random(n) < attack_fraction
                ev = []
                for i in range(n):
                    pool = a if is_att[i] else c
                    esc, mal, p = pool[int(rng.integers(len(pool)))]
                    dev = int(rng.integers(k)) if is_att[i] else int(rng.integers(fleet))
                    if esc:
                        ev.append((t_arr[i], bool(mal), float(p), dev if is_att[i] else 10_000 + dev))
                w, pend, _, nt = _simulate_shift(ev, disc, analysts, service_min, rng_for("coal-svc", k, r), horizon)
                tot = len(w) + pend
                miss.append((sum(x > sla for x in w) + pend) / tot if tot else 0.0); tick.append(nt)
            rows.append({"evidence_class": EVIDENCE_CLASS, "compromised_devices": k, "discipline": disc, "attack_fraction": attack_fraction,
                         "malicious_sla_miss": float(np.mean(miss)), "tickets_per_shift": float(np.mean(tick))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E15
def audit_tamper(n_records=1000, trials=300) -> pd.DataFrame:
    def build():
        log = AuditLog(now=lambda: "t")
        for j in range(n_records):
            log.append(incident_id=f"inc-{j}", evidence_ids=["e1", "e2"], action={"action_type": "notify", "device_id": "cam_01"},
                       decision={"allowed": True}, policy_version="1.0.0")
        return [dict(r) for r in log.records], log.head

    base, head = build()
    rng = np.random.default_rng(15)

    def rechain(recs):
        from .audit import _canon, _sha
        prev = GENESIS
        for i, r in enumerate(recs):
            r["seq"], r["prev_hash"] = i, prev
            body = {k: v for k, v in r.items() if k != "record_hash"}
            r["record_hash"] = _sha(_canon(body)); prev = r["record_hash"]
        return recs

    def tamper(kind):
        recs = [dict(r) for r in base]
        i = int(rng.integers(1, n_records - 1))
        if kind == "edit one record":
            recs[i]["decision"] = {"allowed": False}
        elif kind == "delete one record":
            del recs[i]
        elif kind == "swap two records":
            j = int(rng.integers(1, n_records - 1)); recs[i], recs[j] = recs[j], recs[i]
            if i == j:
                recs[i]["decision"] = {"allowed": False}
        elif kind == "truncate the tail":
            recs = recs[:i]
        elif kind == "edit and rebuild all hashes":
            recs[i]["decision"] = {"allowed": False}; recs = rechain(recs)
        return recs

    rows = []
    for kind in ("edit one record", "delete one record", "swap two records", "truncate the tail", "edit and rebuild all hashes"):
        chain = anchored = 0
        for _ in range(trials):
            recs = tamper(kind)
            ok, _ = AuditLog.verify_records(recs)
            chain += not ok
            anchored += (not ok) or (recs[-1]["record_hash"] != head if recs else True)
        rows.append({"evidence_class": "measured (deterministic check)", "tampering": kind, "trials": trials,
                     "detected_chain_only": chain / trials, "detected_with_anchored_head": anchored / trials})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E16
def trust_assumption_stress(cfg, seeds):
    """Outside the threat model on purpose: what happens when trusted state is wrong."""
    reg_true = build_registry()
    rows = []
    # (a) the attacker also controls the detector for a share of incidents
    for pi in (0.0, 0.1, 0.25, 0.5, 1.0):
        c = copy.deepcopy(cfg); c["poison_p_share"] = pi
        sysmap = {"B1_unguarded": AgenticSystem("B1_unguarded", c, guarded=False), "B2_guarded": AgenticSystem("B2_guarded", c, guarded=True)}
        a = _attack_frame(c, seeds, sysmap, variants={"A1.3", "A4.1"})
        for name in sysmap:
            d = a[(a.system == name) & a.injected]
            rows.append({"evidence_class": EVIDENCE_CLASS, "stress": "detector probability poisoned", "level": pi, "system": name,
                         "harm": "benign device isolated", "harm_rate": d.benign_isolated.mean(), "spec_violation_rate": d.any_violation.mean(), "n": len(d)})
    # (b) the registry mislabels high-criticality devices as low
    for k in (0, 1, 2, 3):
        reg = dict(reg_true)
        for dev in ["actuator_01", "actuator_02", "pump_01"][:k]:
            reg[dev] = Asset(device_id=dev, criticality="low", allowed_actions=list(reg_true[dev].allowed_actions), owner="plant-safety", is_online=True)
        sysmap = {"B1_unguarded": AgenticSystem("B1_unguarded", cfg, guarded=False), "B2_guarded": AgenticSystem("B2_guarded", cfg, guarded=True)}
        a = _attack_frame(cfg, seeds, sysmap, registry=reg, variants={"A2.2", "A3.1", "A4.2"})
        for name in sysmap:
            d = a[(a.system == name) & a.injected]
            harm = (d.executed_action == "restrict_egress") & d.executed_device.isin(HIGH)
            rows.append({"evidence_class": EVIDENCE_CLASS, "stress": "critical devices mislabelled as low", "level": k / 3, "system": name,
                         "harm": "truly critical device isolated", "harm_rate": harm.mean(), "spec_violation_rate": d.any_violation.mean(), "n": len(d)})
    # (c) the attacker hides real attacks from the detector (forced false negatives)
    for pi in (0.0, 0.1, 0.25, 0.5, 1.0):
        c = copy.deepcopy(cfg); c["suppress_p_share"] = pi
        d = run_response_benchmark(c, seeds, systems=["B0_rules", "B1_unguarded", "B2_guarded"])
        for name, g in d.groupby("system", sort=False):
            mal = g[g.truth_malicious]
            rows.append({"evidence_class": EVIDENCE_CLASS, "stress": "detector false negatives forced", "level": pi, "system": name,
                         "harm": "malicious incident neither contained nor escalated", "harm_rate": mal.missed.mean(),
                         "spec_violation_rate": g.spec_violation.mean(), "n": len(mal)})
    return pd.DataFrame(rows)
