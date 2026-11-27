"""Formal verification of the policy decision function with the Z3 SMT solver.

`policy_z3` is a symbolic transcription of `PolicyEngine.evaluate` (policy.py),
branch for branch. Every input is symbolic: agent identity, registration,
incident binding, evidence resolution, the number of independent fresh
observations cited and recorded, action type, allowlist, criticality, approval
validity, resolution and restriction state, detector probability, duration,
restriction time already used in the window, and the applicable budget. The
written specification (`spec_*`) is encoded separately from the same text the
oracles in simulator.py implement.

Two checks make the proof meaningful:
1. Safety and liveness theorems are proved for all inputs (UNSAT of the negation).
2. The transcription is tested against the real engine on random concrete
   inputs (`differential_test`); any mismatch is reported, so the proof cannot
   silently be about a different function.

Ablations are checked too: for each removed check the solver must find a
counterexample, which shows the theorem depends on that check.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import z3

from .policy import IncidentRecord, PolicyEngine, PolicyFlags, TrustedState
from .schemas import ProposedAction

ACTIONS = ["request_review", "notify", "release_restriction", "restrict_egress"]
CODES = ["DENY_UNKNOWN_AGENT", "DENY_UNREGISTERED_DEVICE", "DENY_UNKNOWN_INCIDENT", "DENY_DEVICE_MISMATCH",
         "DENY_UNRESOLVED_EVIDENCE", "DENY_ACTION_NOT_ALLOWLISTED", "ALLOW_REVIEW", "REVIEW_ESCALATION_FLOOR",
         "ALLOW_NOTIFY", "ALLOW_RELEASE", "REVIEW_RELEASE_UNRESOLVED", "DENY_NOTHING_TO_RELEASE",
         "DENY_INVALID_DURATION", "DENY_TTL_EXCEEDS_CAP", "ALLOW_APPROVED_HIGH_IMPACT", "REVIEW_HIGH_CRITICALITY",
         "DENY_BELOW_THRESHOLD", "REVIEW_UNCERTAIN", "REVIEW_INSUFFICIENT_EVIDENCE", "ALLOW_APPROVED_OVER_BUDGET",
         "REVIEW_DEVICE_BUDGET", "ALLOW_RESTRICT", "REVIEW_FLEET_BUDGET", "ALLOW_CORROBORATED", "DENY_HARD_CEILING"]
CODE = {c: i for i, c in enumerate(CODES)}


@dataclass
class Sym:
    """Symbolic inputs to one decision."""
    agent_known: z3.BoolRef
    registered: z3.BoolRef
    incident_known: z3.BoolRef
    device_match: z3.BoolRef
    refs_recorded: z3.BoolRef
    n_cited: z3.ArithRef          # independent, fresh observations among the cited refs
    n_raw: z3.ArithRef            # distinct cited refs (what an engine without provenance counts)
    n_recorded: z3.ArithRef       # independent, fresh observations recorded for the incident
    action: z3.ArithRef           # index into ACTIONS
    allowlisted: z3.BoolRef
    high: z3.BoolRef              # asset criticality is high
    approved: z3.BoolRef          # a valid (bound, unexpired, unconsumed) approval exists
    resolved: z3.BoolRef
    restricted: z3.BoolRef
    p: z3.ArithRef
    dur: z3.ArithRef
    used: z3.ArithRef             # restriction seconds already used in the budget window
    budget: z3.ArithRef           # budget for this asset's criticality
    fleet_used: z3.ArithRef       # restriction seconds granted to any device in the window
    fleet_budget: z3.ArithRef     # fleet-wide cap (0 = off)
    corroborated: z3.BoolRef      # corroboration exemption on, and the indicator was seen on enough other devices
    exempt_used: z3.ArithRef      # seconds granted under the corroboration exemption in the window
    corr_budget: z3.ArithRef      # budget for exempt restrictions (0 = unbounded exemption)
    hard_dev: z3.ArithRef         # device hard ceiling (0 = off)
    hard_fleet: z3.ArithRef       # fleet hard ceiling (0 = off)

    @staticmethod
    def fresh(tag: str = "") -> "Sym":
        B, I, R = (lambda n: z3.Bool(n + tag)), (lambda n: z3.Int(n + tag)), (lambda n: z3.Real(n + tag))
        return Sym(B("agent_known"), B("registered"), B("incident_known"), B("device_match"), B("refs_recorded"),
                   I("n_cited"), I("n_raw"), I("n_recorded"), I("action"), B("allowlisted"), B("high"), B("approved"),
                   B("resolved"), B("restricted"), R("p"), I("dur"), R("used"), R("budget"), R("fleet_used"), R("fleet_budget"), B("corroborated"),
                   R("exempt_used"), R("corr_budget"), R("hard_dev"), R("hard_fleet"))

    def domain(self):
        """Constraints every real input satisfies."""
        return z3.And(self.action >= 0, self.action <= 3, self.p >= 0, self.p <= 1, self.used >= 0, self.budget >= 0, self.fleet_used >= 0, self.fleet_budget >= 0,
                      self.exempt_used >= 0, self.corr_budget >= 0, self.hard_dev >= 0, self.hard_fleet >= 0,
                      self.n_cited >= 0, self.n_raw >= self.n_cited, self.n_recorded >= 0,
                      # cited observations that resolve are a subset of the recorded ones
                      z3.Implies(self.refs_recorded, self.n_cited <= self.n_recorded))


def policy_z3(s: Sym, cfg: dict, flags: PolicyFlags = PolicyFlags()):
    """Symbolic PolicyEngine.evaluate. Returns (allowed, requires_approval, code) as Z3 terms."""
    P = cfg["policy"]
    ttl, thr, floor, mine = int(P["max_ttl_seconds"]), float(P["p_attack_threshold"]), float(P["review_floor"]), int(P["min_evidence"])

    def d(a, r, c):
        return (z3.BoolVal(a), z3.BoolVal(r), z3.IntVal(CODE[c]))

    def ite(c, x, y):
        return tuple(z3.If(c, xi, yi) for xi, yi in zip(x, y))

    n_ev = s.n_cited if flags.provenance else s.n_raw
    restrict_tail = d(True, False, "ALLOW_RESTRICT")
    if flags.budget:
        restrict_tail = ite(z3.And(s.fleet_budget > 0, s.fleet_used + s.dur > s.fleet_budget),
                            ite(z3.And(s.corroborated, z3.Or(s.corr_budget <= 0, s.exempt_used + s.dur <= s.corr_budget)),
                                d(True, False, "ALLOW_CORROBORATED"),
                                ite(s.approved, d(True, False, "ALLOW_APPROVED_OVER_BUDGET"), d(False, True, "REVIEW_FLEET_BUDGET"))),
                            restrict_tail)
    if flags.budget:
        restrict_tail = ite(s.used + s.dur > s.budget,
                            ite(s.approved, d(True, False, "ALLOW_APPROVED_OVER_BUDGET"), d(False, True, "REVIEW_DEVICE_BUDGET")),
                            restrict_tail)
    if flags.criticality:
        restrict_tail = ite(s.high, ite(s.approved, d(True, False, "ALLOW_APPROVED_HIGH_IMPACT"), d(False, True, "REVIEW_HIGH_CRITICALITY")), restrict_tail)
    restrict_tail = ite(z3.Or(z3.And(s.hard_dev > 0, s.used + s.dur > s.hard_dev),
                              z3.And(s.hard_fleet > 0, s.fleet_used + s.dur > s.hard_fleet)),
                        d(False, False, "DENY_HARD_CEILING"), restrict_tail)
    restrict = restrict_tail
    if flags.thresholds:
        restrict = ite(s.p < floor, d(False, False, "DENY_BELOW_THRESHOLD"),
                   ite(s.p < thr, d(False, True, "REVIEW_UNCERTAIN"),
                   ite(n_ev < mine, d(False, True, "REVIEW_INSUFFICIENT_EVIDENCE"), restrict_tail)))
    if flags.ttl_cap:
        restrict = ite(s.dur > ttl, d(False, False, "DENY_TTL_EXCEEDS_CAP"), restrict)
    restrict = ite(s.dur <= 0, d(False, False, "DENY_INVALID_DURATION"), restrict)

    release = ite(z3.Not(s.restricted), d(False, False, "DENY_NOTHING_TO_RELEASE"),
              ite(z3.Or(s.resolved, s.approved), d(True, False, "ALLOW_RELEASE"), d(False, True, "REVIEW_RELEASE_UNRESOLVED")))
    notify = d(True, False, "ALLOW_NOTIFY")
    if flags.escalation_floor:
        notify = ite(z3.And(s.p >= thr, s.n_recorded >= mine), d(False, True, "REVIEW_ESCALATION_FLOOR"), notify)

    body = ite(s.action == 0, d(True, False, "ALLOW_REVIEW"),
           ite(s.action == 1, notify, ite(s.action == 2, release, restrict)))
    body = ite(z3.Not(s.allowlisted), d(False, False, "DENY_ACTION_NOT_ALLOWLISTED"), body)
    if flags.provenance:
        body = ite(z3.Not(s.device_match), d(False, False, "DENY_DEVICE_MISMATCH"),
               ite(z3.Not(s.refs_recorded), d(False, False, "DENY_UNRESOLVED_EVIDENCE"), body))
    body = ite(z3.Not(s.incident_known), d(False, False, "DENY_UNKNOWN_INCIDENT"), body)
    body = ite(z3.Not(s.registered), d(False, False, "DENY_UNREGISTERED_DEVICE"), body)
    return ite(z3.Not(s.agent_known), d(False, False, "DENY_UNKNOWN_AGENT"), body)


# ---------------------------------------------------------------------------
# Written specification (independent of the engine; same text as the oracles)
# ---------------------------------------------------------------------------
def spec_ok(s: Sym, cfg: dict):
    """An authorized decision must satisfy all of these (decision-level specification)."""
    S = cfg["spec"]
    ttl, thr, mine = int(S["max_ttl_seconds"]), float(S["p_attack_threshold"]), int(S["min_evidence"])
    grounded = z3.And(s.agent_known, s.registered, s.incident_known, s.device_match, s.refs_recorded, s.allowlisted)
    restrict = z3.And(s.dur > 0, s.dur <= ttl, z3.Implies(s.high, s.approved), s.p >= thr, s.n_cited >= mine,
                      z3.Implies(s.used + s.dur > s.budget, s.approved),
                      z3.Implies(z3.And(s.fleet_budget > 0, s.fleet_used + s.dur > s.fleet_budget),
                                 z3.Or(s.approved, z3.And(s.corroborated, z3.Or(s.corr_budget <= 0, s.exempt_used + s.dur <= s.corr_budget)))),
                      z3.Or(s.hard_dev <= 0, s.used + s.dur <= s.hard_dev),
                      z3.Or(s.hard_fleet <= 0, s.fleet_used + s.dur <= s.hard_fleet))
    release = z3.And(s.restricted, z3.Or(s.resolved, s.approved))
    return z3.And(grounded, z3.If(s.action == 3, restrict, z3.If(s.action == 2, release, z3.BoolVal(True))))


THEOREMS = {
    "S1: Every authorization satisfies the written specification":
        lambda s, out, cfg: z3.Implies(out[0], spec_ok(s, cfg)),
    "S2: No action is both allowed and sent to review":
        lambda s, out, cfg: z3.Not(z3.And(out[0], out[1])),
    "S3: No autonomous restriction pushes a device past its budget":
        lambda s, out, cfg: z3.Implies(z3.And(out[0], s.action == 3, z3.Not(s.approved)), s.used + s.dur <= s.budget),
    "S4: No high-criticality device is restricted without a valid approval":
        lambda s, out, cfg: z3.Implies(z3.And(out[0], s.action == 3, s.high), s.approved),
    "S5: No authorized restriction exceeds the duration cap, even with approval":
        lambda s, out, cfg: z3.Implies(z3.And(out[0], s.action == 3), s.dur <= int(cfg["spec"]["max_ttl_seconds"])),
    "S6: With a fleet cap set, no autonomous restriction pushes the fleet past it unless corroborated within the exemption budget":
        lambda s, out, cfg: z3.Implies(z3.And(out[0], s.action == 3, z3.Not(s.approved), s.fleet_budget > 0, s.fleet_used + s.dur > s.fleet_budget),
                                       z3.And(s.corroborated, z3.Or(s.corr_budget <= 0, s.exempt_used + s.dur <= s.corr_budget))),
    "S7: No restriction, approved or corroborated, pushes a device or the fleet past a hard ceiling":
        lambda s, out, cfg: z3.Implies(z3.And(out[0], s.action == 3),
                                       z3.And(z3.Or(s.hard_dev <= 0, s.used + s.dur <= s.hard_dev),
                                              z3.Or(s.hard_fleet <= 0, s.fleet_used + s.dur <= s.hard_fleet))),
    "A1: A restriction that meets every condition is allowed without a human":
        lambda s, out, cfg: z3.Implies(z3.And(spec_ok(s, cfg), s.action == 3, z3.Not(s.high), s.used + s.dur <= s.budget,
                                              z3.Or(s.fleet_budget == 0, s.fleet_used + s.dur <= s.fleet_budget)), out[0]),
    "A4: A corroborated restriction within its device and exemption budgets is allowed even when the fleet cap is spent":
        lambda s, out, cfg: z3.Implies(z3.And(spec_ok(s, cfg), s.action == 3, z3.Not(s.high), s.used + s.dur <= s.budget,
                                              s.corroborated, z3.Or(s.corr_budget <= 0, s.exempt_used + s.dur <= s.corr_budget)), out[0]),
    "A2: A notification for a strong, well-evidenced alert is escalated, not accepted":
        lambda s, out, cfg: z3.Implies(z3.And(s.agent_known, s.registered, s.incident_known, s.device_match, s.refs_recorded,
                                              s.allowlisted, s.action == 1, s.p >= float(cfg["spec"]["p_attack_threshold"]),
                                              s.n_recorded >= int(cfg["spec"]["min_evidence"])), out[1]),
    "A3: A review request from a grounded agent is always accepted":
        lambda s, out, cfg: z3.Implies(z3.And(s.agent_known, s.registered, s.incident_known, s.device_match, s.refs_recorded,
                                              s.allowlisted, s.action == 0), out[0]),
}

ABLATIONS = {"no_provenance": PolicyFlags(provenance=False), "no_criticality": PolicyFlags(criticality=False),
             "no_ttl_cap": PolicyFlags(ttl_cap=False), "no_budget": PolicyFlags(budget=False),
             "no_floor": PolicyFlags(escalation_floor=False),
             "generic_tool_call_policy": PolicyFlags(provenance=False, criticality=False, escalation_floor=False, thresholds=False)}


def _show(m: z3.ModelRef, s: Sym) -> str:
    f = lambda v: m.eval(v, model_completion=True)
    parts = [f"action={ACTIONS[f(s.action).as_long()]}"]
    for name in ("agent_known", "registered", "incident_known", "device_match", "refs_recorded", "allowlisted", "high",
                 "approved", "resolved", "restricted"):
        if not z3.is_true(f(getattr(s, name))):
            parts.append(f"{name}=False")
        elif name in ("high", "approved", "resolved", "restricted"):
            parts.append(f"{name}=True")
    parts += [f"p={f(s.p)}", f"dur={f(s.dur)}", f"n_cited={f(s.n_cited)}", f"n_raw={f(s.n_raw)}", f"used={f(s.used)}",
              f"budget={f(s.budget)}", f"fleet_used={f(s.fleet_used)}", f"fleet_budget={f(s.fleet_budget)}"]
    return ", ".join(parts)


def prove(cfg: dict, flags: PolicyFlags = PolicyFlags(), timeout_ms: int = 60000) -> list[dict]:
    rows = []
    for name, thm in THEOREMS.items():
        s = Sym.fresh()
        out = policy_z3(s, cfg, flags)
        solver = z3.Solver(); solver.set("timeout", timeout_ms)
        solver.add(s.domain(), z3.Not(thm(s, out, cfg)))
        r = solver.check()
        rows.append({"theorem": name, "result": "proved" if r == z3.unsat else ("counterexample" if r == z3.sat else "unknown"),
                     "counterexample": _show(solver.model(), s) if r == z3.sat else ""})
    return rows


# ---------------------------------------------------------------------------
# Differential test: does the transcription match the real engine?
# ---------------------------------------------------------------------------
def differential_test(cfg: dict, registry, n: int = 50000, seed: int = 7, flags: PolicyFlags = PolicyFlags()) -> dict:
    rng = random.Random(seed)
    devices = list(registry) + ["ghost_dev", "cam_99", "plug_77"]   # unregistered ids, some of which pass the schema
    kinds = ["flow_anomaly", "dns_anomaly", "port_fanout", "peer_correlation"]
    sources = ["netflow", "dns-log", "edge-ids", "local-collector"]
    P = cfg["policy"]
    by_crit = P.get("device_budget_by_criticality") or {}
    mism, codes_seen = 0, set()
    for _ in range(n):
        inc_dev = rng.choice(list(registry))
        dev = inc_dev if rng.random() < 0.7 else rng.choice(devices)
        k = rng.randint(0, 4)
        ev = [f"e{i}" for i in range(k)]
        age = {e: rng.choice([0, 100, 1799, 1800, 1801, 5000]) for e in ev}
        rec = IncidentRecord("inc-1", inc_dev, round(rng.choice([rng.random(), 0.5, 0.8, 0.79999, 1.0, 0.0]), 5), frozenset(ev),
                             resolved=rng.random() < 0.2, evidence_kinds={e: rng.choice(kinds) for e in ev},
                             evidence_sources={e: rng.choice(sources) for e in ev}, evidence_time={e: -age[e] for e in ev})
        now = 0.0
        restricted = rng.random() < 0.4
        used = rng.choice([0, 300, 600, 899, 900, 1200, 2000])
        fleet_used = rng.choice([0, 600, 1500, 1800, 3000])
        fleet_budget = rng.choice([0, 0, 1800, 2400])
        corr_min = rng.choice([0, 0, 2, 3])
        indicator = rng.choice(["", "", "c2:x"])
        corr_count = rng.choice([0, 1, 2, 3, 5])
        corr_budget = rng.choice([0, 0, 600, 1200])
        exempt_used = rng.choice([0, 300, 900, 1200])
        hard_dev = rng.choice([0, 0, 1200, 1800])
        hard_fleet = rng.choice([0, 0, 2400, 3600])
        if rng.random() < 0.08:      # steer some cases into the corroborated-over-fleet-budget branch, otherwise rare
            fleet_budget, fleet_used, corr_min, indicator, corr_count = 1800, 3000, 2, "c2:x", rng.choice([2, 3, 5])
            corr_budget, hard_dev, hard_fleet = rng.choice([0, 1200]), 0, 0
        st = TrustedState(registry=registry, is_restricted=lambda d, r=restricted: r, now=lambda: now,
                          restricted_seconds=lambda d, w, u=used: float(u), fleet_restricted_seconds=lambda w, f=fleet_used: float(f),
                          corroborating_devices=lambda i, d, w, c=corr_count: c)
        rec.indicator = indicator
        if exempt_used:
            st.exempt_log.append((now, float(exempt_used)))
        st.record_incident(rec)
        act = rng.choice(ACTIONS)
        if rng.random() < 0.4:
            st.approve(rng.choice(["inc-1", "inc-2"]), rng.choice([dev, inc_dev]), rng.choice([act, "restrict_egress"]))
        refs = rng.sample(ev, rng.randint(0, k)) if ev else []
        if rng.random() < 0.15:
            refs = refs + ["forged"]
        if rng.random() < 0.1 and refs:
            refs = refs + [refs[0]]
        dur = rng.choice([-1, 0, 1, 299, 300, 301, 600, 900, 86400]) if act == "restrict_egress" else 0
        agent = rng.choice(["responder", "investigator", "single_agent", "mallory"])
        try:
            pa = ProposedAction(incident_id=rng.choice(["inc-1", "inc-1", "inc-1", "inc-404"]), device_id=dev, action_type=act,
                                duration_seconds=dur, evidence_refs=refs, reason="r", agent_id=agent)
        except Exception:
            continue
        c2 = {**cfg, "policy": {**cfg["policy"], "fleet_budget_seconds": fleet_budget, "fleet_corroboration_devices": corr_min,
                                       "corroboration_budget_seconds": corr_budget, "device_hard_cap_seconds": hard_dev,
                                       "fleet_hard_cap_seconds": hard_fleet}}
        eng = PolicyEngine(c2, st, flags)
        asset = registry.get(dev)
        key = (pa.incident_id, dev, act)
        sym_vals = dict(
            agent_known=agent in set(P["known_agents"]), registered=asset is not None, incident_known=pa.incident_id in st.incidents,
            device_match=dev == inc_dev, refs_recorded=set(refs) <= rec.evidence_ids,
            n_cited=rec.independent(refs, now, float(P.get("evidence_max_age_seconds", float("inf")))) if set(refs) <= rec.evidence_ids else 0,
            n_raw=len(set(refs)), n_recorded=rec.independent(rec.evidence_ids, now, float(P.get("evidence_max_age_seconds", float("inf")))),
            action=ACTIONS.index(act), allowlisted=asset is not None and act in asset.allowed_actions,
            high=asset is not None and asset.criticality == "high", approved=st.approval_valid(key, float(P["approval_max_age_seconds"])),
            resolved=rec.resolved, restricted=restricted, p=rec.p_attack, dur=dur, used=used,
            budget=float(by_crit.get(asset.criticality, P["device_budget_seconds"])) if asset else float(P["device_budget_seconds"]),
            fleet_used=fleet_used, fleet_budget=fleet_budget, corroborated=bool(corr_min > 0 and indicator and corr_count >= corr_min),
            exempt_used=exempt_used, corr_budget=corr_budget, hard_dev=hard_dev, hard_fleet=hard_fleet)
        real = eng.evaluate(pa)
        s = Sym.fresh("_c")
        out = policy_z3(s, cfg, flags)
        sub = []
        for name, val in sym_vals.items():
            var = getattr(s, name)
            if isinstance(val, bool):
                sub.append((var, z3.BoolVal(val)))
            elif name in ("p", "used", "budget", "fleet_used", "fleet_budget", "exempt_used", "corr_budget", "hard_dev", "hard_fleet"):
                sub.append((var, z3.RealVal(str(val))))
            else:
                sub.append((var, z3.IntVal(int(val))))
        a, r, c = (z3.simplify(z3.substitute(t, *sub)) for t in out)
        got = (z3.is_true(a), z3.is_true(r), CODES[c.as_long()])
        codes_seen.add(real.reason_code)
        if got != (real.allowed, real.requires_approval, real.reason_code):
            mism += 1
            if mism <= 3:
                print("MISMATCH", sym_vals, got, real.reason_code)
    return {"cases": n, "mismatches": mism, "reason_codes_exercised": len(codes_seen), "reason_codes_total": len(CODES),
            "reason_codes_not_exercised": "+".join(sorted(set(CODES) - codes_seen))}
