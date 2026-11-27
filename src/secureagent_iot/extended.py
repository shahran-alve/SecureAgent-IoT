"""Extended evaluations. All TESTBED-SYNTHETIC with scripted agents (no LLM, no real traffic).

E1 scenario-population replication     E5 mutation testing of the policy engine
E2 agent-profile robustness            E6 review-queue simulation under flooding
E3 detector-quality sweep              E7 scalability of the enforcement path
E4 policy-threshold sensitivity
"""
from __future__ import annotations

import ast
import copy
import heapq
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

from .agents import AgenticSystem, B0Rules, build_systems
from .audit import AuditLog
from .config import rng_for
from .evaluation import (EVIDENCE_CLASS, RESPONSE_METRICS, cluster_bootstrap_ci, puppet_proposals, run_attack_benchmark,
                         run_response_benchmark, summarize_response)
from .policy import IncidentRecord, PolicyEngine, TrustedState
from .schemas import Asset, ProposalRejected, ProposedAction, parse_proposal
from .simulator import Testbed, build_registry, build_templates, decision_violations, generate_population, realize, spec_violations

CORE = ["B0_rules", "B1_unguarded", "B1v_schema_only", "B2_guarded", "B3_single_guarded"]
KEY_METRICS = ["safe_containment", "containment_any", "false_isolation", "unauthorized_action", "service_interruption",
               "missed_malicious", "review_rate", "utility_mean"]


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ------------------------------------------------------------------ E1
def population_replication(cfg: dict, n: int = 2000, seeds: tuple[int, ...] = (11, 23, 42, 77, 101), pop_seed: int = 2026):
    templates = generate_population(n, pop_seed)
    df = run_response_benchmark(cfg, list(seeds), systems=CORE, templates=templates)
    summ = summarize_response(df, cfg, seed=1)
    summ.insert(1, "population", f"generated n={n}")
    mix = pd.Series([t.family for t in templates]).value_counts().rename_axis("family").reset_index(name="n_templates")
    return df, summ, mix


# ------------------------------------------------------------------ E2
def agent_profiles(cfg: dict, seeds: list[int]):
    registry = build_registry()
    rows_c, rows_a = [], []
    for prof in ("greedy", "cautious", "timid", "erratic"):
        sysmap = {"B1_unguarded": AgenticSystem("B1_unguarded", cfg, guarded=False, profile=prof, registry=registry),
                  "B2_guarded": AgenticSystem("B2_guarded", cfg, guarded=True, profile=prof, registry=registry)}
        c = run_response_benchmark(cfg, seeds, sysmap=sysmap)
        s = summarize_response(c, cfg, seed=2)
        s.insert(1, "profile", prof)
        rows_c.append(s)
        a = run_attack_benchmark(cfg, seeds, controls=False, sysmap=sysmap)
        v = a[~a["suppression_scored"]]
        for system, g in v.groupby("system", sort=False):
            sup = a[(a["system"] == system) & a["suppression_scored"]]
            lo, hi = cluster_bootstrap_ci(g.assign(_c=g["attack_id"] + "|" + g["scenario_id"]).groupby("_c")["exec_success"].mean().values,
                                          int(cfg["bootstrap"]["n_resamples"]), rng_for("prof", prof, system))
            rows_a.append({"evidence_class": EVIDENCE_CLASS, "profile": prof, "system": system, "attempts": len(g),
                           "proposal_asr": g["proposal_success"].mean(), "exec_successes": int(g["exec_success"].sum()),
                           "exec_asr": g["exec_success"].mean(), "exec_asr_ci95_lo": lo, "exec_asr_ci95_hi": hi,
                           "any_violation": int(g["any_violation"].sum()), "suppression_rate": sup["suppression_success"].mean()})
    return pd.concat(rows_c, ignore_index=True), pd.DataFrame(rows_a)


# ------------------------------------------------------------------ E3
def detector_quality_sweep(cfg: dict, qualities=(0.4, 0.55, 0.7, 0.85, 1.0), n: int = 600, seeds=(11, 23, 42)):
    templates = generate_population(n, 7)
    out = []
    for q in qualities:
        c = copy.deepcopy(cfg); c["detector_quality"] = float(q)
        df = run_response_benchmark(c, list(seeds), systems=["B0_rules", "B1_unguarded", "B2_guarded"], templates=templates)
        one = df[df["system"] == "B2_guarded"]
        auc = _auc(one.loc[one["truth_malicious"], "p_attack"].to_numpy(), one.loc[~one["truth_malicious"], "p_attack"].to_numpy())
        s = summarize_response(df, c, seed=3)
        s.insert(1, "quality", q); s.insert(2, "detector_auc", auc)
        a = run_attack_benchmark(c, list(seeds), systems=["B1_unguarded", "B2_guarded"], controls=False)
        for system, g in a.groupby("system", sort=False):
            e = g[~g["suppression_scored"]]; sup = g[g["suppression_scored"]]
            for metric, val in (("attack_exec_asr", e["exec_success"].mean()), ("attack_goal_executed", e["goal_executed"].mean()),
                                ("attack_benign_isolation", e["benign_isolated"].mean()), ("attack_suppression", sup["suppression_success"].mean()),
                                ("attack_review_rate", g["escalated"].mean())):
                s = pd.concat([s, pd.DataFrame([{"evidence_class": EVIDENCE_CLASS, "quality": q, "detector_auc": auc, "system": system,
                                                 "metric": metric, "mean": val, "n_rows": len(g)}])], ignore_index=True)
        out.append(s)
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ E4
def threshold_sensitivity(cfg: dict, p_thresholds=(0.6, 0.7, 0.8, 0.9, 0.95), min_evidences=(1, 2, 3), n: int = 600, seeds=(11, 23, 42)):
    templates = generate_population(n, 7)
    rows = []
    for pt in p_thresholds:
        for me in min_evidences:
            c = copy.deepcopy(cfg)
            c["policy"]["p_attack_threshold"] = c["spec"]["p_attack_threshold"] = float(pt)
            c["policy"]["min_evidence"] = c["spec"]["min_evidence"] = int(me)
            c["policy"]["review_floor"] = min(float(c["policy"]["review_floor"]), float(pt))
            df = run_response_benchmark(c, list(seeds), systems=["B2_guarded"], templates=templates)
            mal, ben = df[df["truth_malicious"]], df[~df["truth_malicious"]]
            a = run_attack_benchmark(c, list(seeds), systems=["B2_guarded"], controls=False)
            e, sup = a[~a["suppression_scored"]], a[a["suppression_scored"]]
            rows.append({"evidence_class": EVIDENCE_CLASS, "p_threshold": pt, "min_evidence": me,
                         "autonomous_containment_of_malicious": mal["contained"].mean(), "false_isolation": ben["false_isolation"].mean(),
                         "review_rate": df["escalated"].mean(), "missed_malicious": mal["missed"].mean(),
                         "unauthorized_action": df["spec_violation"].mean(), "attack_exec_asr": e["exec_success"].mean(),
                         "attack_benign_isolation": e["benign_isolated"].mean(),
                         "attack_suppression": sup["suppression_success"].mean() if len(sup) else float("nan"),
                         "n_clean": len(df), "n_attack": len(a)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E5
class _Mutator(ast.NodeTransformer):
    """Applies exactly one mutation, the `target`-th candidate site, inside PolicyEngine.evaluate."""
    SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
            ast.In: ast.NotIn, ast.NotIn: ast.In}

    def __init__(self, target: int | None):
        self.target, self.count, self.sites, self.inside = target, 0, [], False

    def _hit(self, kind: str, node) -> bool:
        idx = self.count; self.count += 1
        self.sites.append((idx, kind, getattr(node, "lineno", 0)))
        return idx == self.target

    def visit_FunctionDef(self, node):
        if node.name == "evaluate":
            self.inside = True; self.generic_visit(node); self.inside = False
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if self.inside and type(node.ops[0]) in self.SWAP and self._hit("relational", node):
            node.ops[0] = self.SWAP[type(node.ops[0])]()
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if self.inside and self._hit("logical", node):
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if self.inside and self._hit("negate-condition", node):
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if (self.inside and isinstance(node.func, ast.Attribute) and node.func.attr == "_d" and len(node.args) >= 2
                and all(isinstance(a, ast.Constant) and isinstance(a.value, bool) for a in node.args[:2])):
            if self._hit("flip-allowed", node):
                node.args[0] = ast.Constant(value=not node.args[0].value)
            if self._hit("flip-approval", node):
                node.args[1] = ast.Constant(value=not node.args[1].value)
        return node


def _policy_source() -> str:
    from . import policy
    return Path(policy.__file__).read_text()


def _load_mutant(tree: ast.AST):
    mod = types.ModuleType("secureagent_iot._policy_mutant")
    mod.__package__ = "secureagent_iot"
    sys.modules[mod.__name__] = mod
    exec(compile(ast.fix_missing_locations(tree), "<mutant>", "exec"), mod.__dict__)
    return mod.PolicyEngine


def _decision_grid(cfg: dict, seed: int):
    """Every schema-valid proposal in the search grid for one seed, with its testbed."""
    registry, spec = build_registry(), cfg["spec"]
    grid = []
    for t in build_templates():
        sc = realize(t, seed)
        valid = []
        for raw in puppet_proposals(sc, registry):
            try:
                valid.append(parse_proposal(raw, registry))
            except ProposalRejected:
                pass
        for pre in (False, True):
            tb = Testbed(cfg, registry, "mut", seed, t.scenario_id, pre)
            tb.register(sc)
            other_dev = "plug_02" if sc.device_id != "plug_02" else "plug_01"
            tb.state.record_incident(IncidentRecord("inc-other", other_dev, 0.95, frozenset({"ev-other-0", "ev-other-1"})))
            if pre:
                prior = IncidentRecord(f"inc-prior-{t.scenario_id}", sc.device_id, 0.95, frozenset({"ev-p0", "ev-p1"}))
                tb.state.record_incident(prior)
                tb.controller.restrict_egress(sc.device_id, int(spec["max_ttl_seconds"]), prior.incident_id)
            acts = [a for a in valid if (not pre or a.action_type == "release_restriction")]
            viol = []
            for a in acts:
                ex = {"action_type": a.action_type, "device_id": a.device_id, "duration_seconds": a.duration_seconds}
                rel = pre and a.action_type == "release_restriction" and a.device_id == sc.device_id
                viol.append((bool(spec_violations(ex, sc, registry, spec, tb.state.all_approvals(), released_active_unresolved=rel)),
                             bool(decision_violations(a.model_dump(), sc, registry, spec, tb.state.all_approvals(),
                                                      device_restricted=pre and a.device_id == sc.device_id, released_active_unresolved=rel))))
            grid.append((tb, acts, viol, sc))
    return grid


def _run_engine(engine_cls, cfg, grid):
    """Return (decision vector, authorized-violating count, under-response count) or None on crash."""
    dec, unsafe, under, unsafe2 = [], 0, 0, 0
    thr, me = float(cfg["spec"]["p_attack_threshold"]), int(cfg["spec"]["min_evidence"])
    try:
        for tb, acts, viol, sc in grid:
            eng = engine_cls(cfg, tb.state)
            strong = sc.p_attack >= thr and len(sc.observations) >= me
            for a, v in zip(acts, viol):
                d = eng.evaluate(a)
                dec.append((bool(d.allowed), bool(d.requires_approval)))
                if d.allowed and v[0]:
                    unsafe += 1
                if d.allowed and v[1]:
                    unsafe2 += 1
                if d.allowed and a.action_type == "notify" and strong:
                    under += 1     # liveness property: a strong alert must not be closed with a bare notification
    except Exception:
        return None
    return dec, unsafe, under, unsafe2


def mutation_testing(cfg: dict, seed: int = 42, run_unit_tests: bool = True, tests_dir: str | Path | None = None):
    """Single-point mutants of PolicyEngine.evaluate, judged by three detectors:
    (1) the specification oracle over the proposal grid (does the mutant authorize a violating action?),
    (2) a liveness property (does it accept a bare notification on a strong alert?),
    (3) the policy unit tests. A differential comparison with the reference shows which mutants change behaviour at all."""
    import secureagent_iot.policy as policy_mod
    src = _policy_source()
    probe = _Mutator(None); probe.visit(ast.parse(src))
    grid = _decision_grid(cfg, seed)
    ref_dec, ref_unsafe, ref_under, ref_unsafe2 = _run_engine(PolicyEngine, cfg, grid)
    assert ref_unsafe == 0 and ref_under == 0 and ref_unsafe2 == 0
    original = policy_mod.PolicyEngine.evaluate
    rows = []
    for idx, kind, lineno in probe.sites:
        m = _Mutator(idx); tree = m.visit(ast.parse(src))
        row = {"evidence_class": EVIDENCE_CLASS, "mutant": idx, "operator": kind, "line": lineno, "crash": False,
               "decisions_changed": 0, "more_permissive": 0, "more_restrictive": 0, "oracle_violations": 0,
               "liveness_violations": 0, "extended_oracle_violations": 0, "unit_tests_fail": None, "unit_tests_extended_fail": None}
        try:
            cls = _load_mutant(tree)
            res = _run_engine(cls, cfg, grid)
        except Exception:
            cls, res = None, None
        if res is None:
            row["crash"] = True
        else:
            dec, unsafe, under, unsafe2 = res
            row["extended_oracle_violations"] = unsafe2
            row["decisions_changed"] = sum(a != b for a, b in zip(dec, ref_dec))
            row["more_permissive"] = sum(a[0] and not b[0] for a, b in zip(dec, ref_dec))
            row["more_restrictive"] = sum(b[0] and not a[0] for a, b in zip(dec, ref_dec))
            row["oracle_violations"], row["liveness_violations"] = unsafe, under
        if run_unit_tests and cls is not None:
            import contextlib, io
            import pytest
            tdir = Path(tests_dir or Path(__file__).resolve().parents[2] / "tests")
            policy_mod.PolicyEngine.evaluate = cls.evaluate
            try:
                for col, files in (("unit_tests_fail", ["test_policy.py"]),
                                   ("unit_tests_extended_fail", ["test_policy.py", "test_policy_decision_table.py", "test_policy_properties.py"])):
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        code = pytest.main(["-q", "-x", "--tb=no", "-p", "no:cacheprovider"] + [str(tdir / f) for f in files])
                    row[col] = bool(code != 0)
            finally:
                policy_mod.PolicyEngine.evaluate = original
        rows.append(row)
    df = pd.DataFrame(rows)
    df["behaviour_changed"] = df["crash"] | (df["decisions_changed"] > 0)
    df["killed_by_oracle"] = df["oracle_violations"] > 0
    df["killed_by_liveness"] = df["liveness_violations"] > 0
    df["killed_by_extended_oracle"] = df["extended_oracle_violations"] > 0
    df["killed_by_tests"] = df["unit_tests_fail"].fillna(False).astype(bool) | df["crash"]
    df["killed_any"] = df["killed_by_oracle"] | df["killed_by_liveness"] | df["killed_by_tests"]
    ext = df["unit_tests_extended_fail"] if "unit_tests_extended_fail" in df else pd.Series(False, index=df.index)
    df["killed_by_tests_extended"] = ext.fillna(False).astype(bool) | df["crash"]
    df["killed_any_extended"] = df["killed_by_oracle"] | df["killed_by_liveness"] | df["killed_by_tests_extended"]
    df["grid_size"] = len(ref_dec)
    return df


def summarize_mutation(df: pd.DataFrame) -> pd.DataFrame:
    def agg(g, label):
        perm = g[g["more_permissive"] > 0]
        return {"evidence_class": EVIDENCE_CLASS, "operator": label, "mutants": len(g), "behaviour_changed": int(g["behaviour_changed"].sum()),
                "equivalent_on_grid": int((~g["behaviour_changed"]).sum()), "permissive": len(perm),
                "permissive_caught_by_oracle_or_liveness": int((perm["killed_by_oracle"] | perm["killed_by_liveness"]).sum()),
                "permissive_caught_by_extended_oracle": int((perm["killed_by_extended_oracle"] | perm["killed_by_liveness"]).sum()),
                "killed_by_extended_oracle": int((g["killed_by_extended_oracle"] | g["killed_by_liveness"]).sum()),
                "killed_by_oracle": int(g["killed_by_oracle"].sum()), "killed_by_liveness": int(g["killed_by_liveness"].sum()),
                "killed_by_unit_tests": int(g["killed_by_tests"].sum()), "killed_any": int(g["killed_any"].sum()),
                "survived_non_equivalent": int((g["behaviour_changed"] & ~g["killed_any"]).sum()),
                "mutation_score": g.loc[g["behaviour_changed"], "killed_any"].mean(),
                "killed_by_unit_tests_extended": int(g["killed_by_tests_extended"].sum()),
                "killed_any_extended": int(g["killed_any_extended"].sum()),
                "survived_non_equivalent_extended": int((g["behaviour_changed"] & ~g["killed_any_extended"]).sum()),
                "mutation_score_extended": g.loc[g["behaviour_changed"], "killed_any_extended"].mean()}
    rows = [agg(g, k) for k, g in df.groupby("operator", sort=False)] + [agg(df, "all")]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E6
def review_queue_simulation(clean_runs: pd.DataFrame, attack_runs: pd.DataFrame, reps: int = 300, hours: float = 8.0,
                            rate_per_hour: float = 30.0, analysts: int = 2, service_min: float = 6.0,
                            attack_fractions=(0.0, 0.1, 0.3, 0.5), fleet: int = 200, attacker_devices: int = 5,
                            sla_min: float = 15.0, service_dist: str = "exponential") -> pd.DataFrame:
    """Discrete-event simulation of the human review queue fed by B2 outcomes.
    Incidents arrive as a Poisson process. A fraction is attacker-injected and
    comes from a few compromised devices. Queue disciplines: FIFO, priority by
    detector probability, and priority plus per-device coalescing (a new
    escalation for a device that already has an open ticket is merged into it)."""
    c = clean_runs[clean_runs["system"] == "B2_guarded"][["escalated", "truth_malicious", "p_attack"]].to_numpy()
    a = attack_runs[(attack_runs["system"] == "B2_guarded") & attack_runs["injected"]][["escalated", "truth_malicious", "p_attack"]].to_numpy()
    rows = []
    for f in attack_fractions:
        for discipline in ("fifo", "priority", "priority+coalesce"):
            waits, sla_miss, backlog, tickets, unreviewed = [], [], [], [], []
            for r in range(reps):
                rng = rng_for("rq", f, r)          # same arrivals for every discipline
                n = rng.poisson(rate_per_hour * hours)
                t_arr = np.sort(rng.random(n) * hours * 60.0)
                is_att = rng.random(n) < f
                svc_rng = rng_for("rq-svc", f, r)
                events = []
                for i in range(n):
                    pool = a if is_att[i] else c
                    esc, mal, p = pool[int(rng.integers(len(pool)))]
                    dev = int(rng.integers(attacker_devices)) if is_att[i] else int(attacker_devices + rng.integers(fleet - attacker_devices))
                    if esc:
                        events.append((t_arr[i], bool(mal), float(p), dev))
                servers = [0.0] * analysts
                heapq.heapify(servers)
                queue, open_ticket, k, seq, n_tickets = [], {}, 0, 0, 0
                w_mal, pending_mal, left = [], 0, 0
                horizon = hours * 60.0

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
                    if now > horizon:        # shift over: whatever is still waiting was never reviewed
                        pending_mal = sum(m for it in queue for _, m in it[3]) + sum(e[1] for e in events[k:])
                        left = len(queue) + len(events) - k
                        break
                    item = heapq.heappop(queue)
                    open_ticket.pop(item[4], None)
                    w_mal += [now - t for t, m in item[3] if m]
                    if service_dist == "lognormal":   # heavy-tailed, same mean: sigma = 1
                        svc = svc_rng.lognormal(np.log(service_min) - 0.5, 1.0)
                    else:
                        svc = svc_rng.exponential(service_min)
                    heapq.heappush(servers, now + svc)
                backlog.append(left)
                waits += w_mal
                tot_mal = len(w_mal) + pending_mal
                sla_miss.append((sum(w > sla_min for w in w_mal) + pending_mal) / tot_mal if tot_mal else 0.0)
                unreviewed.append(pending_mal / tot_mal if tot_mal else 0.0)
                tickets.append(n_tickets)
            w = np.array(waits) if waits else np.array([0.0])
            rows.append({"evidence_class": EVIDENCE_CLASS, "attack_fraction": f, "discipline": discipline, "reps": reps,
                         "tickets_per_shift": float(np.mean(tickets)), "wait_median_min": float(np.median(w)),
                         "wait_p95_min": float(np.percentile(w, 95)), "malicious_sla_miss": float(np.mean(sla_miss)),
                         "malicious_unreviewed_at_end": float(np.mean(unreviewed)), "backlog_at_end": float(np.mean(backlog))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E7
def scalability(cfg: dict, registry_sizes=(100, 1_000, 10_000, 100_000), n_decisions: int = 20_000, audit_sizes=(1_000, 10_000, 100_000)):
    rows = []
    letters = "abcdefghijklmnopqrstuvwxyz"
    for n in registry_sizes:
        reg = {}
        i = 0
        while len(reg) < n:
            pre = "".join(letters[(i // (100 * 26 ** k)) % 26] for k in range(4))
            d = f"{pre}_{i % 100:02d}"
            reg[d] = Asset(device_id=d, criticality=("low", "medium", "high")[i % 3],
                           allowed_actions=["notify", "request_review", "restrict_egress", "release_restriction"], owner="o", is_online=True)
            i += 1
        devs = list(reg)
        state = TrustedState(registry=reg)
        rng = np.random.default_rng(0)
        raws = []
        for j in range(n_decisions):
            dev = devs[int(rng.integers(n))]
            ev = [f"ev-{j}-{k}" for k in range(4)]
            state.record_incident(IncidentRecord(f"inc-{j}", dev, float(rng.random()), frozenset(ev)))
            raws.append({"incident_id": f"inc-{j}", "device_id": dev, "action_type": "restrict_egress", "duration_seconds": 300,
                         "evidence_refs": ev[:3], "reason": "r", "agent_id": "responder"})
        eng = PolicyEngine(cfg, state)
        t0 = time.perf_counter()
        for raw in raws:
            eng.evaluate(parse_proposal(raw, reg))
        dt = time.perf_counter() - t0
        rows.append({"evidence_class": "measured (orchestration only)", "component": "validate+authorize", "size": n, "unit": "registered devices",
                     "operations": n_decisions, "seconds": dt, "ops_per_second": n_decisions / dt, "microseconds_per_op": 1e6 * dt / n_decisions})
    for n in audit_sizes:
        log = AuditLog(now=lambda: "t")
        t0 = time.perf_counter()
        for j in range(n):
            log.append(incident_id=f"inc-{j}", evidence_ids=["e1", "e2"], action={"action_type": "notify"}, decision={"allowed": True},
                       policy_version="1.0.0")
        dt = time.perf_counter() - t0
        t1 = time.perf_counter(); ok, _ = log.verify(); dv = time.perf_counter() - t1
        assert ok
        rows.append({"evidence_class": "measured (orchestration only)", "component": "audit append", "size": n, "unit": "records",
                     "operations": n, "seconds": dt, "ops_per_second": n / dt, "microseconds_per_op": 1e6 * dt / n})
        rows.append({"evidence_class": "measured (orchestration only)", "component": "audit verify", "size": n, "unit": "records",
                     "operations": n, "seconds": dv, "ops_per_second": n / dv, "microseconds_per_op": 1e6 * dv / n})
    return pd.DataFrame(rows)
