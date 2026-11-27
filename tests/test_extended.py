import ast

import pandas as pd

from secureagent_iot import extended as X
from secureagent_iot.agents import AgenticSystem
from secureagent_iot.evaluation import run_attack_benchmark, run_response_benchmark
from secureagent_iot.policy import PolicyEngine
from secureagent_iot.simulator import generate_population, realize


def test_population_generator_is_deterministic_and_varied(registry):
    a, b = generate_population(300, 1), generate_population(300, 1)
    assert a == b and a != generate_population(300, 2)
    assert len({t.scenario_id for t in a}) == 300 and len({t.family for t in a}) == 6
    assert all(t.device_id in registry for t in a)


def test_detector_quality_degrades_separation(templates):
    def gap(q):
        mal = [realize(t, s, quality=q).p_attack for t in templates if t.truth_malicious for s in (1, 2, 3)]
        ben = [realize(t, s, quality=q).p_attack for t in templates if not t.truth_malicious for s in (1, 2, 3)]
        return sum(mal) / len(mal) - sum(ben) / len(ben)
    assert gap(1.0) > gap(0.7) > gap(0.4) > 0
    assert realize(templates[0], 1, quality=1.0).p_attack == realize(templates[0], 1).p_attack


def test_guard_holds_for_every_agent_profile(cfg, registry):
    for prof in ("greedy", "cautious", "timid", "erratic"):
        sysmap = {"B1_unguarded": AgenticSystem("B1_unguarded", cfg, guarded=False, profile=prof, registry=registry),
                  "B2_guarded": AgenticSystem("B2_guarded", cfg, guarded=True, profile=prof, registry=registry)}
        clean = run_response_benchmark(cfg, [42], sysmap=sysmap)
        assert not clean[clean.system == "B2_guarded"].spec_violation.any(), prof
        atk = run_attack_benchmark(cfg, [42], controls=False, sysmap=sysmap)
        assert not atk[atk.system == "B2_guarded"].any_violation.any(), prof
        assert atk[atk.system == "B1_unguarded"].exec_success.mean() > 0.4, prof


def test_schema_only_baseline_does_not_stop_wellformed_attacks(cfg):
    atk = run_attack_benchmark(cfg, [42], systems=["B1v_schema_only"], controls=False)
    assert atk[~atk.suppression_scored].exec_success.mean() > 0.5
    assert not atk.executed_device.isin(["cam_99", "10.0.0.5"]).any()


def test_mutation_machinery_flags_a_removed_ttl_check(cfg):
    src = X._policy_source()
    probe = X._Mutator(None); probe.visit(ast.parse(src))
    assert len(probe.sites) >= 60
    grid = X._decision_grid(cfg, 42)[:20]
    ref = X._run_engine(PolicyEngine, cfg, grid)
    assert ref[1] == 0 and ref[2] == 0 and ref[3] == 0
    unsafe_found = 0
    for idx, kind, _ in probe.sites:
        if kind != "flip-allowed":
            continue
        cls = X._load_mutant(X._Mutator(idx).visit(ast.parse(src)))
        res = X._run_engine(cls, cfg, grid)
        unsafe_found += bool(res and res[1] > 0)
    assert unsafe_found >= 5      # several deny->allow flips are caught by the oracle alone


def test_review_queue_simulation_orders_disciplines(cfg):
    clean = run_response_benchmark(cfg, [42], systems=["B2_guarded"])
    atk = run_attack_benchmark(cfg, [42], systems=["B2_guarded"], controls=False)
    q = X.review_queue_simulation(clean, atk, reps=40, attack_fractions=(0.0, 0.5)).set_index(["attack_fraction", "discipline"])
    assert q.loc[(0.5, "fifo"), "malicious_sla_miss"] > q.loc[(0.0, "fifo"), "malicious_sla_miss"]
    assert q.loc[(0.5, "priority+coalesce"), "malicious_sla_miss"] < q.loc[(0.5, "fifo"), "malicious_sla_miss"]
    assert q.loc[(0.5, "priority+coalesce"), "tickets_per_shift"] <= q.loc[(0.5, "fifo"), "tickets_per_shift"]


def test_scalability_runs(cfg):
    s = X.scalability(cfg, registry_sizes=(100,), n_decisions=500, audit_sizes=(200,))
    assert (s.ops_per_second > 0).all() and set(s.component) == {"validate+authorize", "audit append", "audit verify"}
