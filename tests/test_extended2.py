import pandas as pd

from secureagent_iot import extended2 as Y
from secureagent_iot.evaluation import run_response_benchmark


def test_design_patterns_and_exposure(cfg):
    summ, clean, atk = Y.design_pattern_comparison(cfg, [42])
    s = summ.set_index("system")
    assert len(s) == 8
    assert s.loc["B2_guarded", "violation_rate_attacked"] == 0 and s.loc["P7_filter_policy", "violation_rate_attacked"] == 0
    assert s.loc["P1_advisory_only", "review_rate_clean"] == 1.0 and s.loc["P1_advisory_only", "wrongful_isolation_s_per_100_attacked"] == 0
    assert s.loc["P2_unbounded_rules", "wrongful_isolation_s_per_100_clean"] > 50 * s.loc["B0_rules", "wrongful_isolation_s_per_100_clean"]
    assert s.loc["P4_agent_filter", "violation_rate_attacked"] < s.loc["B1_unguarded", "violation_rate_attacked"]
    assert s.loc["P4_agent_filter", "violation_rate_attacked"] >= s.loc["P4_agent_filter", "violation_rate_control"] - 0.02
    assert s.loc["P7_filter_policy", "review_rate_attacked"] < s.loc["B2_guarded", "review_rate_attacked"]
    expo, dist = Y.exposure_by_pattern(clean, atk, reps=15)
    e = expo.set_index("system")
    assert e.loc["P1_advisory_only", "handled_autonomously"] == 0
    assert (dist.groupby("system").share_responded.apply(lambda x: x.is_monotonic_increasing)).all()


def test_campaign_risk_math():
    d = Y.campaign_risk({"a": 0.02}, ns=(1, 35))
    assert abs(d.p_at_least_one.iloc[0] - 0.02) < 1e-9 and d.p_at_least_one.iloc[1] > 0.5


def test_audit_tamper_detection_matches_the_stated_guarantee():
    t = Y.audit_tamper(n_records=120, trials=25).set_index("tampering")
    for k in ("edit one record", "delete one record", "swap two records"):
        assert t.loc[k, "detected_chain_only"] == 1.0
    for k in ("truncate the tail", "edit and rebuild all hashes"):
        assert t.loc[k, "detected_chain_only"] == 0.0 and t.loc[k, "detected_with_anchored_head"] == 1.0


def test_trust_stress_never_reports_a_spec_violation_for_guarded_system(cfg):
    t = Y.trust_assumption_stress(cfg, [42])
    b2 = t[t.system == "B2_guarded"]
    assert (b2.spec_violation_rate == 0).all()
    crit = b2[b2.stress.str.startswith("critical")].sort_values("level")
    assert crit.harm_rate.iloc[0] == 0 and crit.harm_rate.iloc[-1] > 0


def test_seed_convergence_and_breakdowns(cfg):
    s = Y.seed_convergence(cfg, 4)
    assert (s[(s.system == "B2_guarded") & (s.metric == "unauthorized_action")].running_mean == 0).all()
    runs = run_response_benchmark(cfg, [42])
    assert set(Y.family_breakdown(runs).family) == set(runs.family)
