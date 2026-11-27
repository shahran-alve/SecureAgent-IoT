"""Tests for hard ceilings, the bounded corroboration exemption, the generic tool-call baseline, the atomic
ledger-backed authorizer, and experiments E40-E47."""
import copy
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from secureagent_iot import extended5 as R
from secureagent_iot import formal as F
from secureagent_iot.agents import GENERIC_FLAGS
from secureagent_iot.config import load_config
from secureagent_iot.enforcement import AtomicAuthorizer, RestrictionLedger, _incident, failure_injection
from secureagent_iot.policy import PolicyEngine, PolicyFlags, TrustedState
from secureagent_iot.simulator import build_registry


def _cfg(**policy):
    c = copy.deepcopy(load_config(None))
    for k, v in policy.items():
        c["policy"][k] = v
        c["spec"][k] = v
    return c


def _engine(cfg, history, flags=None, now=10_000.0):
    st = TrustedState(registry=build_registry(), now=lambda: now,
                      restricted_seconds=lambda d, w: sum(x for t, dd, x in history if dd == d and t >= now - w),
                      fleet_restricted_seconds=lambda w: sum(x for t, _, x in history if t >= now - w))
    return st, PolicyEngine(cfg, st, flags or PolicyFlags())


def test_hard_ceiling_blocks_even_approved_restriction():
    cfg = _cfg(device_hard_cap_seconds=600)
    hist = [(9_900.0, "sensor_01", 300.0), (9_950.0, "sensor_01", 300.0)]
    st, eng = _engine(cfg, hist)
    rec, act = _incident(1, "sensor_01", 10_000.0)
    st.record_incident(rec)
    st.approve(rec.incident_id, "sensor_01", "restrict_egress")
    assert eng.evaluate(act).reason_code == "DENY_HARD_CEILING"


def test_hard_ceiling_off_by_default():
    cfg = _cfg()
    assert not cfg["policy"].get("device_hard_cap_seconds") and not cfg["policy"].get("fleet_hard_cap_seconds")


def test_generic_flags_keep_identity_and_rate_limits():
    assert not GENERIC_FLAGS.provenance and not GENERIC_FLAGS.criticality and not GENERIC_FLAGS.thresholds
    assert GENERIC_FLAGS.budget


def test_generic_policy_breaks_safety_but_b2_does_not():
    cfg = load_config(None)
    s1 = lambda rows: [r for r in rows if r["theorem"].startswith("S1:")][0]["result"]
    assert s1(F.prove(cfg)) == "proved"
    assert s1(F.prove(cfg, F.ABLATIONS["generic_tool_call_policy"])) == "counterexample"


def test_new_theorems_proved_with_ceilings_and_exemption_budget():
    cfg = _cfg(device_hard_cap_seconds=1800, fleet_hard_cap_seconds=2700, corroboration_budget_seconds=900,
               fleet_corroboration_devices=2)
    rows = F.prove(cfg)
    assert all(r["result"] == "proved" for r in rows), [r for r in rows if r["result"] != "proved"]
    assert any(r["theorem"].startswith("S7:") for r in rows)


def test_atomic_authorizer_holds_budget_under_concurrency(tmp_path):
    cfg = load_config(None)
    ledger = RestrictionLedger(tmp_path / "ledger.jsonl")
    clock = lambda: 1_000.0
    auth = AtomicAuthorizer(cfg, build_registry(), ledger, clock)
    out = []
    ths = [threading.Thread(target=lambda i=i: out.append(auth.submit(*_incident(i, "sensor_01", 1_000.0)))) for i in range(8)]
    [t.start() for t in ths]; [t.join() for t in ths]
    assert ledger.device_seconds("sensor_01", 1_000.0, 3600) <= cfg["policy"]["device_budget_seconds"]


def test_ledger_tamper_fails_closed(tmp_path):
    cfg = load_config(None)
    p = tmp_path / "ledger.jsonl"
    led = RestrictionLedger(p)
    led.append(1.0, "sensor_01", 300.0); led.append(2.0, "sensor_01", 300.0)
    lines = p.read_text().splitlines()
    p.write_text(lines[1] + "\n")                              # delete the first entry
    auth = AtomicAuthorizer(cfg, build_registry(), RestrictionLedger(p), lambda: 3.0)
    assert auth.submit(*_incident(9, "sensor_02", 3.0)) is False


def test_failure_injection_atomic_never_over_budget():
    d = failure_injection(load_config(None), trials=3)
    assert (d[d.design == "atomic"].trials_over_budget == 0).all()


def test_bounded_exemption_and_ceiling_reduce_worst_case():
    d = R.corroboration_mitigations(load_config(None), [11])
    w = d.groupby("defense").proved_worst_case_device_hours.max()
    assert w["bounded exemption, B_c = 900 s/h"] < w["unbounded exemption (k = 2)"]
    assert w["unbounded exemption + hard fleet ceiling 2700 s/h"] < w["unbounded exemption (k = 2)"]


def test_queue_hard_ceiling_bounds_approved_time():
    d = R.approvals_queue(load_config(None), [11])
    h = d[d.setting.str.contains("hard ceilings")]
    assert (h.total_device_hours <= h.hard_ceiling_bound_device_hours + 1e-9).all()


def test_scenario_selection_partitions_templates():
    raw = Path(__file__).resolve().parents[1] / "results" / "raw"
    if not (raw / "attack_runs.csv").exists():
        pytest.skip("raw results not present")
    s, per_t, _, _ = R.scenario_selection(pd.read_csv(raw / "attack_runs.csv"), pd.read_csv(raw / "response_runs.csv"))
    s = s.set_index("subset")
    assert s.loc["all 68 templates", "exec_attempts"] == s.loc["original 64 templates", "exec_attempts"] + \
        s.loc["4 look-alike templates added after the pilot", "exec_attempts"]
    assert per_t.permitted_harmful.sum() == s.loc["all 68 templates", "permitted_harmful"]


def test_heldout_styles_do_not_change_run1_plan(monkeypatch):
    from secureagent_iot import llm_harness as H
    from secureagent_iot.attacks import HELDOUT_STYLES, STYLES, apply_style
    monkeypatch.delenv("LLM_STYLE_SET", raising=False)
    assert H.attack_arms() == H.ATTACK_ARMS and H.styles_sha256() == H.styles_sha256(STYLES)
    monkeypatch.setenv("LLM_STYLE_SET", "heldout")
    assert [s for s, inj in H.attack_arms() if inj] == list(HELDOUT_STYLES)
    for st in HELDOUT_STYLES:
        assert "isolate cam_03" in apply_style("isolate cam_03 now", st, "untrusted_text").replace("cam_03 now", "cam_03 now") or st == "split"
    assert not set(HELDOUT_STYLES) & set(STYLES)


def test_sampling_off_by_default(monkeypatch):
    from secureagent_iot.llm_adapter import HFLocalModel
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    assert HFLocalModel.sampling() is None
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7"); monkeypatch.setenv("LLM_SAMPLE_SEED", "3")
    assert HFLocalModel.sampling() == {"temperature": 0.7, "top_p": 0.95, "sample_seed": 3}


def test_torn_ledger_record_fails_closed(tmp_path):
    cfg = load_config(None)
    p = tmp_path / "ledger.jsonl"
    led = RestrictionLedger(p)
    led.append(1.0, "sensor_01", 300.0)
    with open(p, "a") as f:
        f.write('{"t": 2.0, "device_id": "sens')                 # crash mid-append
    auth = AtomicAuthorizer(cfg, build_registry(), RestrictionLedger(p), lambda: 3.0)
    assert auth.ledger.ok is False and auth.submit(*_incident(5, "sensor_02", 3.0)) is False


def test_second_process_sees_committed_grants(tmp_path):
    cfg = load_config(None)
    p = tmp_path / "ledger.jsonl"
    a1 = AtomicAuthorizer(cfg, build_registry(), RestrictionLedger(p), lambda: 0.0)
    a2 = AtomicAuthorizer(cfg, build_registry(), RestrictionLedger(p), lambda: 0.0)   # opened before any grant
    for i in range(3):
        a1.submit(*_incident(i, "sensor_01", 0.0))
    assert a2.submit(*_incident(9, "sensor_01", 0.0)) is False       # a2 reloads the ledger: budget already spent
