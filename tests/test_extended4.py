"""Tests for experiments E30-E32 (extended4)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secureagent_iot import extended4 as R
from secureagent_iot.agents import AgenticSystem
from secureagent_iot.config import load_config
from secureagent_iot.evaluation import run_response_benchmark
from secureagent_iot.extended2 import _attack_frame


def _cfg():
    return load_config(None)


def test_approvals_never_violate_spec():
    cfg = _cfg()
    sysmap = {"rubber": R.ApprovingAnalyst(cfg, 1.0, 1.0, "rubber")}
    c = run_response_benchmark(cfg, [11], sysmap=sysmap)
    a = _attack_frame(cfg, [11], sysmap)
    assert c.spec_violation.sum() == 0
    assert a[a.injected & ~a.suppression_scored].any_violation.sum() == 0
    assert c.policy_decisions.str.contains("ANALYST_APPROVED").any()   # the approval path is actually exercised


def test_no_approval_setting_matches_plain_b2():
    cfg = _cfg()
    a = run_response_benchmark(cfg, [11], sysmap={"x": R.ApprovingAnalyst(cfg, 0.0, 0.0, "x")})
    b = run_response_benchmark(cfg, [11], sysmap={"x": AgenticSystem("x", cfg, guarded=True)})
    cols = ["executed_action", "executed_duration", "escalated", "safe_contained"]
    assert a[cols].equals(b[cols])


def test_quarantine_filter_off_equals_inner():
    cfg = _cfg()
    f = run_response_benchmark(cfg, [11], sysmap={"x": R.QuarantineFilter(AgenticSystem("i", cfg, guarded=True), 0.0, 0.0, "x")})
    b = run_response_benchmark(cfg, [11], sysmap={"x": AgenticSystem("x", cfg, guarded=True)})
    assert f[["executed_action", "escalated"]].equals(b[["executed_action", "escalated"]])


def test_manufactured_corroboration_needs_k_devices():
    d = R.manufactured_corroboration(_cfg(), [11], compromised=(0, 1, 2))
    corr = d[d.defense.str.contains("corroboration")].set_index("compromised_devices").wrongful_device_hours
    plain = d[~d.defense.str.contains("corroboration")].set_index("compromised_devices").wrongful_device_hours
    assert corr[0] == corr[1] == plain[0]
    assert corr[2] > plain[2]
