"""Exact (allowed, requires_approval, reason_code) for every decision branch.

Added after mutation testing showed that the original suite did not pin the
deny-versus-review distinction. The table itself lives in secureagent_iot.decision_table."""
import copy

import pytest

from secureagent_iot.decision_table import CASES, act, engine
from secureagent_iot.policy import IncidentRecord as _Inc, PolicyEngine as _Engine, TrustedState as _State
from secureagent_iot.schemas import ProposedAction as _Action


@pytest.mark.parametrize("name,ekw,akw,expected", CASES, ids=[c[0] for c in CASES])
def test_decision_table(cfg, registry, name, ekw, akw, expected):
    d = engine(cfg, registry, **ekw).evaluate(act(**akw))
    assert (d.allowed, d.requires_approval, d.reason_code) == expected


def test_approval_is_single_use(cfg, registry):
    eng = engine(cfg, registry, device="actuator_01", approve=("actuator_01", "restrict_egress"))
    first = eng.evaluate(act(device_id="actuator_01"))
    replay = eng.evaluate(act(device_id="actuator_01"))
    assert first.reason_code == "ALLOW_APPROVED_HIGH_IMPACT" and replay.reason_code == "REVIEW_HIGH_CRITICALITY"


def test_release_approval_is_single_use(cfg, registry):
    """Found by mutation testing: a release approval must not be replayable after the device is restricted again."""
    eng = engine(cfg, registry, restricted=("cam_01",), approve=("cam_01", "release_restriction"))
    rel = dict(action_type="release_restriction", duration_seconds=0)
    assert eng.evaluate(act(**rel)).reason_code == "ALLOW_RELEASE"
    assert eng.evaluate(act(**rel)).reason_code == "REVIEW_RELEASE_UNRESOLVED"


def test_resolved_release_does_not_spend_an_approval(cfg, registry):
    eng = engine(cfg, registry, device="actuator_01", resolved=True, restricted=("actuator_01",), approve=("actuator_01", "restrict_egress"))
    assert eng.evaluate(act(device_id="actuator_01", action_type="release_restriction", duration_seconds=0)).allowed
    assert eng.evaluate(act(device_id="actuator_01")).reason_code == "ALLOW_APPROVED_HIGH_IMPACT"



@pytest.mark.parametrize("device_cap, fleet_cap", [(900, 0), (0, 900)])
def test_hard_ceiling_boundary(cfg, registry, device_cap, fleet_cap):
    """A restriction that brings the load exactly to a hard ceiling is allowed; one second more is denied."""
    c = copy.deepcopy(cfg)
    for k, v in (("device_hard_cap_seconds", device_cap), ("fleet_hard_cap_seconds", fleet_cap)):
        c["policy"][k] = v
        c.setdefault("spec", {})[k] = v
    for used, allowed in ((600, True), (601, False)):
        st = _State(registry=registry, restricted_seconds=lambda d, w, u=used: u,
                    fleet_restricted_seconds=lambda w, u=used: u)
        st.record_incident(_Inc("inc-1", "cam_01", 0.95, frozenset({"ev-1", "ev-2", "ev-3"})))
        d = _Engine(c, st).evaluate(_Action(incident_id="inc-1", device_id="cam_01", action_type="restrict_egress", duration_seconds=300,
                                            evidence_refs=["ev-1", "ev-2"], reason="r", agent_id="responder"))
        assert d.allowed is allowed and (allowed or d.reason_code == "DENY_HARD_CEILING")
