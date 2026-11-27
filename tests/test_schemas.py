import pytest
from pydantic import ValidationError

from secureagent_iot.schemas import (ActionOutcome, Alert, Asset, PolicyDecision, ProposalRejected, ProposedAction,
                                     TelemetryEvent, parse_proposal)

GOOD = {"incident_id": "inc-1", "device_id": "cam_01", "action_type": "restrict_egress", "duration_seconds": 120,
        "evidence_refs": ["ev-1", "ev-2"], "reason": "r", "agent_id": "responder"}


def test_valid_proposal(registry):
    assert parse_proposal(dict(GOOD), registry).device_id == "cam_01"


def test_extra_field_rejected(registry):
    with pytest.raises(ProposalRejected) as e:
        parse_proposal({**GOOD, "shell": "rm -rf /"}, registry)
    assert e.value.code == "SCHEMA_INVALID"


@pytest.mark.parametrize("action", ["disable_monitoring", "grant_admin", "iptables", "", None, 3])
def test_unknown_action_rejected(registry, action):
    with pytest.raises(ProposalRejected):
        parse_proposal({**GOOD, "action_type": action}, registry)


def test_empty_evidence_rejected(registry):
    for refs in ([], [""], ["  "], None):
        with pytest.raises(ProposalRejected):
            parse_proposal({**GOOD, "evidence_refs": refs}, registry)


@pytest.mark.parametrize("dev", ["10.0.0.5", "cam_99; rm", "CAM_01", "example.com", "", "cam_1"])
def test_malformed_device_rejected(registry, dev):
    with pytest.raises(ProposalRejected) as e:
        parse_proposal({**GOOD, "device_id": dev}, registry)
    assert e.value.code == "SCHEMA_INVALID"


def test_unregistered_device_rejected(registry):
    with pytest.raises(ProposalRejected) as e:
        parse_proposal({**GOOD, "device_id": "cam_99"}, registry)
    assert e.value.code == "UNREGISTERED_DEVICE"


def test_strict_types(registry):
    for bad in ({"duration_seconds": "120"}, {"duration_seconds": -1}, {"duration_seconds": 10**7}, {"duration_seconds": 1.5}):
        with pytest.raises(ProposalRejected):
            parse_proposal({**GOOD, **bad}, registry)
    with pytest.raises(ProposalRejected):
        parse_proposal("restrict cam_01", registry)
    with pytest.raises(ProposalRejected):
        parse_proposal({k: v for k, v in GOOD.items() if k != "agent_id"}, registry)


def test_other_models_forbid_extras_and_bad_values():
    with pytest.raises(ValidationError):
        Alert(alert_id="a", device_id="cam_01", attack_family="x", probability=1.2, evidence_refs=[])
    with pytest.raises(ValidationError):
        Asset(device_id="cam_01", criticality="extreme", allowed_actions=["notify"], owner="o", is_online=True)
    with pytest.raises(ValidationError):
        Asset(device_id="cam_01", criticality="low", allowed_actions=["reboot"], owner="o", is_online=True)
    with pytest.raises(ValidationError):
        TelemetryEvent(event_id="e", device_id="cam_01", timestamp=0.0, features={}, source="s", extra=1)
    with pytest.raises(ValidationError):
        PolicyDecision(allowed=True, requires_approval=False, reason_code="x", policy_version="1", foo=1)
    with pytest.raises(ValidationError):
        ActionOutcome(executed="yes", success=True, service_disruption=False, elapsed_ms=1.0)
    ev = TelemetryEvent(event_id="e", device_id="cam_01", timestamp=1.0, features={"a": 1.0}, untrusted_text="x", source="s")
    with pytest.raises(ValidationError):
        ev.device_id = "cam_02"  # frozen
    assert ProposedAction(**GOOD).action_type == "restrict_egress"
