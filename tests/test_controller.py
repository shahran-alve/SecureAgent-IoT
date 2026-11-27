import socket
import subprocess

from secureagent_iot.controller import SimClock, SimulatedNetworkController


def make(registry):
    clock = SimClock()
    return SimulatedNetworkController(registry, clock), clock


def test_ttl_expiry_is_enforced_by_controller(registry):
    c, clock = make(registry)
    assert c.restrict_egress("cam_01", 300, "inc-1").executed
    clock.advance(299)
    assert c.is_restricted("cam_01")
    clock.advance(1)
    assert not c.is_restricted("cam_01")
    assert any(e["kind"] == "expired" for e in c.events)


def test_rollback_and_release(registry):
    c, _ = make(registry)
    c.restrict_egress("cam_01", 300, "inc-1")
    assert c.rollback("cam_01") and not c.is_restricted("cam_01")
    assert c.rollback("cam_01") is False
    c.restrict_egress("cam_02", 60, "inc-2")
    out = c.release_restriction("cam_02", "inc-2")
    assert out.executed and out.rollback_success and not c.is_restricted("cam_02")


def test_unknown_device_and_bad_duration_refused(registry):
    c, _ = make(registry)
    for dev in ("cam_99", "10.0.0.5", "example.com"):
        assert not c.restrict_egress(dev, 60, "inc").executed
        assert not c.notify(dev, "inc").executed
    for dur in (0, -5, 1.5, "60", True):
        assert not c.restrict_egress("cam_01", dur, "inc").executed
    assert c.rules == {}


def test_no_arbitrary_method_dispatch(registry):
    c, _ = make(registry)
    for name in ("disable_monitoring", "grant_admin", "rollback", "expire", "dump", "__init__", "os.system"):
        assert not c.execute_raw(name, "cam_01", 60, "inc").executed
    assert c.rules == {} and c.events[-1]["why"] == "unknown_action"


def test_hard_max_ttl_option(registry):
    c = SimulatedNetworkController(registry, SimClock(), hard_max_ttl=300)
    assert not c.restrict_egress("cam_01", 301, "inc").executed
    assert c.restrict_egress("cam_01", 300, "inc").executed


def test_service_disruption_flag_on_high_criticality(registry):
    c, _ = make(registry)
    assert c.restrict_egress("actuator_01", 60, "inc").service_disruption
    assert not c.restrict_egress("cam_01", 60, "inc").service_disruption


def test_controller_has_no_side_effect_channels(registry, monkeypatch):
    """Any attempt to open a socket or spawn a process during controller use fails the test."""
    def boom(*a, **k):
        raise AssertionError("controller attempted an external side effect")
    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    c, clock = make(registry)
    c.restrict_egress("cam_01", 10, "inc")
    c.notify("cam_01", "inc")
    c.request_review("cam_01", "inc")
    clock.advance(11)
    c.release_restriction("cam_01", "inc")
