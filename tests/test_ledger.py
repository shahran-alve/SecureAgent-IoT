"""Ledger kind/approval persistence, head commitment, absolute expiry, single-use approvals."""
import json, sys, tempfile, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from secureagent_iot.config import load_config
from secureagent_iot.enforcement import (AtomicAuthorizer, RestrictionLedger, _incident, build_registry,
                                         failure_injection)

CFG = load_config(None)
REG = build_registry()


def _tmp():
    return Path(tempfile.mkdtemp())


def test_record_has_kind_and_canonical_hash():
    d = _tmp(); L = RestrictionLedger(d / "l.jsonl")
    L.append(0.0, "cam_02", 300, "cor"); L.append(1.0, "cam_02", 300, "app", "i|cam_02|restrict_egress")
    L2 = RestrictionLedger(d / "l.jsonl")
    assert L2.ok and [e["kind"] for e in L2.entries] == ["cor", "app"]
    assert L2.exempt_log() == [(0.0, 300.0)] and L2.consumed_approvals() == {"i|cam_02|restrict_egress"}


def test_kind_edit_breaks_chain():
    d = _tmp(); L = RestrictionLedger(d / "l.jsonl"); L.append(0.0, "cam_02", 300, "cor")
    rec = json.loads((d / "l.jsonl").read_text()); rec["kind"] = "aut"
    (d / "l.jsonl").write_text(json.dumps(rec) + "\n")
    assert not RestrictionLedger(d / "l.jsonl").ok


def test_truncation_detected_only_with_anchor():
    d = _tmp(); L = RestrictionLedger(d / "l.jsonl", d / "a.json")
    for i in range(3):
        L.append(float(i), "cam_02", 300)
    lines = (d / "l.jsonl").read_text().splitlines()
    (d / "l.jsonl").write_text(lines[0] + "\n")
    assert RestrictionLedger(d / "l.jsonl").ok                 # chain alone accepts the valid prefix
    assert not RestrictionLedger(d / "l.jsonl", d / "a.json").ok


def test_anchor_tolerates_ledger_one_ahead():
    d = _tmp(); L = RestrictionLedger(d / "l.jsonl", d / "a.json"); L.append(0.0, "cam_02", 300)
    anchor = (d / "a.json").read_text()
    L.append(1.0, "cam_02", 300)
    (d / "a.json").write_text(anchor)                           # crash between ledger fsync and anchor replace
    assert RestrictionLedger(d / "l.jsonl", d / "a.json").ok


def test_install_uses_remaining_lifetime():
    t = {"now": 0.0}; got = {}
    auth = AtomicAuthorizer(CFG, REG, RestrictionLedger(_tmp() / "l.jsonl"), lambda: t["now"],
                            install=lambda d, rem, exp: got.update(rem=rem, exp=exp))
    orig = auth.ledger.append

    def late(*a, **k):
        e = orig(*a, **k); t["now"] = 120.0; return e
    auth.ledger.append = late
    assert auth.submit(*_incident(1, "cam_02", 0.0))
    assert got["exp"] == 300.0 and got["rem"] == 180.0


def test_expired_grant_not_installed():
    t = {"now": 0.0}; calls = []
    auth = AtomicAuthorizer(CFG, REG, RestrictionLedger(_tmp() / "l.jsonl"), lambda: t["now"],
                            install=lambda *a: calls.append(a))
    orig = auth.ledger.append

    def late(*a, **k):
        e = orig(*a, **k); t["now"] = 400.0; return e
    auth.ledger.append = late
    assert auth.submit(*_incident(2, "cam_02", 0.0))           # still counted
    assert calls == [] and auth.ledger.device_seconds("cam_02", 0.0, 3600) == 300.0


def test_approval_single_use_concurrent_and_after_restart():
    d = _tmp(); t = {"now": 0.0}; clock = lambda: t["now"]
    auth = AtomicAuthorizer(CFG, REG, RestrictionLedger(d / "l.jsonl"), clock)
    for i in range(3):
        auth.submit(*_incident(10 + i, "cam_02", 0.0))
    rec, act = _incident(99, "cam_02", 0.0)
    auth.state.approve(rec.incident_id, "cam_02", "restrict_egress")
    ths = [threading.Thread(target=auth.submit, args=(rec, act)) for _ in range(8)]
    [x.start() for x in ths]; [x.join() for x in ths]
    auth2 = AtomicAuthorizer(CFG, REG, RestrictionLedger(d / "l.jsonl"), clock)
    auth2.state.approve(rec.incident_id, "cam_02", "restrict_egress")
    assert not auth2.submit(rec, act)
    assert sum(e["kind"] == "app" for e in auth2.ledger.entries) == 1


def test_failed_install_consumes_approval():
    d = _tmp(); t = {"now": 0.0}

    def boom(*a):
        raise RuntimeError("install failed")
    auth = AtomicAuthorizer(CFG, REG, RestrictionLedger(d / "l.jsonl"), lambda: t["now"], install=boom)
    for i in range(3):
        auth.submit(*_incident(10 + i, "cam_02", 0.0))
    rec, act = _incident(98, "cam_02", 0.0)
    auth.state.approve(rec.incident_id, "cam_02", "restrict_egress")
    assert auth.submit(rec, act)
    auth.state.approve(rec.incident_id, "cam_02", "restrict_egress")
    assert not auth.submit(rec, act)


def test_exempt_load_rebuilt_after_restart():
    d = _tmp(); L = RestrictionLedger(d / "l.jsonl"); L.append(0.0, "cam_03", 600, "cor")
    auth = AtomicAuthorizer(CFG, REG, RestrictionLedger(d / "l.jsonl"), lambda: 10.0)
    auth._sync_state()
    assert auth.state.exempt_seconds(3600) == 600.0


def test_failure_injection_new_modes():
    df = failure_injection(CFG, trials=2)
    trunc = df[df.failure.str.contains("truncation")].set_index("design")
    assert trunc.loc["chain + head commitment", "trials_over_budget"] == 0
    appr = df[df.failure.str.contains("one approval")].set_index("design")
    assert appr.loc["atomic", "trials_over_budget"] == 0


def _release(act):
    return act.model_copy(update={"action_type": "release_restriction", "duration_seconds": 0})


def test_release_approval_single_use_concurrent_and_after_restart():
    d = _tmp(); t = {"now": 0.0}; clock = lambda: t["now"]
    released = []
    auth = AtomicAuthorizer(CFG, REG, RestrictionLedger(d / "l.jsonl"), clock, uninstall=released.append)
    rec, act = _incident(500, "cam_02", 0.0)
    assert auth.submit(rec, act)                                # restricted, incident unresolved
    rel = _release(act)
    assert not auth.submit(rec, rel)                            # no approval: goes to review
    auth.state.approve(rec.incident_id, "cam_02", "release_restriction")
    ths = [threading.Thread(target=auth.submit, args=(rec, rel)) for _ in range(8)]
    [x.start() for x in ths]; [x.join() for x in ths]
    recs = [e for e in auth.ledger.entries if e["kind"] == "rel"]
    assert len(recs) == 1 and recs[0]["approval"] and released == ["cam_02"]
    # restrict again, then a restarted authorizer whose approval store still lists the used approval
    rec2, act2 = _incident(501, "cam_02", 0.0)
    assert auth.submit(rec2, act2)
    auth2 = AtomicAuthorizer(CFG, REG, RestrictionLedger(d / "l.jsonl"), clock)
    auth2.state.approve(rec.incident_id, "cam_02", "release_restriction")
    auth2.state.record_incident(rec2)
    assert not auth2.submit(rec, rel)
    assert sum(e["kind"] == "rel" for e in auth2.ledger.entries) == 1


def test_release_does_not_change_loads():
    d = _tmp(); auth = AtomicAuthorizer(CFG, REG, RestrictionLedger(d / "l.jsonl"), lambda: 0.0)
    rec, act = _incident(600, "cam_02", 0.0)
    auth.submit(rec, act)
    auth.state.approve(rec.incident_id, "cam_02", "release_restriction")
    assert auth.submit(rec, _release(act))
    assert auth.ledger.device_seconds("cam_02", 0.0, 3600) == 300.0
    assert not auth.ledger.is_restricted("cam_02", 0.0)
