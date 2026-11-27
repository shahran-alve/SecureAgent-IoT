"""Atomic authorization and durable restriction history.

The budget theorem assumes that each admission decision sees every restriction granted before it, on a clock that
does not jump. This module provides the pieces a deployment needs for that assumption to hold, and the failure-
injection experiment that shows what goes wrong without them:

* `RestrictionLedger`: an append-only, hash-chained, file-backed record of every granted restriction. Budgets are
  computed from it, so a controller restart does not reset them, and a ledger that fails verification makes the
  authorizer fail closed (no autonomous restriction).
* `AtomicAuthorizer`: evaluates a proposal and commits the grant to the ledger under one lock, before the controller
  installs the rule (write-ahead). Concurrent requests cannot read the same remaining budget. If installation fails
  after the commit, the ledger still counts the grant, which can only over-count.
* Budget windows use a monotonic time source, so a wall-clock step (for example an NTP correction) cannot make the
  window forget grants.

Everything here runs in memory or in a temporary directory; nothing touches a real network.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import tempfile
import threading
import time
from pathlib import Path

import pandas as pd

from .policy import IncidentRecord, PolicyEngine, TrustedState
from .schemas import ProposedAction
from .simulator import build_registry

GENESIS = "0" * 64


KINDS = ("aut", "cor", "app", "rel")   # autonomous, corroboration-exempt, approved; release (duration 0)
KIND_OF_CODE = {"ALLOW_RESTRICT": "aut", "ALLOW_CORROBORATED": "cor",
                "ALLOW_APPROVED_HIGH_IMPACT": "app", "ALLOW_APPROVED_OVER_BUDGET": "app"}
RECORD_FIELDS = ("t", "device_id", "duration", "kind", "approval", "prev")


def _canonical(body: dict) -> bytes:
    """Unambiguous serialization of a record body: fixed field set, sorted keys, no whitespace, UTF-8."""
    return json.dumps({k: body[k] for k in RECORD_FIELDS}, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


class RestrictionLedger:
    """Append-only, hash-chained ledger of granted restrictions in one JSONL file.

    Record: t (decision time, monotonic), device_id, duration, kind (aut | cor | app), approval (the consumed approval
    key, or ""), prev, and hash = SHA-256 of the canonical serialization of the other fields. The kind lets the exempt
    load be rebuilt after a restart, and the approval field makes single-use consumption durable.

    Durability: every append is flushed and fsync'ed before `append` returns (assuming the storage honours fsync).
    Crash consistency: a torn last record fails to parse or verify; `ok` becomes False and the authorizer fails closed.

    What the chain detects: an edited, reordered, or deleted interior record, and a torn record. What it does not
    detect on its own: removal of a valid suffix, or rollback of the whole file to an earlier valid state. With
    `anchor_path`, the ledger also keeps a head commitment (record count and last hash), replaced atomically after
    each append; a ledger shorter than its anchor, or whose record at the anchored position has a different hash,
    fails verification. The anchor defeats suffix truncation only if it is stored where the party that can truncate
    the ledger cannot also rewrite it (for example a write-once or remote store); rollback of both is not detected."""

    def __init__(self, path: Path, anchor_path: Path | None = None):
        self.path = Path(path)
        self.anchor_path = Path(anchor_path) if anchor_path else None
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.entries: list[dict] = []
        self.ok = True
        self.refresh()

    def refresh(self) -> None:
        self.entries, self.ok = [], True
        if self.path.exists():
            self._load()
        if self.ok and self.anchor_path and self.anchor_path.exists():
            try:
                a = json.loads(self.anchor_path.read_text())
                n, head = int(a["n"]), a["head"]
            except (ValueError, KeyError, TypeError):
                self.ok = False
                return
            # the ledger may be one record ahead of its anchor (crash between the two writes), never behind
            if len(self.entries) < n or (n > 0 and self.entries[n - 1]["hash"] != head):
                self.ok = False

    def _load(self) -> None:
        prev = GENESIS
        self.entries = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                digest = hashlib.sha256(_canonical(e)).hexdigest()
            except (ValueError, KeyError, TypeError):
                self.ok = False                                # torn or corrupted record
                return
            if e["prev"] != prev or e.get("hash") != digest or e["kind"] not in KINDS:
                self.ok = False
                return
            self.entries.append(e)
            prev = e["hash"]

    def lock(self):
        """Exclusive inter-process lock (context manager)."""
        ledger = self

        class _L:
            def __enter__(self_inner):
                self_inner.fd = os.open(ledger.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
                fcntl.flock(self_inner.fd, fcntl.LOCK_EX)
                return self_inner

            def __exit__(self_inner, *a):
                fcntl.flock(self_inner.fd, fcntl.LOCK_UN)
                os.close(self_inner.fd)
        return _L()

    def append(self, t: float, device_id: str, duration: float, kind: str = "aut", approval: str = "") -> dict:
        if kind not in KINDS:
            raise ValueError(kind)
        prev = self.entries[-1]["hash"] if self.entries else GENESIS
        body = {"t": float(t), "device_id": device_id, "duration": float(duration), "kind": kind,
                "approval": approval, "prev": prev}
        e = {**body, "hash": hashlib.sha256(_canonical(body)).hexdigest()}
        with open(self.path, "a") as f:
            f.write(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.entries.append(e)
        if self.anchor_path:
            tmp = self.anchor_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                f.write(json.dumps({"n": len(self.entries), "head": e["hash"]}))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.anchor_path)                  # atomic replacement of the head commitment
        return e

    def device_seconds(self, device_id: str, now: float, window: float) -> float:
        return sum(e["duration"] for e in self.entries if e["device_id"] == device_id and e["t"] >= now - window)

    def fleet_seconds(self, now: float, window: float) -> float:
        return sum(e["duration"] for e in self.entries if e["t"] >= now - window)

    def exempt_log(self) -> list:
        return [(e["t"], e["duration"]) for e in self.entries if e["kind"] == "cor"]

    def consumed_approvals(self) -> set:
        """Approvals consumed by any committed record: approved restrictions and approved releases."""
        return {e["approval"] for e in self.entries if e["approval"]}

    def is_restricted(self, device_id: str, now: float) -> bool:
        """A device is restricted if a grant to it is unexpired and no release was committed after that grant."""
        active = False
        for e in self.entries:
            if e["device_id"] != device_id:
                continue
            if e["kind"] == "rel":
                active = False
            elif e["t"] + e["duration"] > now:
                active = True
        return active


def approval_id(key) -> str:
    return "|".join(key)


class AtomicAuthorizer:
    """Algorithm 2: evaluate and commit under one lock; fail closed if the ledger cannot be trusted.

    * Serialization: a thread mutex and, with `file_lock`, an exclusive flock shared by every authorizer process.
    * Before each decision the ledger is reloaded and verified; device, fleet, and exempt loads and the set of
      consumed approvals are rebuilt from it, so a restarted or second process sees every committed grant.
    * Approvals: validated and consumed by the engine inside the same critical section, and the consumed key is
      written into the grant's ledger record. A consumed approval cannot be reused by a concurrent request or after
      a restart. If installing the rule then fails, the approval stays consumed and the grant stays counted
      (conservative in both respects).
    * Absolute expiry: a grant committed at decision time s with duration d expires at e = s + d. The rule is
      installed with the remaining lifetime max(0, e - t_install); a grant whose expiry has passed is not installed.
      `install(device_id, remaining, expires_at)` receives both, so a controller adapter can enforce the deadline
      rather than restart a relative timer."""

    def __init__(self, cfg, registry, ledger: RestrictionLedger, clock, install=None, file_lock: bool = True,
                 uninstall=None):
        self.ledger, self.clock, self.lock, self.file_lock = ledger, clock, threading.Lock(), file_lock
        self.install = install or (lambda device_id, remaining, expires_at: None)
        self.uninstall = uninstall or (lambda device_id: None)
        self.state = TrustedState(registry=registry, now=clock,
                                  is_restricted=lambda d: ledger.is_restricted(d, clock()),
                                  restricted_seconds=lambda d, w: ledger.device_seconds(d, clock(), w),
                                  fleet_restricted_seconds=lambda w: ledger.fleet_seconds(clock(), w))
        self.engine = PolicyEngine(cfg, self.state)
        self.installed: list[tuple[str, float, float]] = []

    def _sync_state(self) -> None:
        self.state.exempt_log = self.ledger.exempt_log()
        for k in list(self.state.approvals):
            if approval_id(k) in self.ledger.consumed_approvals():
                self.state.approvals.discard(k)
                self.state.consumed.add(k)

    def _decide_and_commit(self, rec: IncidentRecord, action: ProposedAction):
        if not self.ledger.ok:
            return None                                        # history cannot be trusted: fail closed
        self._sync_state()
        self.state.record_incident(rec)
        key = (action.incident_id, action.device_id, action.action_type)
        d = self.engine.evaluate(action)
        if d.allowed and action.action_type == "release_restriction":
            # a release adds no restriction time; it is committed so that the approval it may have consumed is durable
            used = key if (key in self.state.consumed and approval_id(key) not in self.ledger.consumed_approvals()) else None
            return self.ledger.append(self.clock(), action.device_id, 0.0, "rel", approval_id(used) if used else "")
        if not (d.allowed and action.action_type == "restrict_egress"):
            return None
        kind = KIND_OF_CODE.get(d.reason_code, "aut")
        t = self.clock()
        return self.ledger.append(t, action.device_id, action.duration_seconds, kind,
                                  approval_id(key) if kind == "app" else "")     # write-ahead, fsync'ed

    def submit(self, rec: IncidentRecord, action: ProposedAction) -> bool:
        with self.lock:
            if self.file_lock:
                with self.ledger.lock():                       # serialize with other authorizer processes
                    self.ledger.refresh()                      # see every grant committed by any process
                    e = self._decide_and_commit(rec, action)
            else:
                e = self._decide_and_commit(rec, action)
            if e is None:
                return False
        if e["kind"] == "rel":
            try:
                self.uninstall(action.device_id)
            except Exception:
                pass                                           # release committed; the rule still lapses at its expiry
            return True
        expires_at = e["t"] + e["duration"]
        remaining = max(0.0, expires_at - self.clock())
        if remaining > 0:
            try:
                self.install(action.device_id, remaining, expires_at)
                self.installed.append((action.device_id, remaining, expires_at))
            except Exception:
                pass                                           # grant already counted: over-count only
        return True


class NaiveAuthorizer:
    """The same engine with check-then-act outside a lock and history held in memory (the unsafe design)."""

    def __init__(self, cfg, registry, clock, gap: float = 0.0):
        self.history: list[tuple[float, str, float]] = []
        self.clock, self.gap = clock, gap
        self.state = TrustedState(registry=registry, now=clock,
                                  restricted_seconds=lambda d, w: sum(x for t, dd, x in self.history if dd == d and t >= clock() - w),
                                  fleet_restricted_seconds=lambda w: sum(x for t, dd, x in self.history if t >= clock() - w))
        self.engine = PolicyEngine(cfg, self.state)

    def submit(self, rec: IncidentRecord, action: ProposedAction) -> bool:
        self.state.record_incident(rec)
        d = self.engine.evaluate(action)
        if not (d.allowed and action.action_type == "restrict_egress"):
            return False
        time.sleep(self.gap)                                   # time between the check and the commit
        self.history.append((self.clock(), action.device_id, float(action.duration_seconds)))
        return True


def _incident(n: int, device: str, now: float) -> tuple[IncidentRecord, ProposedAction]:
    iid = f"fi-{n}"
    ev = (f"{iid}-a", f"{iid}-b")
    rec = IncidentRecord(iid, device, 0.99, frozenset(ev), evidence_kinds={ev[0]: "flow_anomaly", ev[1]: "peer_correlation"},
                         evidence_sources={ev[0]: "netflow-collector", ev[1]: "peer-correlator"}, evidence_time={e: now for e in ev})
    act = ProposedAction(incident_id=iid, device_id=device, action_type="restrict_egress", duration_seconds=300,
                         evidence_refs=list(ev), reason="failure injection", agent_id="responder")
    return rec, act


def failure_injection(cfg: dict, trials: int = 20, trial_log: list | None = None) -> pd.DataFrame:
    """Failure modes, each run against the naive design and the atomic, ledger-backed design.

    If `trial_log` is a list, one dict per trial (mode, design, trial index, measured value, unit) is appended to it."""
    registry = build_registry()
    B = float(cfg["policy"]["device_budget_seconds"])
    dev = "cam_02"
    rows = []

    def log(mode, design, values, unit):
        if trial_log is not None:
            trial_log.extend({"failure": mode, "design": design, "trial": i, "value": float(v), "unit": unit}
                             for i, v in enumerate(values))

    def row(mode, design, granted, note):
        log(mode, design, granted, "granted seconds for the device in one window")
        rows.append({"evidence_class": "testbed-synthetic (real policy engine; failures injected in software)",
                     "failure": mode, "design": design, "trials": trials,
                     "max_granted_s_in_one_window": max(granted), "device_budget_s": B,
                     "trials_over_budget": sum(g > B for g in granted), "note": note})

    # 1. concurrent requests that all read the remaining budget before any commit
    for design in ("naive", "atomic"):
        granted = []
        for k in range(trials):
            now = lambda: 0.0
            if design == "naive":
                auth = NaiveAuthorizer(cfg, registry, now, gap=0.01)
            else:
                tmp = Path(tempfile.mkdtemp()) / "ledger.jsonl"
                auth = AtomicAuthorizer(cfg, registry, RestrictionLedger(tmp), now)
            threads = [threading.Thread(target=auth.submit, args=_incident(100 * k + i, dev, 0.0)) for i in range(8)]
            [t.start() for t in threads]; [t.join() for t in threads]
            granted.append(sum(x for _, _, x in auth.history) if design == "naive" else auth.ledger.device_seconds(dev, 0.0, 3600))
        row("8 concurrent requests for one device", design, granted,
            "check and commit not atomic" if design == "naive" else "evaluate and commit under one lock")

    # 2. controller restart: in-memory history is lost, ledger history is reloaded
    for design in ("naive", "atomic"):
        granted = []
        for k in range(trials):
            t = {"now": 0.0}
            clock = lambda: t["now"]
            tmp = Path(tempfile.mkdtemp()) / "ledger.jsonl"
            mk = (lambda: NaiveAuthorizer(cfg, registry, clock)) if design == "naive" else \
                 (lambda: AtomicAuthorizer(cfg, registry, RestrictionLedger(tmp), clock))
            auth, total = mk(), 0.0
            for i in range(6):
                if i == 3:
                    auth = mk()                                    # crash and restart mid-hour
                total += 300 * auth.submit(*_incident(100 * k + i, dev, t["now"]))
                t["now"] += 310
            granted.append(total)
        row("controller restart within the window", design, granted,
            "history in memory, lost on restart" if design == "naive" else "history reloaded from the hash-chained ledger")

    # 3. wall-clock step forward by one window (e.g. an NTP correction) between requests
    for design in ("naive", "atomic"):
        granted = []
        for k in range(trials):
            wall = {"t": 0.0}
            mono = {"t": 0.0}
            clock = (lambda: wall["t"]) if design == "naive" else (lambda: mono["t"])
            tmp = Path(tempfile.mkdtemp()) / "ledger.jsonl"
            auth = NaiveAuthorizer(cfg, registry, clock) if design == "naive" else AtomicAuthorizer(cfg, registry, RestrictionLedger(tmp), clock)
            total = 0.0
            for i in range(6):
                if i == 3:
                    wall["t"] += 3600                               # wall clock jumps; monotonic time does not
                total += 300 * auth.submit(*_incident(100 * k + i, dev, clock()))
                wall["t"] += 310; mono["t"] += 310
            granted.append(total)
        row("wall-clock step forward by one window", design, granted,
            "budget window on wall-clock time" if design == "naive" else "budget window on monotonic time")

    # 4. rule installation fails after the grant (interrupted commit)
    granted = []
    for k in range(trials):
        tmp = Path(tempfile.mkdtemp()) / "ledger.jsonl"
        t = {"now": 0.0}
        fail = {"n": 0}

        def install(d, dur, exp):
            fail["n"] += 1
            if fail["n"] % 2:
                raise RuntimeError("controller crashed during install")
        auth = AtomicAuthorizer(cfg, registry, RestrictionLedger(tmp), lambda: t["now"], install=install)
        for i in range(6):
            auth.submit(*_incident(100 * k + i, dev, t["now"])); t["now"] += 310
        granted.append(auth.ledger.device_seconds(dev, t["now"], 3600))
    row("rule installation fails after the grant", "atomic", granted, "failed installs still counted; budget only over-counted")

    # 5. tampered history: one ledger entry deleted on disk
    refused = []
    for k in range(trials):
        tmp = Path(tempfile.mkdtemp()) / "ledger.jsonl"
        t = {"now": 0.0}
        auth = AtomicAuthorizer(cfg, registry, RestrictionLedger(tmp), lambda: t["now"])
        for i in range(2):
            auth.submit(*_incident(100 * k + i, dev, t["now"])); t["now"] += 310
        lines = tmp.read_text().splitlines()
        tmp.write_text("\n".join(lines[1:]) + "\n")              # delete the first grant
        auth2 = AtomicAuthorizer(cfg, registry, RestrictionLedger(tmp), lambda: t["now"])
        refused.append(0.0 if not auth2.submit(*_incident(100 * k + 9, dev, t["now"])) else 300.0)
    log("ledger entry deleted on disk", "atomic", refused, "seconds granted after tampering (0 = refused)")
    rows.append({"evidence_class": "testbed-synthetic (real policy engine; failures injected in software)",
                 "failure": "ledger entry deleted on disk", "design": "atomic", "trials": trials,
                 "max_granted_s_in_one_window": max(refused), "device_budget_s": B, "trials_over_budget": 0,
                 "note": f"chain verification failed in {sum(r == 0 for r in refused)}/{trials}; authorizer failed closed"})

    # 6. two authorizer processes share one ledger file, four concurrent requests each
    for design, flock in (("per-process lock only", False), ("atomic", True)):
        granted = []
        for k in range(trials):
            tmp = Path(tempfile.mkdtemp()) / "ledger.jsonl"
            ctx = mp.get_context("fork")
            barrier, q = ctx.Barrier(2), ctx.Queue()
            procs = [ctx.Process(target=_process_worker, args=(cfg, str(tmp), flock, 1000 * k + 10 * j, dev, barrier, q)) for j in range(2)]
            [p.start() for p in procs]
            got = [q.get() for _ in procs]
            [p.join() for p in procs]
            granted.append(float(sum(got)))                    # granted seconds as reported to the controllers
        row("two authorizer processes on one ledger", design, granted,
            "each process trusts its own in-memory view" if not flock else "flock on the ledger, reload before each decision")

    # 7. crash during an append leaves a torn last record
    refused = []
    for k in range(trials):
        tmp = Path(tempfile.mkdtemp()) / "ledger.jsonl"
        t = {"now": 0.0}
        auth = AtomicAuthorizer(cfg, registry, RestrictionLedger(tmp), lambda: t["now"])
        auth.submit(*_incident(100 * k, dev, 0.0)); t["now"] += 310
        line = tmp.read_text().splitlines()[0]
        with open(tmp, "a") as f:
            f.write(line[: len(line) // 2])                     # half-written second record, no newline
        auth2 = AtomicAuthorizer(cfg, registry, RestrictionLedger(tmp), lambda: t["now"])
        refused.append(0.0 if not auth2.submit(*_incident(100 * k + 9, dev, t["now"])) else 300.0)
    log("crash during a ledger append (torn record)", "atomic", refused, "seconds granted after the torn write (0 = refused)")
    rows.append({"evidence_class": "testbed-synthetic (real policy engine; failures injected in software)",
                 "failure": "crash during a ledger append (torn record)", "design": "atomic", "trials": trials,
                 "max_granted_s_in_one_window": max(refused), "device_budget_s": B, "trials_over_budget": 0,
                 "note": f"torn record detected in {sum(r == 0 for r in refused)}/{trials}; authorizer failed closed"})
    # 8. delayed installation: the controller is reached 100 s after the grant was committed
    for design in ("relative timer", "absolute expiry"):
        ends = []
        for k in range(trials):
            tmp = Path(tempfile.mkdtemp()) / "ledger.jsonl"
            t = {"now": 0.0}
            got = {}
            auth = AtomicAuthorizer(cfg, registry, RestrictionLedger(tmp), lambda: t["now"],
                                    install=lambda d, rem, exp: got.update(rem=rem, exp=exp))
            orig = auth.ledger.append

            def delayed_append(*a, **kw):
                e = orig(*a, **kw)
                t["now"] = 100.0                            # commit at t = 0, installation at t = 100
                return e
            auth.ledger.append = delayed_append
            auth.submit(*_incident(100 * k, dev, 0.0))
            e = auth.ledger.entries[-1]
            if design == "relative timer":
                ends.append(t["now"] + e["duration"])          # a timer restarted with the full duration
            else:
                ends.append(t["now"] + got["rem"])             # remaining lifetime passed to install()
        over = [x - 300.0 for x in ends]
        log("rule installed 100 s after the grant", design, over, "seconds the rule stayed active past s + delta")
        rows.append({"evidence_class": "testbed-synthetic (real policy engine; failures injected in software)",
                     "failure": "rule installed 100 s after the grant", "design": design, "trials": trials,
                     "max_granted_s_in_one_window": 300.0, "device_budget_s": B, "trials_over_budget": 0,
                     "note": f"rule active until s + delta + {max(over):.0f} s"})

    # 9. a valid suffix of the ledger is removed (truncation), with and without a head commitment
    for design, anchored in (("chain only", False), ("chain + head commitment", True)):
        granted, refused = [], 0
        for k in range(trials):
            d0 = Path(tempfile.mkdtemp())
            tmp, anc = d0 / "ledger.jsonl", (d0 / "anchor.json" if anchored else None)
            t = {"now": 0.0}
            auth = AtomicAuthorizer(cfg, registry, RestrictionLedger(tmp, anc), lambda: t["now"])
            for i in range(3):
                auth.submit(*_incident(100 * k + i, dev, t["now"])); t["now"] += 310
            lines = tmp.read_text().splitlines()
            tmp.write_text("\n".join(lines[:1]) + "\n")       # keep a valid prefix: the chain still verifies
            auth2 = AtomicAuthorizer(cfg, registry, RestrictionLedger(tmp, anc), lambda: t["now"])
            ok = [auth2.submit(*_incident(100 * k + 10 + i, dev, t["now"])) for i in range(3)]
            refused += not any(ok)
            granted.append(900.0 + 300.0 * sum(ok))             # real restriction time in the window
        row("valid ledger suffix removed (truncation)", design, granted,
            f"truncation detected in {refused}/{trials}" + ("; authorizer failed closed" if anchored else "; the chain alone cannot detect it"))

    # 10. one approval, eight concurrent requests, then a restart that reloads the approval store
    for design in ("in-memory consumption", "atomic"):
        approved = []
        for k in range(trials):
            tmp = Path(tempfile.mkdtemp()) / "ledger.jsonl"
            t = {"now": 0.0}
            clock = lambda: t["now"]
            mk = (lambda: NaiveAuthorizer(cfg, registry, clock)) if design == "in-memory consumption" else \
                 (lambda: AtomicAuthorizer(cfg, registry, RestrictionLedger(tmp), clock))
            auth = mk()
            for i in range(3):                                  # spend the device budget (900 s)
                auth.submit(*_incident(1000 * k + i, dev, 0.0))
            rec, act = _incident(1000 * k + 99, dev, 0.0)
            auth.state.approve(rec.incident_id, dev, "restrict_egress")
            ths = [threading.Thread(target=auth.submit, args=(rec, act)) for _ in range(8)]
            [x.start() for x in ths]; [x.join() for x in ths]
            auth2 = mk() if design == "atomic" else NaiveAuthorizer(cfg, registry, clock)
            if design != "atomic":
                auth2.history = list(auth.history)              # even with history kept, consumption is lost
            auth2.state.approve(rec.incident_id, dev, "restrict_egress")   # the approval store still lists it
            auth2.submit(rec, act)
            if design == "atomic":
                approved.append(sum(e["kind"] == "app" for e in auth2.ledger.entries))
            else:
                approved.append(len(auth2.history) - 3)
        log("one approval: 8 concurrent requests, then a restart", design, approved, "grants made on the one approval")
        rows.append({"evidence_class": "testbed-synthetic (real policy engine; failures injected in software)",
                     "failure": "one approval: 8 concurrent requests, then a restart", "design": design, "trials": trials,
                     "max_granted_s_in_one_window": 900.0 + 300.0 * max(approved), "device_budget_s": B,
                     "trials_over_budget": sum(a > 1 for a in approved),
                     "note": f"approved grants per approval: max {max(approved)} (single use requires 1)"})
    # 11. one release approval: 8 concurrent releases, then a restart (atomic design only)
    used = []
    for k in range(trials):
        tmp = Path(tempfile.mkdtemp()) / "ledger.jsonl"
        t = {"now": 0.0}
        clock = lambda: t["now"]
        auth = AtomicAuthorizer(cfg, registry, RestrictionLedger(tmp), clock)
        rec, act = _incident(2000 * k, dev, 0.0)
        auth.submit(rec, act)
        rel = act.model_copy(update={"action_type": "release_restriction", "duration_seconds": 0})
        auth.state.approve(rec.incident_id, dev, "release_restriction")
        ths = [threading.Thread(target=auth.submit, args=(rec, rel)) for _ in range(8)]
        [x.start() for x in ths]; [x.join() for x in ths]
        rec2, act2 = _incident(2000 * k + 1, dev, 0.0)
        auth.submit(rec2, act2)                                 # restricted again
        auth2 = AtomicAuthorizer(cfg, registry, RestrictionLedger(tmp), clock)
        auth2.state.approve(rec.incident_id, dev, "release_restriction")
        auth2.submit(rec, rel)
        used.append(sum(1 for e in auth2.ledger.entries if e["kind"] == "rel" and e["approval"]))
    log("one release approval: 8 concurrent releases, then a restart", "atomic", used, "releases made on the one approval")
    rows.append({"evidence_class": "testbed-synthetic (real policy engine; failures injected in software)",
                 "failure": "one release approval: 8 concurrent releases, then a restart", "design": "atomic",
                 "trials": trials, "max_granted_s_in_one_window": float("nan"), "device_budget_s": B,
                 "trials_over_budget": sum(u != 1 for u in used),
                 "note": f"releases per approval: max {max(used)}, min {min(used)} (single use requires 1)"})
    return pd.DataFrame(rows)


def _process_worker(cfg, path, flock, base, dev, barrier, q):
    """One authorizer process: four concurrent requests at t = 0 against the shared ledger file."""
    registry = build_registry()
    auth = AtomicAuthorizer(cfg, registry, RestrictionLedger(Path(path)), lambda: 0.0, file_lock=flock)
    barrier.wait()
    out = []
    ths = [threading.Thread(target=lambda i=i: out.append(auth.submit(*_incident(base + i, dev, 0.0)))) for i in range(4)]
    [t.start() for t in ths]; [t.join() for t in ths]
    q.put(300.0 * sum(out))
