"""Append-only, hash-chained JSONL audit log.

Integrity guarantee, stated narrowly: each record carries the SHA-256 of the
previous record, so *in-place edits, deletions or reordering of earlier
records are detectable by `verify()`* as long as the verifier holds the final
hash from an independent place. It is NOT tamper-proof: someone with write
access can rewrite the whole chain. External anchoring (e.g. periodically
publishing the head hash) is out of scope.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

GENESIS = "0" * 64


def _canon(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path | None = None, now: Callable[[], str] | None = None):
        self.path = Path(path) if path else None
        self._now = now or (lambda: datetime.now(timezone.utc).isoformat())
        self.records: list[dict] = []
        self.head = GENESIS
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.records.append(rec)
                    self.head = rec["record_hash"]

    def append(self, *, incident_id: str, evidence_ids: list[str], action: dict | None, decision: dict,
               policy_version: str, system: str = "") -> dict:
        body = {
            "seq": len(self.records),
            "timestamp": self._now(),
            "system": system,
            "incident_id": incident_id,
            "evidence_ids": sorted(evidence_ids),
            "action": action,
            "decision": decision,
            "policy_version": policy_version,
            "content_hash": _sha(_canon({"incident_id": incident_id, "evidence_ids": sorted(evidence_ids),
                                         "action": action, "decision": decision})),
            "prev_hash": self.head,
        }
        body["record_hash"] = _sha(_canon(body))
        self.records.append(body)
        self.head = body["record_hash"]
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(_canon(body) + "\n")
        return body

    @staticmethod
    def verify_records(records: list[dict]) -> tuple[bool, int | None]:
        prev = GENESIS
        for i, rec in enumerate(records):
            body = {k: v for k, v in rec.items() if k != "record_hash"}
            if rec.get("prev_hash") != prev or rec.get("seq") != i or _sha(_canon(body)) != rec.get("record_hash"):
                return False, i
            prev = rec["record_hash"]
        return True, None

    def verify(self) -> tuple[bool, int | None]:
        if self.path and self.path.exists():
            recs = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
            return self.verify_records(recs)
        return self.verify_records(self.records)
