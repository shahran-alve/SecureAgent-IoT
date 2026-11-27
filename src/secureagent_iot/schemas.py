"""Strict typed schemas. Every model/tool output is parsed through these
before it can reach the policy engine."""
from __future__ import annotations

import re
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

ACTION_TYPES: tuple[str, ...] = (
    "notify",
    "request_review",
    "restrict_egress",
    "release_restriction",
)
ActionType = Literal["notify", "request_review", "restrict_egress", "release_restriction"]
Criticality = Literal["low", "medium", "high"]

DEVICE_ID_RE = re.compile(r"^[a-z]+_[0-9]{2}$")
MAX_SCHEMA_DURATION = 86_400  # type sanity bound only; the policy enforces the real cap


def _check_device_id(v: str) -> str:
    if not DEVICE_ID_RE.match(v):
        raise ValueError(f"device_id '{v}' is not a registered-synthetic-format id")
    return v


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class TelemetryEvent(Strict):
    event_id: str = Field(min_length=1)
    device_id: str
    timestamp: float
    features: dict[str, float]
    untrusted_text: str = ""
    source: str = Field(min_length=1)

    _dev = field_validator("device_id")(_check_device_id)


class Alert(Strict):
    alert_id: str = Field(min_length=1)
    device_id: str
    attack_family: str = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0)
    evidence_refs: list[str]

    _dev = field_validator("device_id")(_check_device_id)


class Asset(Strict):
    device_id: str
    criticality: Criticality
    allowed_actions: list[ActionType]
    owner: str = Field(min_length=1)
    is_online: bool

    _dev = field_validator("device_id")(_check_device_id)


class ProposedAction(Strict):
    incident_id: str = Field(min_length=1)
    device_id: str
    action_type: ActionType
    duration_seconds: int = Field(ge=0, le=MAX_SCHEMA_DURATION)
    evidence_refs: list[str] = Field(min_length=1)
    reason: str
    agent_id: str = Field(min_length=1)

    _dev = field_validator("device_id")(_check_device_id)

    @field_validator("evidence_refs")
    @classmethod
    def _non_empty_refs(cls, v: list[str]) -> list[str]:
        if any((not isinstance(x, str)) or not x.strip() for x in v):
            raise ValueError("evidence_refs must be non-empty strings")
        return v


class PolicyDecision(Strict):
    allowed: bool
    requires_approval: bool
    reason_code: str
    policy_version: str
    risk_score: float | None = None  # ordering aid for the review queue only


class ActionOutcome(Strict):
    executed: bool
    success: bool
    service_disruption: bool
    elapsed_ms: float
    rollback_success: bool | None = None


class ProposalRejected(Exception):
    """Raised when a raw agent/tool output fails schema or registry validation."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def parse_proposal(raw: Any, registry: Mapping[str, Asset]) -> ProposedAction:
    """Validate an untrusted raw proposal. Rejects non-dicts, extra fields,
    unknown action types, empty evidence, malformed and unregistered devices."""
    if not isinstance(raw, dict):
        raise ProposalRejected("SCHEMA_NOT_OBJECT", type(raw).__name__)
    try:
        action = ProposedAction.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(x) for x in first.get("loc", ()))
        raise ProposalRejected("SCHEMA_INVALID", f"{loc}: {first.get('type')}") from exc
    if action.device_id not in registry:
        raise ProposalRejected("UNREGISTERED_DEVICE", action.device_id)
    return action
