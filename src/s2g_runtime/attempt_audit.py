"""Two-phase S2G controller-attempt audit persistence.

Each provider result is first durably journaled with raw output and explicit
PENDING validation fields. The journal hash is verified before parsing or
local validation. The same artifact is then atomically finalized. If parsing
or validation raises, the finalized failed record already exists.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from src.product_s2g.io import atomic_json


AUDIT_SCHEMA_VERSION = "s2g-controller-attempt-audit-v1"
REQUIRED_AUDIT_FIELDS = (
    "schema_version", "audit_state", "experiment_id", "method", "query_id",
    "round", "controller_stage", "logical_call_index", "provider_attempt_index",
    "attempt_type", "requested_model", "resolved_snapshot", "max_output_tokens",
    "input_hash", "prompt_hash", "schema_hash", "provider_response_id",
    "provider_status", "incomplete_details", "raw_output", "parsed_output",
    "schema_validation_result", "local_validation_result", "validation_error",
    "input_tokens", "output_tokens", "reasoning_tokens", "cached_input_tokens",
    "latency_seconds", "transport_retry_index", "completion_attempt_index",
    "semantic_attempt_index", "timestamp", "artifact_hash",
)


class AttemptType(str, Enum):
    INITIAL = "INITIAL"
    TRANSPORT_RETRY = "TRANSPORT_RETRY"
    COMPLETION_REATTEMPT = "COMPLETION_REATTEMPT"
    LOCAL_SEMANTIC_REATTEMPT = "LOCAL_SEMANTIC_REATTEMPT"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash_without_self(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "artifact_hash"}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _assert_secret_safe(value: dict[str, Any]) -> None:
    prohibited = {"api_key", "openai_api_key", "authorization", "authorization_header", "request_headers", "headers"}

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if str(key).casefold() in prohibited:
                    raise ValueError("Attempt audit contains a prohibited secret-bearing field")
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    secret = os.environ.get("OPENAI_API_KEY", "")
    if secret and secret in json.dumps(value, ensure_ascii=False):
        raise ValueError("Attempt audit contains OPENAI_API_KEY material")


def _validate_shape(value: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_AUDIT_FIELDS if field not in value]
    if missing:
        raise ValueError(f"Attempt audit missing required fields: {missing}")
    if value["schema_version"] != AUDIT_SCHEMA_VERSION:
        raise ValueError("Unexpected attempt-audit schema version")
    if value["attempt_type"] not in {item.value for item in AttemptType}:
        raise ValueError("Invalid attempt_type")
    for field in ("logical_call_index", "provider_attempt_index", "transport_retry_index", "completion_attempt_index", "semantic_attempt_index"):
        if not isinstance(value[field], int) or value[field] < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    _assert_secret_safe(value)


def verify_attempt_artifact(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _validate_shape(value)
    if value["artifact_hash"] != _hash_without_self(value):
        raise ValueError(f"Attempt audit hash mismatch: {path}")
    return value


@dataclass
class LogicalCall:
    scope: "AttemptAuditScope"
    controller_stage: str
    round_index: int
    logical_call_index: int
    provider_attempt_index: int = 0

    def next_provider_attempt(self) -> int:
        self.provider_attempt_index += 1
        return self.provider_attempt_index


@dataclass
class AttemptAuditScope:
    root: Path
    experiment_id: str
    method: str
    query_id: str
    logical_call_count: int = 0
    stage_invocations: dict[str, int] = field(default_factory=dict)

    def begin_logical_call(self, controller_stage: str) -> LogicalCall:
        self.logical_call_count += 1
        round_index = self.stage_invocations.get(controller_stage, 0)
        self.stage_invocations[controller_stage] = round_index + 1
        return LogicalCall(self, controller_stage, round_index, self.logical_call_count)

    def path_for(self, call: LogicalCall, provider_attempt_index: int) -> Path:
        stage = call.controller_stage.replace("/", "_")
        return self.root / self.method / self.query_id / f"lc{call.logical_call_index:03d}_pa{provider_attempt_index:03d}_{stage}.json"

    def persist_receipt(self, call: LogicalCall, record: dict[str, Any]) -> Path:
        value = self._complete_common(call, record, audit_state="RECEIVED")
        path = self.path_for(call, value["provider_attempt_index"])
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing provider attempt: {path}")
        self._write_and_verify(path, value)
        return path

    def finalize(self, path: Path, updates: dict[str, Any]) -> dict[str, Any]:
        value = verify_attempt_artifact(path)
        if value["audit_state"] not in {"RECEIVED", "PARSED"}:
            raise ValueError("Only a received or parsed attempt can be finalized")
        value.update(updates)
        value["audit_state"] = "FINALIZED"
        value["timestamp"] = _utc_now()
        self._write_and_verify(path, value)
        return verify_attempt_artifact(path)

    def persist_parsed(self, path: Path, parsed_output: Any) -> dict[str, Any]:
        value = verify_attempt_artifact(path)
        if value["audit_state"] != "RECEIVED":
            raise ValueError("Only a received attempt can enter PARSED state")
        value.update({
            "audit_state": "PARSED",
            "parsed_output": parsed_output,
            "schema_validation_result": "PASS",
            "timestamp": _utc_now(),
        })
        self._write_and_verify(path, value)
        return verify_attempt_artifact(path)

    def _complete_common(self, call: LogicalCall, record: dict[str, Any], *, audit_state: str) -> dict[str, Any]:
        value = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "audit_state": audit_state,
            "experiment_id": self.experiment_id,
            "method": self.method,
            "query_id": self.query_id,
            "round": call.round_index,
            "controller_stage": call.controller_stage,
            "logical_call_index": call.logical_call_index,
            "timestamp": _utc_now(),
            **record,
        }
        value.setdefault("resolved_snapshot", None)
        value.setdefault("provider_response_id", None)
        value.setdefault("provider_status", None)
        value.setdefault("incomplete_details", None)
        value.setdefault("raw_output", None)
        value.setdefault("parsed_output", None)
        value.setdefault("schema_validation_result", "PENDING")
        value.setdefault("local_validation_result", "PENDING")
        value.setdefault("validation_error", None)
        value.setdefault("input_tokens", None)
        value.setdefault("output_tokens", None)
        value.setdefault("reasoning_tokens", None)
        value.setdefault("cached_input_tokens", None)
        value.setdefault("latency_seconds", None)
        value["artifact_hash"] = None
        return value

    @staticmethod
    def _write_and_verify(path: Path, value: dict[str, Any]) -> None:
        _validate_shape(value)
        value["artifact_hash"] = _hash_without_self(value)
        atomic_json(path, value)
        verify_attempt_artifact(path)


_SCOPE: ContextVar[AttemptAuditScope | None] = ContextVar("s2g_attempt_audit", default=None)


@contextmanager
def controller_attempt_audit_scope(root: Path, experiment_id: str, method: str, query_id: str) -> Iterator[AttemptAuditScope]:
    scope = AttemptAuditScope(Path(root), experiment_id, method, query_id)
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        _SCOPE.reset(token)


def current_attempt_scope(*, required: bool = False) -> AttemptAuditScope | None:
    scope = _SCOPE.get()
    if required and scope is None:
        raise RuntimeError("S2G controller-attempt audit scope is required")
    return scope
