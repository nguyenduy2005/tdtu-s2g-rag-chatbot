"""S2G provider caller with complete two-phase attempt auditing.

Retry and scientific semantics are unchanged from the frozen Max-Output V2 and
Local-Semantic V3 policies. Only audit persistence is replaced.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable

from .openai_adapter import (
    PermanentControllerFailure,
    PermanentModelError,
    _metadata_dict,
    canonical_schema_sha256,
)
from .runtime_types import ModelCallRecord, ModelResult

from .attempt_audit import AttemptType, current_attempt_scope
from .local_semantic_policy import (
    LocalSemanticValidationPolicy,
    _VALIDATOR,
)
from .output_policy import PolicyBoundOpenAICaller


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _plain(value: Any) -> Any:
    try:
        return asdict(value)
    except TypeError:
        if isinstance(value, (dict, list, tuple, str, int, float, bool, type(None))):
            return list(value) if isinstance(value, tuple) else value
        return repr(value)


def _usage_fields(usage: Any) -> dict[str, int | None]:
    value = _metadata_dict(usage) or {}
    input_details = value.get("input_tokens_details") or {}
    output_details = value.get("output_tokens_details") or {}
    return {
        "input_tokens": value.get("input_tokens"),
        "output_tokens": value.get("output_tokens"),
        "reasoning_tokens": output_details.get("reasoning_tokens"),
        "cached_input_tokens": input_details.get("cached_tokens"),
    }


class AuditedLocalSemanticOpenAICaller(PolicyBoundOpenAICaller):
    """Persist and verify receipt before any parser or semantic validator runs."""

    def __init__(
        self,
        *args: Any,
        local_semantic_policy: LocalSemanticValidationPolicy,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.local_semantic_policy = local_semantic_policy

    def request(
        self,
        *,
        stage: str,
        prompt: str,
        template_sha256: str,
        parser: Callable[[str], Any],
        max_output_tokens: int,
        schema: dict[str, Any],
        source_text_sha256: tuple[str, ...] = (),
    ) -> ModelResult:
        effective = self.max_output_policy.resolve(stage, max_output_tokens)
        if stage not in self.local_semantic_policy.allowed_stages:
            raise ValueError(f"Controller stage outside S2G local-validation policy: {stage}")
        binding = _VALIDATOR.get()
        if binding is None or binding.stage != stage:
            raise RuntimeError(f"Missing frozen local validator for stage: {stage}")
        scope = current_attempt_scope(required=True)
        assert scope is not None
        logical = scope.begin_logical_call(stage)

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        schema_hash = canonical_schema_sha256(schema)
        parameters = self._request_parameters(stage=stage, prompt=prompt, schema=schema, max_output_tokens=effective)
        input_hash = hashlib.sha256(json.dumps(parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        completion_index = 1
        semantic_index = 1
        completion_retries = 0
        semantic_retries = 0
        total_transport_retries = 0
        total_latency = 0.0
        attempt_type = AttemptType.INITIAL
        response_attempts: list[dict[str, Any]] = []
        raw_attempts: list[str] = []

        while True:
            started = time.monotonic()
            started_at = _utc_now()
            try:
                response, transport_retries, latency, transport_failures = self._transport(
                    stage=stage, prompt=prompt, schema=schema, max_output_tokens=effective
                )
            except PermanentModelError as exc:
                failures = list(exc.audit.get("transport_failures", ()))
                for item in failures:
                    provider_index = logical.next_provider_attempt()
                    transport_index = int(item.get("transport_attempt", provider_index)) - 1
                    kind = attempt_type if transport_index == 0 else AttemptType.TRANSPORT_RETRY
                    receipt = {
                        "provider_attempt_index": provider_index,
                        "attempt_type": kind.value,
                        "requested_model": self.config.requested_model,
                        "resolved_snapshot": None,
                        "max_output_tokens": effective,
                        "input_hash": input_hash,
                        "prompt_hash": prompt_hash,
                        "schema_hash": schema_hash,
                        "provider_response_id": None,
                        "provider_status": "TRANSPORT_FAILURE",
                        "incomplete_details": None,
                        "raw_output": None,
                        "parsed_output": None,
                        "schema_validation_result": "NOT_REACHED",
                        "local_validation_result": "NOT_REACHED",
                        "validation_error": f"{item.get('error_category')}: transport failure",
                        "input_tokens": None,
                        "output_tokens": None,
                        "reasoning_tokens": None,
                        "cached_input_tokens": None,
                        "latency_seconds": time.monotonic() - started,
                        "transport_retry_index": transport_index,
                        "completion_attempt_index": completion_index,
                        "semantic_attempt_index": semantic_index,
                    }
                    path = scope.persist_receipt(logical, receipt)
                    scope.finalize(path, {})
                audit = dict(exc.audit)
                audit.update({
                    "failure_kind": "INFRASTRUCTURE_FAILURE",
                    "input_hash": input_hash,
                    "prompt_hash": prompt_hash,
                    "schema_hash": schema_hash,
                    "completion_reattempt_count": completion_retries,
                    "local_semantic_reattempt_count": semantic_retries,
                })
                raise PermanentModelError(str(exc), audit=audit) from exc

            total_latency += latency
            total_transport_retries += transport_retries
            for item in transport_failures:
                provider_index = logical.next_provider_attempt()
                transport_index = int(item.get("transport_attempt", provider_index)) - 1
                kind = attempt_type if transport_index == 0 else AttemptType.TRANSPORT_RETRY
                receipt = {
                    "provider_attempt_index": provider_index,
                    "attempt_type": kind.value,
                    "requested_model": self.config.requested_model,
                    "resolved_snapshot": None,
                    "max_output_tokens": effective,
                    "input_hash": input_hash,
                    "prompt_hash": prompt_hash,
                    "schema_hash": schema_hash,
                    "provider_response_id": None,
                    "provider_status": "TRANSPORT_FAILURE",
                    "incomplete_details": None,
                    "raw_output": None,
                    "parsed_output": None,
                    "schema_validation_result": "NOT_REACHED",
                    "local_validation_result": "NOT_REACHED",
                    "validation_error": f"{item.get('error_category')}: transport failure",
                    "input_tokens": None,
                    "output_tokens": None,
                    "reasoning_tokens": None,
                    "cached_input_tokens": None,
                    "latency_seconds": None,
                    "transport_retry_index": transport_index,
                    "completion_attempt_index": completion_index,
                    "semantic_attempt_index": semantic_index,
                }
                path = scope.persist_receipt(logical, receipt)
                scope.finalize(path, {})

            provider_index = logical.next_provider_attempt()
            transport_index = transport_retries
            kind = AttemptType.TRANSPORT_RETRY if transport_retries else attempt_type
            raw = getattr(response, "output_text", "")
            raw = raw if isinstance(raw, str) else ""
            raw_attempts.append(raw)
            status = getattr(response, "status", None)
            response_id = getattr(response, "id", None)
            resolved = getattr(response, "model", None)
            usage = _usage_fields(getattr(response, "usage", None))
            receipt = {
                "provider_attempt_index": provider_index,
                "attempt_type": kind.value,
                "requested_model": self.config.requested_model,
                "resolved_snapshot": resolved,
                "max_output_tokens": effective,
                "input_hash": input_hash,
                "prompt_hash": prompt_hash,
                "schema_hash": schema_hash,
                "provider_response_id": response_id,
                "provider_status": status,
                "incomplete_details": _metadata_dict(getattr(response, "incomplete_details", None)),
                "raw_output": raw,
                "parsed_output": None,
                "schema_validation_result": "PENDING",
                "local_validation_result": "PENDING",
                "validation_error": None,
                **usage,
                "latency_seconds": latency,
                "transport_retry_index": transport_index,
                "completion_attempt_index": completion_index,
                "semantic_attempt_index": semantic_index,
            }
            path = scope.persist_receipt(logical, receipt)
            # Durability and self-hash have been verified by persist_receipt.

            if resolved != self.config.qualified_resolved_model:
                final = scope.finalize(path, {
                    "schema_validation_result": "NOT_REACHED",
                    "local_validation_result": "NOT_REACHED",
                    "validation_error": "Resolved snapshot differs from frozen identity",
                })
                raise PermanentModelError("OpenAI resolved model differs from the qualified frozen identity", audit={"failure_kind": "SNAPSHOT_MISMATCH", "stage": stage, "attempt": final})

            if status == "incomplete":
                final = scope.finalize(path, {
                    "schema_validation_result": "NOT_REACHED",
                    "local_validation_result": "NOT_REACHED",
                    "validation_error": "Provider response incomplete",
                })
                response_attempts.append(final)
                allowed = self.completion_policy.maximum_completion_reattempts if self.completion_policy else 0
                if completion_retries < allowed:
                    completion_retries += 1
                    completion_index += 1
                    attempt_type = AttemptType.COMPLETION_REATTEMPT
                    continue
                raise PermanentControllerFailure("OpenAI controller remained incomplete after one completion re-attempt", audit={"failure_kind": "PERMANENT_CONTROLLER_FAILURE", "stage": stage, "response_attempts": response_attempts, "completion_reattempt_count": completion_retries, "local_semantic_reattempt_count": semantic_retries})

            if status not in (None, "completed"):
                final = scope.finalize(path, {
                    "schema_validation_result": "NOT_REACHED",
                    "local_validation_result": "NOT_REACHED",
                    "validation_error": f"Unsupported provider status: {status}",
                })
                raise PermanentModelError(f"Unsupported provider status: {status}", audit={"failure_kind": "INFRASTRUCTURE_FAILURE", "stage": stage, "attempt": final})

            try:
                parsed = parser(raw)
            except Exception as exc:
                final = scope.finalize(path, {
                    "schema_validation_result": "FAIL",
                    "local_validation_result": "NOT_REACHED",
                    "validation_error": f"{type(exc).__name__}: {exc}",
                })
                permanent = completion_retries > 0
                error_type = PermanentControllerFailure if permanent else PermanentModelError
                raise error_type(f"Strict structured output failed local parse validation: {type(exc).__name__}", audit={"failure_kind": "PERMANENT_CONTROLLER_FAILURE" if permanent else "PROVIDER_SCHEMA_FAILURE", "stage": stage, "response_attempts": [*response_attempts, final]}) from exc

            parsed_plain = _plain(parsed)
            scope.persist_parsed(path, parsed_plain)
            try:
                binding.validator(parsed)
            except Exception as exc:
                final = scope.finalize(path, {
                    "local_validation_result": "FAIL",
                    "validation_error": f"{type(exc).__name__}: {exc}",
                })
                response_attempts.append(final)
                if semantic_retries < self.local_semantic_policy.maximum_reattempts:
                    semantic_retries += 1
                    semantic_index += 1
                    attempt_type = AttemptType.LOCAL_SEMANTIC_REATTEMPT
                    continue
                raise PermanentControllerFailure("Controller output failed frozen local semantic validation twice", audit={"failure_kind": "PERMANENT_CONTROLLER_FAILURE", "terminal_precursor": "LOCAL_SEMANTIC_VALIDATION_FAILURE", "stage": stage, "exact_validator_error": final["validation_error"], "response_attempts": response_attempts, "completion_reattempt_count": completion_retries, "local_semantic_reattempt_count": semantic_retries}) from exc

            final = scope.finalize(path, {"local_validation_result": "PASS", "validation_error": None})
            response_attempts.append(final)
            call = ModelCallRecord(
                stage=stage,
                raw_response=raw,
                parsed_response=parsed_plain,
                response_id=response_id,
                model_name=self.config.requested_model,
                model_version=resolved,
                usage_metadata=_metadata_dict(getattr(response, "usage", None)),
                latency_seconds=total_latency,
                retry_count=total_transport_retries,
                completion_reattempt_count=completion_retries,
                schema_repair_used=False,
                prompt_sha256=prompt_hash,
                template_sha256=template_sha256,
                allowlist_validated=True,
                source_text_sha256=source_text_sha256,
                raw_response_attempts=tuple(raw_attempts),
                response_attempts=tuple(response_attempts),
            )
            return ModelResult(parsed, call)
