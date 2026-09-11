"""OpenAI Responses API transport for S2G controller calls."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from .completion_policy import ControllerIncompletePolicy
from .runtime_types import ModelCallRecord, ModelResult


class MissingAPIKeyError(RuntimeError):
    pass


class PermanentModelError(RuntimeError):
    def __init__(self, message: str, *, audit: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.audit = audit or {}


class PermanentControllerFailure(PermanentModelError):
    """The single allowed completion re-attempt did not yield valid output."""


def canonical_schema_sha256(schema: dict[str, Any]) -> str:
    encoded = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class OpenAIProviderConfig:
    requested_model: str = "gpt-5-nano"
    qualified_resolved_model: str = "gpt-5-nano-2025-08-07"
    api_key_environment_variable: str = "OPENAI_API_KEY"
    request_timeout_seconds: float = 120.0
    maximum_transport_retries: int = 2
    transport_backoff_seconds: tuple[float, ...] = (1.0, 2.0)
    store: bool = False
    truncation: str = "disabled"

    @classmethod
    def load(cls, path: Path) -> "OpenAIProviderConfig":
        value = json.loads(path.read_text(encoding="utf-8"))
        provider = value["provider"]
        if value.get("schema_version") != "s2g-openai-provider-v1":
            raise ValueError("Unexpected S2G OpenAI provider configuration schema")
        if provider.get("name") != "OpenAI" or provider.get("api_interface") != "Responses API":
            raise ValueError("S2G OpenAI runtime must use the Responses API")
        if provider.get("api_method") != "client.responses.create":
            raise ValueError("Unexpected OpenAI API method")
        if provider.get("sdk") != f"openai=={version('openai')}":
            raise ValueError("Installed OpenAI SDK differs from candidate config")
        if provider.get("requested_model") != "gpt-5-nano":
            raise ValueError("Unexpected OpenAI requested model")
        if provider.get("qualified_resolved_model") != "gpt-5-nano-2025-08-07":
            raise ValueError("Unexpected qualified OpenAI resolved model")
        if provider.get("api_key_environment_variable") != "OPENAI_API_KEY":
            raise ValueError("OpenAI API key must come only from OPENAI_API_KEY")
        structured = provider.get("structured_outputs") or {}
        if structured != {"type": "json_schema", "strict": True, "schema_repair_requests": 0}:
            raise ValueError("S2G OpenAI runtime requires strict Structured Outputs with no repair request")
        return cls(
            requested_model=provider["requested_model"],
            qualified_resolved_model=provider["qualified_resolved_model"],
            api_key_environment_variable=provider["api_key_environment_variable"],
            request_timeout_seconds=float(provider["request_timeout_seconds"]),
            maximum_transport_retries=int(provider["maximum_transport_retries"]),
            transport_backoff_seconds=tuple(float(item) for item in provider["transport_backoff_seconds"]),
            store=bool(provider["store"]),
            truncation=str(provider["truncation"]),
        )


def create_openai_client(config: OpenAIProviderConfig, *, environment: dict[str, str] | None = None) -> OpenAI:
    source = os.environ if environment is None else environment
    api_key = source.get(config.api_key_environment_variable, "")
    if not api_key.strip():
        raise MissingAPIKeyError(f"Missing required environment variable: {config.api_key_environment_variable}")
    return OpenAI(
        api_key=api_key,
        timeout=config.request_timeout_seconds,
        max_retries=0,
    )


def _status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def is_retryable_transport_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    if type(exc).__name__ in {"APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError"}:
        return True
    status = _status_code(exc)
    return status in {408, 409, 429, 500, 502, 503, 504}


def _metadata_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return dict(value)
    return {"value": str(value)}


class OpenAICaller:
    """One logged semantic-call adapter over ``client.responses.create``."""

    def __init__(
        self,
        client: Any,
        config: OpenAIProviderConfig,
        *,
        sleeper: Callable[[float], None] = time.sleep,
        completion_policy: ControllerIncompletePolicy | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.sleeper = sleeper
        self.completion_policy = completion_policy

    def _request_parameters(
        self, *, stage: str, prompt: str, schema: dict[str, Any], max_output_tokens: int
    ) -> dict[str, Any]:
        return {
            "model": self.config.requested_model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
            "store": self.config.store,
            "truncation": self.config.truncation,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": f"s2g_{stage}_response",
                    "strict": True,
                    "schema": schema,
                }
            },
        }

    def _transport(
        self,
        *,
        stage: str,
        prompt: str,
        schema: dict[str, Any],
        max_output_tokens: int,
    ) -> tuple[Any, int, float, list[dict[str, Any]]]:
        started = time.monotonic()
        parameters = self._request_parameters(
            stage=stage, prompt=prompt, schema=schema, max_output_tokens=max_output_tokens
        )
        failures: list[dict[str, Any]] = []
        for attempt in range(self.config.maximum_transport_retries + 1):
            try:
                response = self.client.responses.create(**parameters)
                return response, attempt, time.monotonic() - started, failures
            except Exception as exc:
                failure = {
                    "transport_attempt": attempt + 1,
                    "error_category": type(exc).__name__,
                    "status_code": _status_code(exc),
                    "retryable": is_retryable_transport_error(exc),
                }
                failures.append(failure)
                if attempt >= self.config.maximum_transport_retries or not failure["retryable"]:
                    raise PermanentModelError(
                        f"OpenAI transport failure: {type(exc).__name__}",
                        audit={
                            "stage": stage,
                            "transport_attempts": attempt + 1,
                            "retry_count": attempt,
                            "status_code": failure["status_code"],
                            "transport_failures": failures,
                            "latency_seconds": time.monotonic() - started,
                        },
                    ) from exc
                self.sleeper(self.config.transport_backoff_seconds[attempt])
        raise AssertionError("unreachable")

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
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        request_parameters = self._request_parameters(
            stage=stage, prompt=prompt, schema=schema, max_output_tokens=max_output_tokens
        )
        request_sha256 = hashlib.sha256(
            json.dumps(request_parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        historical = self.completion_policy.historical_attempts(stage) if self.completion_policy else ()
        attempts: list[dict[str, Any]] = [dict(item) for item in historical]
        audit_attempts: list[dict[str, Any]] = [dict(item) for item in historical]
        raw_attempts: list[str] = [str(item.get("raw_response", "")) for item in historical]
        total_latency = 0.0
        total_transport_retries = 0
        response = None
        raw = ""
        parsed = None

        while True:
            try:
                response, retries, latency, failures = self._transport(
                    stage=stage,
                    prompt=prompt,
                    schema=schema,
                    max_output_tokens=max_output_tokens,
                )
            except PermanentModelError as exc:
                audit = dict(exc.audit)
                audit.update({
                    "failure_kind": "INFRASTRUCTURE_FAILURE",
                    "request_sha256": request_sha256,
                    "prompt_sha256": prompt_sha256,
                    "max_output_tokens": max_output_tokens,
                    "completion_response_attempts": attempts,
                    "response_attempts": audit_attempts,
                    "completion_reattempt_count": max(0, len(attempts) - 1),
                })
                raise PermanentModelError(str(exc), audit=audit) from exc

            total_latency += latency
            total_transport_retries += retries
            audit_attempts.extend({"attempt_type": "transport_failure", **item} for item in failures)
            raw = getattr(response, "output_text", "")
            raw = raw if isinstance(raw, str) else ""
            raw_attempts.append(raw)
            resolved_model = getattr(response, "model", None)
            response_id = getattr(response, "id", None)
            status = getattr(response, "status", None)
            attempt_record = {
                "completion_attempt": len(attempts) + 1,
                "attempt_source": "current_execution",
                "transport_attempts": retries + 1,
                "transport_retry_count": retries,
                "transport_failures": failures,
                "response_id": response_id,
                "requested_model": self.config.requested_model,
                "resolved_model": resolved_model,
                "response_status": status,
                "incomplete_details": _metadata_dict(getattr(response, "incomplete_details", None)),
                "usage_metadata": _metadata_dict(getattr(response, "usage", None)),
                "raw_response": raw,
                "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "request_sha256": request_sha256,
                "prompt_sha256": prompt_sha256,
                "schema_sha256": canonical_schema_sha256(schema),
                "max_output_tokens": max_output_tokens,
                "api_interface": "Responses API",
            }
            attempts.append(attempt_record)
            audit_attempts.append(attempt_record)

            if resolved_model != self.config.qualified_resolved_model:
                raise PermanentModelError(
                    "OpenAI resolved model differs from the qualified frozen identity",
                    audit={
                        "failure_kind": "SNAPSHOT_MISMATCH",
                        "stage": stage,
                        "response_id": response_id,
                        "requested_model": self.config.requested_model,
                        "resolved_model": resolved_model,
                        "qualified_resolved_model": self.config.qualified_resolved_model,
                        "response_status": status,
                        "retry_count": total_transport_retries,
                        "completion_reattempt_count": max(0, len(attempts) - 1),
                        "schema_repair_used": False,
                        "completion_response_attempts": attempts,
                        "response_attempts": audit_attempts,
                    },
                )

            if status == "incomplete":
                allowed = self.completion_policy.maximum_completion_reattempts if self.completion_policy else 0
                if len(attempts) <= allowed:
                    continue
                raise PermanentControllerFailure(
                    "OpenAI controller remained incomplete after the single allowed completion re-attempt",
                    audit={
                        "failure_kind": "PERMANENT_CONTROLLER_FAILURE",
                        "stage": stage,
                        "response_id": response_id,
                        "requested_model": self.config.requested_model,
                        "resolved_model": resolved_model,
                        "response_status": status,
                        "incomplete_details": attempt_record["incomplete_details"],
                        "retry_count": total_transport_retries,
                        "completion_reattempt_count": max(0, len(attempts) - 1),
                        "schema_repair_used": False,
                        "schema_sha256": canonical_schema_sha256(schema),
                        "prompt_sha256": prompt_sha256,
                        "request_sha256": request_sha256,
                        "max_output_tokens": max_output_tokens,
                        "completion_response_attempts": attempts,
                        "response_attempts": audit_attempts,
                    },
                )

            if status not in (None, "completed"):
                raise PermanentModelError(
                    f"OpenAI response ended with unsupported status: {status}",
                    audit={
                        "failure_kind": "INFRASTRUCTURE_FAILURE",
                        "stage": stage,
                        "response_status": status,
                        "retry_count": total_transport_retries,
                        "completion_reattempt_count": max(0, len(attempts) - 1),
                        "completion_response_attempts": attempts,
                        "response_attempts": audit_attempts,
                    },
                )

            try:
                parsed = parser(raw)
            except Exception as exc:
                error_class = PermanentControllerFailure if len(attempts) > 1 else PermanentModelError
                failure_kind = "PERMANENT_CONTROLLER_FAILURE" if len(attempts) > 1 else "CONTROLLER_SCHEMA_FAILURE"
                raise error_class(
                    f"OpenAI strict structured output failed local validation: {type(exc).__name__}",
                    audit={
                        "failure_kind": failure_kind,
                        "stage": stage,
                        "response_id": response_id,
                        "requested_model": self.config.requested_model,
                        "resolved_model": resolved_model,
                        "response_status": status,
                        "retry_count": total_transport_retries,
                        "completion_reattempt_count": max(0, len(attempts) - 1),
                        "schema_repair_used": False,
                        "schema_sha256": canonical_schema_sha256(schema),
                        "prompt_sha256": prompt_sha256,
                        "request_sha256": request_sha256,
                        "max_output_tokens": max_output_tokens,
                        "completion_response_attempts": attempts,
                        "response_attempts": audit_attempts,
                    },
                ) from exc
            break

        assert response is not None and parsed is not None
        resolved_model = getattr(response, "model", None)
        response_id = getattr(response, "id", None)
        call = ModelCallRecord(
            stage=stage,
            raw_response=raw,
            parsed_response=asdict(parsed),
            response_id=response_id,
            model_name=self.config.requested_model,
            model_version=resolved_model,
            usage_metadata=_metadata_dict(getattr(response, "usage", None)),
            latency_seconds=total_latency,
            retry_count=total_transport_retries,
            completion_reattempt_count=max(0, len(attempts) - 1),
            schema_repair_used=False,
            prompt_sha256=prompt_sha256,
            template_sha256=template_sha256,
            allowlist_validated=True,
            source_text_sha256=source_text_sha256,
            raw_response_attempts=tuple(raw_attempts),
            response_attempts=tuple(audit_attempts),
        )
        return ModelResult(parsed, call)
