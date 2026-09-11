"""Qualified reasoning policy for the S2G controller.

The policy changes only the provider request parameter ``reasoning.effort``.
It does not inspect benchmark records, retrieval results, or ground truth.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .local_semantic_audit import AuditedLocalSemanticOpenAICaller
from .output_policy import CONTROLLER_STAGES, MaxOutputTokenPolicy


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "config/s2g_reasoning_policy.json"
EXPERIMENT_ID = "s2g_full_pdf_product_v1"


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ControllerReasoningPolicy:
    policy_version: str
    experiment_id: str
    reasoning_effort: str
    max_output_tokens: int
    allowed_stages: tuple[str, ...]
    maximum_completion_reattempts: int
    maximum_local_semantic_reattempts: int

    @classmethod
    def load(
        cls,
        path: Path = POLICY_PATH,
        *,
        require_frozen: bool = True,
    ) -> "ControllerReasoningPolicy":
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != "s2g-reasoning-policy-v1":
            raise ValueError("Unexpected S2G reasoning policy schema")
        if value.get("status") != "ACTIVE":
            raise ValueError("S2G reasoning policy is not active")
        if value.get("created_without_ground_truth") is not True or value.get("created_without_benchmark_execution") is not True:
            raise ValueError("S2G policy must be benchmark/ground-truth clean")
        if value.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError("Unexpected S2G product identity")
        if value.get("requested_model") != "gpt-5-nano" or value.get("required_snapshot") != "gpt-5-nano-2025-08-07":
            raise ValueError("S2G model identity differs from the qualified controller")
        if value.get("reasoning_effort") != "low":
            raise ValueError("S2G requires reasoning.effort=low")
        if value.get("max_output_tokens") != 8192:
            raise ValueError("S2G requires max_output_tokens=8192")
        if value.get("maximum_completion_reattempts") != 1 or value.get("maximum_local_semantic_reattempts") != 1:
            raise ValueError("V5 retry counts differ from the frozen policies")
        stages = tuple(value.get("applies_to_stages", ()))
        if stages != CONTROLLER_STAGES:
            raise ValueError("Reasoning policy must cover both S2G controller stages")
        return cls(
            policy_version=str(value["policy_version"]),
            experiment_id=str(value["experiment_id"]),
            reasoning_effort=str(value["reasoning_effort"]),
            max_output_tokens=int(value["max_output_tokens"]),
            allowed_stages=stages,
            maximum_completion_reattempts=1,
            maximum_local_semantic_reattempts=1,
        )

    def bind_request(self, stage: str, parameters: dict[str, Any]) -> dict[str, Any]:
        if stage not in self.allowed_stages:
            raise ValueError(f"Controller stage outside S2G reasoning policy: {stage}")
        if parameters.get("max_output_tokens") != self.max_output_tokens:
            raise ValueError("S2G request does not use the 8192-token ceiling")
        if "reasoning" in parameters:
            raise ValueError("Refusing to overwrite a pre-existing reasoning parameter")
        value = dict(parameters)
        value["reasoning"] = {"effort": self.reasoning_effort}
        return value


RequestObserver = Callable[[dict[str, Any]], None]


class ReasoningAuditedOpenAICaller(AuditedLocalSemanticOpenAICaller):
    """Audited caller with the qualified S2G reasoning setting."""

    def __init__(
        self,
        *args: Any,
        reasoning_policy: ControllerReasoningPolicy,
        request_observer: RequestObserver | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reasoning_policy = reasoning_policy
        self.request_observer = request_observer

    def _request_parameters(
        self,
        *,
        stage: str,
        prompt: str,
        schema: dict[str, Any],
        max_output_tokens: int,
    ) -> dict[str, Any]:
        base = super()._request_parameters(
            stage=stage,
            prompt=prompt,
            schema=schema,
            max_output_tokens=max_output_tokens,
        )
        value = self.reasoning_policy.bind_request(stage, base)
        if self.request_observer is not None:
            self.request_observer(dict(value))
        return value
