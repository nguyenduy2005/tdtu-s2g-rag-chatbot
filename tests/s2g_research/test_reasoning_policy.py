from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.s2g_runtime.attempt_audit import controller_attempt_audit_scope
from src.s2g_runtime.local_semantic_policy import LocalSemanticValidationPolicy
from src.s2g_runtime.output_policy import MaxOutputTokenPolicy
from src.s2g_runtime.reasoning_policy import (
    ControllerReasoningPolicy,
    ReasoningAuditedOpenAICaller,
)
from src.product_s2g.semantic_wrappers import S2GSelectorValidationWrapper
from src.s2g_runtime.openai_adapter import OpenAIProviderConfig
from src.retrieval.s2g.sentence_context import SentenceRecord
from src.retrieval.s2g_research.openai_adapter import OpenAIResearchEvidenceExtractor


class _Responses:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self._responses)


class _Client:
    def __init__(self, responses):
        self.responses = _Responses(responses)


def _response(status, text, identifier):
    return SimpleNamespace(
        status=status,
        output_text=text,
        id=identifier,
        model="gpt-5-nano-2025-08-07",
        usage={"input_tokens": 10, "output_tokens": 4, "output_tokens_details": {"reasoning_tokens": 1}},
        incomplete_details={"reason": "max_output_tokens"} if status == "incomplete" else None,
    )


def _caller(client):
    provider = OpenAIProviderConfig()
    max_policy = MaxOutputTokenPolicy.load()
    return ReasoningAuditedOpenAICaller(
        client,
        provider,
        sleeper=lambda _: None,
        completion_policy=max_policy.completion_policy(),
        max_output_policy=max_policy,
        local_semantic_policy=LocalSemanticValidationPolicy.load(),
        reasoning_policy=ControllerReasoningPolicy.load(require_frozen=False),
    )


def _pool():
    return (SentenceRecord("S001", "parent", "doc", 1, 1, (), 0, 10, "Synthetic evidence."),)


def test_reasoning_policy_covers_current_s2g_controller_stages():
    policy = ControllerReasoningPolicy.load(require_frozen=False)
    assert policy.reasoning_effort == "low"
    assert policy.max_output_tokens == 8192
    assert {"s2g_judge", "s2g_sentence_selector"}.issubset(policy.allowed_stages)


def test_request_parameters_add_only_low_reasoning_policy():
    caller = _caller(_Client([]))
    params = caller._request_parameters(
        stage="s2g_sentence_selector",
        prompt="synthetic",
        schema={"type": "object"},
        max_output_tokens=8192,
    )
    assert params["reasoning"] == {"effort": "low"}
    assert params["max_output_tokens"] == 8192
    assert "temperature" not in params and "top_p" not in params and "seed" not in params


def test_completion_reattempt_is_identical_for_current_evidence_extractor(tmp_path):
    client = _Client([
        _response("incomplete", "", "resp-incomplete"),
        _response("completed", '{"evidence_global_ids":["S001"]}', "resp-completed"),
    ])
    extractor = S2GSelectorValidationWrapper(OpenAIResearchEvidenceExtractor(_caller(client)))
    with controller_attempt_audit_scope(tmp_path, "synthetic-research", "s2g_research", "SYN001"):
        result = extractor.select("Which evidence is synthetic?", (), _pool())
    assert result.value.sentence_ids == ("S001",)
    assert len(client.responses.calls) == 2
    assert client.responses.calls[0] == client.responses.calls[1]
    rows = [json.loads(path.read_text()) for path in sorted(tmp_path.rglob("*.json"))]
    assert [row["provider_status"] for row in rows] == ["incomplete", "completed"]
    assert len({row["input_hash"] for row in rows}) == 1


def test_policy_rejects_unknown_stage_or_wrong_ceiling():
    policy = ControllerReasoningPolicy.load(require_frozen=False)
    with pytest.raises(ValueError, match="outside"):
        policy.bind_request("unknown", {"max_output_tokens": 8192})
    with pytest.raises(ValueError, match="8192"):
        policy.bind_request("s2g_judge", {"max_output_tokens": 4096})
