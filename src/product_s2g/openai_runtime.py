"""Product-local construction of the qualified S2G OpenAI caller."""

from __future__ import annotations

from pathlib import Path

from src.s2g_runtime.local_semantic_policy import LocalSemanticValidationPolicy
from src.s2g_runtime.output_policy import MaxOutputTokenPolicy
from src.s2g_runtime.reasoning_policy import (
    ControllerReasoningPolicy,
    ReasoningAuditedOpenAICaller,
)
from src.s2g_runtime.openai_adapter import (
    OpenAIProviderConfig,
    create_openai_client,
)


def build_product_s2g_caller() -> ReasoningAuditedOpenAICaller:
    """Build the same frozen caller without importing experiment executors."""

    provider = OpenAIProviderConfig.load(Path("config/openai_provider.json"))
    max_policy = MaxOutputTokenPolicy.load()
    return ReasoningAuditedOpenAICaller(
        create_openai_client(provider),
        provider,
        completion_policy=max_policy.completion_policy(),
        max_output_policy=max_policy,
        local_semantic_policy=LocalSemanticValidationPolicy.load(),
        reasoning_policy=ControllerReasoningPolicy.load(require_frozen=False),
    )
