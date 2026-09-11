"""S2G-only local semantic validation wrappers for the product runtime."""

from __future__ import annotations

from typing import Any

from src.s2g_runtime.local_semantic_policy import (
    LocalSemanticValidationError,
    local_semantic_validator,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LocalSemanticValidationError(message)


class S2GJudgeValidationWrapper:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate

    def judge(self, q_0: str, context: tuple[Any, ...]):
        def validate(decision: Any) -> None:
            _require(
                not decision.sufficient or bool(context),
                "sufficient=true requires non-empty evidence context",
            )

        with local_semantic_validator("s2g_judge", validate):
            return self.delegate.judge(q_0, context)


class S2GSelectorValidationWrapper:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate

    def select(self, q_0: str, gaps: tuple[Any, ...], sentence_pool: tuple[Any, ...]):
        valid_ids = {item.sentence_id for item in sentence_pool}

        def validate(decision: Any) -> None:
            identifiers = tuple(decision.sentence_ids)
            _require(len(identifiers) == len(set(identifiers)), "Duplicate sentence pointer")
            _require(
                all(pointer in valid_ids for pointer in identifiers),
                "Invalid or hallucinated sentence pointer",
            )

        with local_semantic_validator("s2g_sentence_selector", validate):
            return self.delegate.select(q_0, gaps, sentence_pool)
