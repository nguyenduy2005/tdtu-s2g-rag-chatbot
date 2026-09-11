"""Validation helpers for durable S2G research trajectories."""

from __future__ import annotations

from typing import Any

from .models import S2GResearchResult


REQUIRED_TURN_FIELDS = {
    "turn_index",
    "original_question",
    "evidence_context_before",
    "judge",
    "sufficient",
    "gap_items",
    "evidence_context_after",
}


def validate_trace(result: S2GResearchResult) -> None:
    """Fail if a completed result lacks the state needed for reproduction."""

    if not result.turns:
        raise ValueError("S2G result has no Judge turns")
    previous_after: list[dict[str, Any]] = []
    for expected_index, turn in enumerate(result.turns):
        missing = REQUIRED_TURN_FIELDS - set(turn)
        if missing:
            raise ValueError(f"Turn {expected_index} is missing trace fields: {sorted(missing)}")
        if turn["turn_index"] != expected_index:
            raise ValueError("Turn indices are not contiguous")
        if turn["evidence_context_before"] != previous_after:
            raise ValueError("Evidence Context continuity failure")
        before_ids = [row["sentence_id"] for row in turn["evidence_context_before"]]
        after_ids = [row["sentence_id"] for row in turn["evidence_context_after"]]
        if after_ids[: len(before_ids)] != before_ids:
            raise ValueError("Evidence Context is not append-only")
        if len(after_ids) != len(set(after_ids)):
            raise ValueError("Evidence Context contains duplicate sentence IDs")
        previous_after = turn["evidence_context_after"]
    if previous_after != result.evidence_context:
        raise ValueError("Final Evidence Context differs from the last turn")
    if result.closed and result.answer is None:
        raise ValueError("Closed result has no Answer Reasoner output")
