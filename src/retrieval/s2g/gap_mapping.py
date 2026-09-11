"""Frozen deterministic S2G structured-gap to retrieval-query mapping."""

from __future__ import annotations

import unicodedata

from .schemas import GapItem


def normalize_space(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def normalize_for_duplicate(value: str) -> str:
    return normalize_space(value).casefold()


def gap_phrase(gap: GapItem) -> str:
    target = normalize_space(gap.target or "")
    slot = normalize_space(gap.slot or "")
    if target and slot:
        return f"{target} {slot}"
    return normalize_space(gap.description)


def select_gap_phrase(gaps: tuple[GapItem, ...]) -> str | None:
    """Select K=1: the first valid phrase in judge output order."""
    for gap in gaps:
        phrase = gap_phrase(gap)
        if phrase:
            return phrase
    return None


def build_next_query(q_0: str, gaps: tuple[GapItem, ...]) -> str | None:
    phrase = select_gap_phrase(gaps)
    return f"{q_0} {phrase}" if phrase else None
