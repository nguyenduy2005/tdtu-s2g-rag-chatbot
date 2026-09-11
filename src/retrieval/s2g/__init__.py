"""Shared S2G primitives used by the judge-first research controller."""

from .gap_mapping import build_next_query, select_gap_phrase
from .sentence_context import EvidenceContext, SentenceRecord, segment_chunk

__all__ = [
    "EvidenceContext",
    "SentenceRecord",
    "build_next_query",
    "segment_chunk",
    "select_gap_phrase",
]
