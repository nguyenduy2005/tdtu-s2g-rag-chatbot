"""Prompts for the judge-first S2G research interfaces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from src.retrieval.s2g.schemas import GapItem
from src.retrieval.s2g.sentence_context import SentenceRecord


JUDGE_PROMPT = """You are the S2G sufficiency and knowledge-gap Judge.
Use ONLY the supplied Evidence Context to decide whether the ORIGINAL QUESTION can be answered reliably.
Never use parametric knowledge to fill missing evidence. With empty or incomplete evidence, sufficient must be false.
If sufficient=true, gap_items must be []. Otherwise return 1-3 precise gap_items and do not answer the question.
Do not invent sources, identifiers, regulatory hierarchy, or hidden labels. Return only strict JSON.

INPUT_JSON:
{{INPUT_JSON}}
"""

EXTRACTOR_PROMPT = """You are the sentence-level Evidence Extractor in S2G-RAG.
Select at most 10 evidence_global_ids from the CURRENT SENTENCE POOL that directly support the ORIGINAL QUESTION or fill the ACTIVE GAP ITEMS.
Return pointers only. Never rewrite, summarize, correct, infer, reorder, or generate evidence text.
Prefer explicit self-contained evidence; add adjacent context only when required for meaning. Return only strict JSON.

INPUT_JSON:
{{INPUT_JSON}}
"""


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


PROMPT_SHA256 = {"judge": _sha(JUDGE_PROMPT), "extractor": _sha(EXTRACTOR_PROMPT)}


def build_judge_prompt(question: str, context: tuple[SentenceRecord, ...]):
    evidence = [
        {"evidence_global_id": row.sentence_id, "parent_chunk_id": row.parent_chunk_id, "text": row.text}
        for row in context
    ]
    payload = {"original_question": question, "evidence_context": evidence}
    return JUDGE_PROMPT.replace("{{INPUT_JSON}}", _compact(payload)), tuple(_sha(row.text) for row in context)


def build_extractor_prompt(
    question: str,
    gaps: tuple[GapItem, ...],
    sentence_pool: tuple[SentenceRecord, ...],
):
    payload = {
        "original_question": question,
        "active_gap_items": [asdict(item) for item in gaps],
        "current_sentence_pool": [
            {"evidence_global_id": row.sentence_id, "parent_chunk_id": row.parent_chunk_id, "text": row.text}
            for row in sentence_pool
        ],
    }
    return EXTRACTOR_PROMPT.replace("{{INPUT_JSON}}", _compact(payload)), tuple(
        _sha(row.text) for row in sentence_pool
    )
