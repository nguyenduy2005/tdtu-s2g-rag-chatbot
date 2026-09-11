"""Runtime-only models for the judge-first S2G controller.

No benchmark labels or ground-truth fields are permitted here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from src.s2g_runtime.runtime_types import EvidenceItem


class S2GResearchStopReason(str, Enum):
    SUFFICIENT = "STOP_SUFFICIENT"
    MAX_TURNS = "STOP_MAX_TURNS"
    NO_VALID_NEW_QUERY = "STOP_NO_VALID_NEW_QUERY"
    EMPTY_RETRIEVAL = "STOP_EMPTY_RETRIEVAL"
    NO_NEW_EVIDENCE = "STOP_NO_NEW_EVIDENCE"
    ERROR = "STOP_ERROR"


@dataclass(frozen=True)
class RetrievalBatch:
    mode: str
    query: str
    turn_index: int
    items: tuple[dict[str, Any], ...]
    evidence: dict[str, EvidenceItem]
    audit: dict[str, Any]


@dataclass
class S2GResearchResult:
    method_id: str
    query_id: str
    original_question: str
    query_history: list[str]
    turns: list[dict[str, Any]]
    evidence_context: list[dict[str, Any]]
    final_parent_chunk_ids: list[str]
    stop_reason: S2GResearchStopReason
    closed: bool
    budget: dict[str, int]
    answer: Any = None
    error: dict[str, Any] | None = None

    def core_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["stop_reason"] = self.stop_reason.value
        return value

    def canonical_sha256(self) -> str:
        raw = json.dumps(
            self.core_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) + "\n"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
