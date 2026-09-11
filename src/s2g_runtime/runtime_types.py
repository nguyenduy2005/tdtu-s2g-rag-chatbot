"""Shared retrieval and model-call records used by the S2G runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceItem:
    chunk_id: str
    source_text: str
    source_document_identifier: str
    page_start: int
    page_end: int
    source_spans: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class BackboneRound:
    bm25: tuple[dict[str, Any], ...]
    dense: tuple[dict[str, Any], ...]
    candidates: tuple[dict[str, Any], ...]
    reranked: tuple[dict[str, Any], ...]
    presented: tuple[dict[str, Any], ...]
    evidence: dict[str, EvidenceItem]


@dataclass(frozen=True)
class ModelCallRecord:
    stage: str
    raw_response: str
    parsed_response: dict[str, Any] | None
    response_id: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    usage_metadata: dict[str, Any] | None = None
    latency_seconds: float = 0.0
    retry_count: int = 0
    completion_reattempt_count: int = 0
    schema_repair_used: bool = False
    prompt_sha256: str = ""
    template_sha256: str = ""
    allowlist_validated: bool = True
    source_text_sha256: tuple[str, ...] = ()
    raw_response_attempts: tuple[str, ...] = ()
    response_attempts: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ModelResult:
    value: Any
    call: ModelCallRecord
