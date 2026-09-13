"""Pluggable retrieval adapters for judge-first S2G-RAG."""

from __future__ import annotations

from typing import Any, Protocol

from src.s2g_runtime.runtime_types import EvidenceItem
from src.retrieval.corpus import RetrievalDocument

from .models import RetrievalBatch


class S2GRetriever(Protocol):
    def retrieve(self, query: str, top_k: int, turn_index: int) -> RetrievalBatch: ...


def _evidence(document: RetrievalDocument) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=document.chunk_id,
        source_text=document.text,
        source_document_identifier=document.document_id,
        page_start=document.page_start,
        page_end=document.page_end,
        source_spans=document.source_spans,
    )


class RankedRetrieverAdapter:
    """Adapt a BM25 or dense retriever exposing ``retrieve(query, top_k)``."""

    def __init__(
        self,
        mode: str,
        backend: Any,
        documents: list[RetrievalDocument],
    ) -> None:
        if mode not in {"bm25", "dense"}:
            raise ValueError("RankedRetrieverAdapter supports only bm25 or dense")
        self.mode = mode
        self.backend = backend
        self.documents = {row.chunk_id: row for row in documents}

    def retrieve(self, query: str, top_k: int, turn_index: int) -> RetrievalBatch:
        rows = tuple(dict(row) for row in self.backend.retrieve(query, top_k=top_k))
        missing = [row.get("chunk_id") for row in rows if row.get("chunk_id") not in self.documents]
        if missing:
            raise ValueError(f"Retriever returned unknown chunk IDs: {missing[:3]}")
        evidence = {row["chunk_id"]: _evidence(self.documents[row["chunk_id"]]) for row in rows}
        return RetrievalBatch(
            mode=self.mode,
            query=query,
            turn_index=turn_index,
            items=rows,
            evidence=evidence,
            audit={self.mode: [dict(row) for row in rows]},
        )


class BackboneRetrieverAdapter:
    """Adapt the existing hybrid/RRF/reranker backbone without changing it."""

    def __init__(self, backbone: Any, *, mode: str = "hybrid_rrf_reranker") -> None:
        if mode not in {"hybrid_rrf", "hybrid_rrf_reranker"}:
            raise ValueError("Invalid hybrid retrieval mode")
        self.backbone = backbone
        self.mode = mode

    def retrieve(self, query: str, top_k: int, turn_index: int) -> RetrievalBatch:
        result = self.backbone.retrieve_round(query, turn_index)
        source_rows = result.reranked if self.mode == "hybrid_rrf_reranker" else result.candidates
        rows = tuple(dict(row) for row in source_rows[:top_k])
        evidence = {}
        for row in rows:
            chunk_id = row["chunk_id"]
            item = result.evidence.get(chunk_id)
            if item is None and hasattr(self.backbone, "documents"):
                document = self.backbone.documents.get(chunk_id)
                if document is not None:
                    item = _evidence(document)
            if item is None:
                raise ValueError(f"Hybrid backbone did not expose evidence for {chunk_id}")
            evidence[chunk_id] = item
        return RetrievalBatch(
            mode=self.mode,
            query=query,
            turn_index=turn_index,
            items=rows,
            evidence=evidence,
            audit={
                "bm25": [dict(row) for row in result.bm25],
                "dense": [dict(row) for row in result.dense],
                "candidates": [dict(row) for row in result.candidates],
                "reranked": [dict(row) for row in result.reranked],
                "presented": [dict(row) for row in rows],
                "latency_seconds": dict(result.latency_seconds or {}),
            },
        )
