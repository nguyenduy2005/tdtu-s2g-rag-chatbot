"""Hybrid retrieval backbone used by the S2G product."""

from __future__ import annotations

import time
from typing import Any

from src.retrieval.corpus import RetrievalDocument
from src.retrieval.fusion import reciprocal_rank_fusion

from src.s2g_runtime.runtime_types import BackboneRound, EvidenceItem


class HybridRetrievalBackbone:
    def __init__(
        self,
        documents: list[RetrievalDocument],
        bm25: Any,
        dense: Any,
        reranker: Any,
        *,
        presented_depth: int = 10,
    ) -> None:
        if presented_depth <= 0 or presented_depth > 50:
            raise ValueError("presented_depth must be between 1 and 50")
        self.documents = {doc.chunk_id: doc for doc in documents}
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.presented_depth = presented_depth

    def retrieve_round(self, query: str, round_index: int) -> BackboneRound:
        total_started = time.monotonic()
        stage_started = time.monotonic()
        bm25 = self.bm25.retrieve(query, top_k=100)
        bm25_latency = time.monotonic() - stage_started
        stage_started = time.monotonic()
        dense = self.dense.retrieve(query, top_k=100)
        dense_latency = time.monotonic() - stage_started
        stage_started = time.monotonic()
        candidates = reciprocal_rank_fusion(
            {"bm25": bm25, "dense": dense}, k_rrf=60, top_k=100
        )
        fusion_latency = time.monotonic() - stage_started
        for row in candidates:
            row.update(
                round=round_index,
                retrieval_query=query,
                source_document_identifier=self.documents[row["chunk_id"]].document_id,
            )
            row.setdefault("bm25_rank", None)
            row.setdefault("bm25_score", None)
            row.setdefault("dense_rank", None)
            row.setdefault("dense_score", None)
        stage_started = time.monotonic()
        reranked = self.reranker.rerank(query, candidates[:50], top_k=50)
        reranker_latency = time.monotonic() - stage_started
        presented = [dict(row) for row in reranked[: self.presented_depth]]
        evidence = {
            row["chunk_id"]: self._evidence(self.documents[row["chunk_id"]]) for row in presented
        }
        return BackboneRound(
            bm25=tuple(dict(row) for row in bm25),
            dense=tuple(dict(row) for row in dense),
            candidates=tuple(dict(row) for row in candidates),
            reranked=tuple(dict(row) for row in reranked),
            presented=tuple(presented),
            evidence=evidence,
            latency_seconds={
                "bm25": bm25_latency,
                "dense": dense_latency,
                "fusion": fusion_latency,
                "reranker": reranker_latency,
                "total": time.monotonic() - total_started,
            },
        )

    @staticmethod
    def _evidence(doc: RetrievalDocument) -> EvidenceItem:
        return EvidenceItem(
            chunk_id=doc.chunk_id,
            source_text=doc.text,
            source_document_identifier=doc.document_id,
            page_start=doc.page_start,
            page_end=doc.page_end,
            source_spans=doc.source_spans,
        )
