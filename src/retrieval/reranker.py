"""Cross-encoder reranking over a fixed RRF candidate set."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .corpus import RetrievalDocument


class PairScorer(Protocol):
    def predict(self, pairs: list[tuple[str, str]], *, batch_size: int) -> np.ndarray: ...


class CrossEncoderScorer:
    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None,
        device: str,
        max_length: int,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            model_name,
            revision=revision,
            device=device,
            max_length=max_length,
        )

    def predict(self, pairs: list[tuple[str, str]], *, batch_size: int) -> np.ndarray:
        return np.asarray(
            self.model.predict(pairs, batch_size=batch_size, show_progress_bar=len(pairs) > batch_size)
        ).reshape(-1)


class Reranker:
    def __init__(
        self,
        documents: list[RetrievalDocument],
        *,
        scorer: PairScorer,
        batch_size: int,
    ) -> None:
        self.documents = {document.chunk_id: document for document in documents}
        self.scorer = scorer
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in candidates:
            chunk_id = candidate["chunk_id"]
            if chunk_id in seen:
                continue
            if chunk_id not in self.documents:
                raise ValueError(f"Reranker candidate is absent from retrieval corpus: {chunk_id}")
            seen.add(chunk_id)
            unique.append(candidate)
        pairs = [(query, self.documents[item["chunk_id"]].text) for item in unique]
        scores = self.scorer.predict(pairs, batch_size=self.batch_size)
        if len(scores) != len(unique):
            raise ValueError("Reranker scorer returned the wrong number of scores")
        rows = []
        for candidate, score in zip(unique, scores):
            row = dict(candidate)
            previous_rank = int(candidate.get("rank", len(rows) + 1))
            row["previous_rank"] = previous_rank
            row.setdefault("rrf_rank", previous_rank)
            row["reranker_score"] = float(score)
            row["score"] = float(score)
            rows.append(row)
        rows.sort(key=lambda row: (-row["reranker_score"], row["chunk_id"]))
        rows = rows[: min(top_k, len(rows))]
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
            row["reranker_rank"] = rank
        return rows
