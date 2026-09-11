"""Deterministic BM25 retrieval."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rank_bm25 import BM25Okapi

from .corpus import RetrievalDocument
from .tokenization import unicode_words_nfc_casefold


class BM25Retriever:
    def __init__(
        self,
        documents: list[RetrievalDocument],
        *,
        tokenizer: Callable[[str], list[str]] = unicode_words_nfc_casefold,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        if not documents:
            raise ValueError("BM25 requires a non-empty corpus")
        self.documents = documents
        self.tokenizer = tokenizer
        tokenized = [tokenizer(document.text) for document in documents]
        self.index = BM25Okapi(tokenized, k1=k1, b=b, epsilon=epsilon)

    def retrieve(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores = self.index.get_scores(self.tokenizer(query))
        order = sorted(
            range(len(self.documents)),
            key=lambda index: (-float(scores[index]), self.documents[index].chunk_id),
        )[: min(top_k, len(self.documents))]
        return [
            {
                "chunk_id": self.documents[index].chunk_id,
                "score": float(scores[index]),
                "bm25_score": float(scores[index]),
                "rank": rank,
                "bm25_rank": rank,
            }
            for rank, index in enumerate(order, start=1)
        ]
