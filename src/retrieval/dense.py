"""Dense retrieval with deterministic caching and ranking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .corpus import RetrievalDocument


class Encoder(Protocol):
    def encode(self, texts: list[str], *, batch_size: int, normalize_embeddings: bool) -> np.ndarray: ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str, *, revision: str | None, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, revision=revision, device=device)

    def encode(self, texts: list[str], *, batch_size: int, normalize_embeddings: bool) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=len(texts) > batch_size,
            ),
            dtype=np.float32,
        )


def _cache_key(
    documents: list[RetrievalDocument], model_name: str, model_revision: str | None, passage_prefix: str
) -> str:
    payload = {
        "chunk_ids": [document.chunk_id for document in documents],
        "model_name": model_name,
        "model_revision": model_revision,
        "passage_prefix": passage_prefix,
        "texts_sha256": hashlib.sha256(
            "\0".join(document.text for document in documents).encode("utf-8")
        ).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class DenseRetriever:
    def __init__(
        self,
        documents: list[RetrievalDocument],
        *,
        encoder: Encoder,
        model_name: str,
        model_revision: str | None,
        batch_size: int,
        query_prefix: str,
        passage_prefix: str,
        normalize_embeddings: bool = True,
        similarity: str = "cosine",
        cache_path: Path | None = None,
    ) -> None:
        if not documents:
            raise ValueError("Dense retrieval requires a non-empty corpus")
        if similarity != "cosine":
            raise ValueError("B0 dense retrieval currently supports only cosine similarity")
        self.documents = documents
        self.encoder = encoder
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.normalize_embeddings = normalize_embeddings
        self.cache_key = _cache_key(documents, model_name, model_revision, passage_prefix)
        self.cache_path = cache_path
        embeddings = self._load_cache(cache_path)
        if embeddings is None:
            passages = [f"{passage_prefix}{document.text}" for document in documents]
            embeddings = encoder.encode(
                passages, batch_size=batch_size, normalize_embeddings=normalize_embeddings
            )
            self._validate_embeddings(embeddings)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    embeddings=embeddings.astype(np.float32),
                    chunk_ids=np.asarray([document.chunk_id for document in documents]),
                    cache_key=np.asarray(self.cache_key),
                )
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def _validate_embeddings(self, embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2 or embeddings.shape[0] != len(self.documents):
            raise ValueError(
                f"Invalid corpus embedding shape {embeddings.shape}; expected rows={len(self.documents)}"
            )

    def _load_cache(self, path: Path | None) -> np.ndarray | None:
        if path is None or not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["cache_key"].item()) != self.cache_key:
                return None
            chunk_ids = cached["chunk_ids"].tolist()
            if chunk_ids != [document.chunk_id for document in self.documents]:
                return None
            embeddings = np.asarray(cached["embeddings"], dtype=np.float32)
        self._validate_embeddings(embeddings)
        return embeddings

    def retrieve(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_embedding = self.encoder.encode(
            [f"{self.query_prefix}{query}"],
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
        )[0]
        scores = self.embeddings @ np.asarray(query_embedding, dtype=np.float32)
        order = sorted(
            range(len(self.documents)),
            key=lambda index: (-float(scores[index]), self.documents[index].chunk_id),
        )[: min(top_k, len(self.documents))]
        return [
            {
                "chunk_id": self.documents[index].chunk_id,
                "score": float(scores[index]),
                "dense_score": float(scores[index]),
                "rank": rank,
                "dense_rank": rank,
            }
            for rank, index in enumerate(order, start=1)
        ]
