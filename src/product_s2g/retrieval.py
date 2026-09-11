"""Hybrid retrieval factory over every non-empty canonical chunk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.retrieval.backbone import HybridRetrievalBackbone
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever, SentenceTransformerEncoder
from src.retrieval.reranker import CrossEncoderScorer, Reranker

from .config import ProductConfig
from .corpus import FullCorpusCatalog


def device_name() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_full_corpus_backbone(
    catalog: FullCorpusCatalog,
    config: ProductConfig,
) -> tuple[HybridRetrievalBackbone, Reranker, dict[str, Any]]:
    """Build the full-corpus hybrid backbone without role-based exclusion."""

    documents = list(catalog.documents)
    retrieval = config.retrieval
    device = device_name()
    bm25 = BM25Retriever(documents, k1=1.5, b=0.75, epsilon=0.25)
    dense = DenseRetriever(
        documents,
        encoder=SentenceTransformerEncoder(
            retrieval["dense_model"],
            revision=retrieval["dense_revision"],
            device=device,
        ),
        model_name=retrieval["dense_model"],
        model_revision=retrieval["dense_revision"],
        batch_size=32,
        query_prefix="query: ",
        passage_prefix="passage: ",
        normalize_embeddings=True,
        similarity="cosine",
        cache_path=Path(__file__).resolve().parents[2] / retrieval["cache_path"],
    )
    reranker = Reranker(
        documents,
        scorer=CrossEncoderScorer(
            retrieval["reranker_model"],
            revision=retrieval["reranker_revision"],
            device=device,
            max_length=512,
        ),
        batch_size=16,
    )
    return HybridRetrievalBackbone(
        documents,
        bm25,
        dense,
        reranker,
        presented_depth=int(retrieval["presented_depth"]),
    ), reranker, {
        "indexed_chunks": len(documents),
        "indexed_documents": len({row.document_id for row in documents}),
        "indexed_roles": list(config.indexed_roles),
        "device": device,
        "dense_model": retrieval["dense_model"],
        "dense_revision": retrieval["dense_revision"],
        "reranker_model": retrieval["reranker_model"],
        "reranker_revision": retrieval["reranker_revision"],
    }
