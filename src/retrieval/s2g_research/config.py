"""Strict configuration for the judge-first S2G research controller."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "config/s2g_research.json"
RETRIEVAL_MODES = {"bm25", "dense", "hybrid_rrf", "hybrid_rrf_reranker"}


@dataclass(frozen=True)
class S2GResearchConfig:
    method_id: str
    corpus_path: Path
    retrieval_mode: str
    top_k: int
    max_turns: int
    max_gaps_for_query: int
    sentence_limit: int
    parent_limit: int
    retrieval: dict[str, Any]

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG) -> "S2GResearchConfig":
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != "s2g-research-controller-v1":
            raise ValueError("Unexpected S2G research configuration schema")
        retrieval = dict(value.get("retrieval") or {})
        controller = dict(value.get("controller") or {})
        mode = retrieval.get("mode")
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unsupported retrieval mode: {mode!r}")
        if retrieval.get("top_k") != 6:
            raise ValueError("Research S2G evidence presentation requires top_k=6")
        if controller.get("max_turns") != 4:
            raise ValueError("Research S2G controller requires max_turns=4")
        if controller.get("max_gaps_for_query") != 1:
            raise ValueError("Research S2G query construction requires K=1 gap")
        required_true = (
            "judge_first",
            "pointer_only_evidence",
            "answer_from_accumulated_evidence_only",
        )
        if any(controller.get(field) is not True for field in required_true):
            raise ValueError("Required S2G architecture guards are disabled")
        sentence_limit = controller.get("sentence_limit")
        parent_limit = controller.get("parent_limit")
        if not isinstance(sentence_limit, int) or sentence_limit < 1:
            raise ValueError("sentence_limit must be a positive integer")
        if not isinstance(parent_limit, int) or parent_limit < 1:
            raise ValueError("parent_limit must be a positive integer")
        corpus = ROOT / str(value["canonical_corpus"])
        if not corpus.is_file():
            raise FileNotFoundError(corpus)
        return cls(
            method_id=str(value["method_id"]),
            corpus_path=corpus,
            retrieval_mode=str(mode),
            top_k=int(retrieval["top_k"]),
            max_turns=int(controller["max_turns"]),
            max_gaps_for_query=int(controller["max_gaps_for_query"]),
            sentence_limit=sentence_limit,
            parent_limit=parent_limit,
            retrieval=retrieval,
        )
