"""Strict configuration loader for the S2G chatbot product."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config/s2g_product.json"


@dataclass(frozen=True)
class ProductConfig:
    product_id: str
    corpus_path: Path
    manifest_path: Path
    coverage_path: Path
    raw_pdf_root: Path
    indexed_roles: tuple[str, ...]
    text_field: str
    retrieval: dict[str, Any]
    research_controller_config: Path
    answer: dict[str, Any]
    output_root: Path
    derived_manifest: Path

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG) -> "ProductConfig":
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != "s2g-full-pdf-product-v1":
            raise ValueError("Unexpected S2G product config schema")
        roles = tuple(value.get("indexed_roles", ()))
        if set(roles) != {"content", "review", "context_only"} or len(roles) != 3:
            raise ValueError("Full-PDF product must index all three canonical retrieval roles")
        coverage = value.get("coverage_policy") or {}
        required_true = {
            "index_every_nonempty_canonical_chunk",
            "preserve_source_text",
            "do_not_infer_unreadable_pages",
            "low_trust_roles_remain_searchable",
            "low_trust_citations_must_be_flagged",
        }
        if any(coverage.get(key) is not True for key in required_true):
            raise ValueError("Full-PDF coverage policy is incomplete")
        retrieval = dict(value["retrieval"])
        expected_depths = {
            "bm25_depth": 100,
            "dense_depth": 100,
            "rrf_k": 60,
            "candidate_depth": 100,
            "reranker_input_depth": 50,
            "presented_depth": 6,
        }
        if any(retrieval.get(key) != expected for key, expected in expected_depths.items()):
            raise ValueError("S2G product retrieval depths differ from the adapted method")
        answer = dict(value["answer"])
        if answer.get("model") != "gpt-5-nano" or answer.get("reasoning_effort") != "low":
            raise ValueError("Unexpected answer model policy")
        resolve = lambda item: ROOT / item
        research_controller_config = resolve(value["research_controller_config"])
        if not research_controller_config.is_file():
            raise FileNotFoundError(research_controller_config)
        return cls(
            product_id=str(value["product_id"]),
            corpus_path=resolve(value["canonical_corpus"]),
            manifest_path=resolve(value["canonical_manifest"]),
            coverage_path=resolve(value["canonical_coverage"]),
            raw_pdf_root=resolve(value["raw_pdf_root"]),
            indexed_roles=roles,
            text_field=str(value["retrieval_text_field"]),
            retrieval=retrieval,
            research_controller_config=research_controller_config,
            answer=answer,
            output_root=resolve(value["deployment_outputs"]),
            derived_manifest=resolve(value["derived_manifest"]),
        )
