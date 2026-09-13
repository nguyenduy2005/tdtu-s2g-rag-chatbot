"""GT-free benchmark executors for local retrieval and the S2G product."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.product_s2g.config import ProductConfig
from src.product_s2g.corpus import FullCorpusCatalog, sha256_file
from src.product_s2g.io import atomic_json
from src.product_s2g.retrieval import build_full_corpus_backbone
from src.product_s2g.service import S2GChatService


def load_runtime_queries(path: Path) -> list[dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "s2g-runtime-queries-v1":
        raise ValueError("Unexpected runtime-query schema")
    rows = value.get("queries")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Runtime queries must be a non-empty list")
    identifiers: set[str] = set()
    for row in rows:
        if set(row) != {"id", "query"}:
            raise ValueError("Runtime receives only id and query; GT firewall FAIL")
        if not all(isinstance(row[key], str) and row[key].strip() for key in row):
            raise ValueError("Runtime query id/query must be non-empty")
        if row["id"] in identifiers:
            raise ValueError(f"Duplicate runtime query ID: {row['id']}")
        identifiers.add(row["id"])
    return rows


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [row["chunk_id"] for row in rows]


def run_b0(runtime_queries: Path, output_path: Path) -> dict[str, Any]:
    queries = load_runtime_queries(runtime_queries)
    config = ProductConfig.load()
    catalog = FullCorpusCatalog.load(config)
    backbone, _, component_info = build_full_corpus_backbone(catalog, config)
    outputs = []
    for query in queries:
        result = backbone.retrieve_round(query["query"], 0)
        outputs.append({
            "id": query["id"],
            "rankings": {
                "bm25": _ids(list(result.bm25)),
                "dense": _ids(list(result.dense)),
                "hybrid_rrf": _ids(list(result.candidates)),
                "hybrid_rrf_reranker": _ids(list(result.reranked)),
            },
            "latency_seconds": dict(result.latency_seconds or {}),
        })
    artifact = {
        "schema_version": "s2g-retrieval-rankings-v1",
        "method_id": "b0_full_corpus_hybrid",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runtime_queries_sha256": sha256_file(runtime_queries),
        "corpus_sha256": catalog.canonical_chunks_sha256,
        "config_sha256": sha256_file(Path("config/s2g_product.json")),
        "environment": {"python": platform.python_version(), **component_info},
        "queries": outputs,
    }
    atomic_json(output_path, artifact)
    return artifact


def run_s2g(runtime_queries: Path, output_root: Path, *, limit: int | None = None) -> dict[str, Any]:
    queries = load_runtime_queries(runtime_queries)
    if limit is not None:
        queries = queries[:limit]
    config = replace(ProductConfig.load(), output_root=output_root)
    service = S2GChatService.build(config)
    completed = skipped = 0
    for query in queries:
        result_path = output_root / "runtime" / query["id"] / "result.json"
        if result_path.is_file():
            value = json.loads(result_path.read_text(encoding="utf-8"))
            core = value.get("s2g_core") or {}
            if core.get("query_id") != query["id"] or core.get("closed") is not True:
                raise ValueError(f"Existing artifact is invalid/incomplete: {query['id']}")
            skipped += 1
            print(f"[S2G] {query['id']} SKIP valid completed artifact", flush=True)
            continue
        print(f"[S2G] {query['id']} START", flush=True)
        service.chat(query["query"], request_id=query["id"])
        completed += 1
        print(f"[S2G] {query['id']} COMPLETE", flush=True)
    return {
        "status": "PASS",
        "attempted_queries": len(queries),
        "newly_completed": completed,
        "skipped_valid": skipped,
        "output_root": str(output_root),
        "runtime_queries_sha256": hashlib.sha256(runtime_queries.read_bytes()).hexdigest(),
    }
