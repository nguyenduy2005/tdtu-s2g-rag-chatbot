"""Benchmark loading, validation, and GT-free runtime export."""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

from src.product_s2g.io import atomic_json


ALLOWED_QUERY_TYPES = {
    "direct_lookup",
    "paraphrase",
    "terminology_mismatch",
    "multi_evidence",
    "cross_document",
}
ALLOWED_VERIFICATION = {"agent_verified_pending_human", "human_verified"}


class BenchmarkValidationError(ValueError):
    pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def corpus_index(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.open(encoding="utf-8"), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        chunk_id = row.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise BenchmarkValidationError(f"Corpus line {line_number} has no chunk_id")
        if chunk_id in rows:
            raise BenchmarkValidationError(f"Duplicate corpus chunk_id: {chunk_id}")
        rows[chunk_id] = row
    return rows


def validate_benchmark(
    benchmark_path: Path,
    corpus_path: Path,
    *,
    require_human_verified: bool = False,
) -> dict[str, Any]:
    value = load_json(benchmark_path)
    if not isinstance(value, dict) or value.get("schema_version") != "s2g-retrieval-benchmark-v1":
        raise BenchmarkValidationError("Unexpected benchmark schema_version")
    queries = value.get("queries")
    if not isinstance(queries, list) or not queries:
        raise BenchmarkValidationError("Benchmark queries must be a non-empty list")
    chunks = corpus_index(corpus_path)
    query_ids: set[str] = set()
    ground_truth_ids: set[str] = set()
    evidence_group_count = 0
    for index, query in enumerate(queries, start=1):
        location = f"queries[{index}]"
        if not isinstance(query, dict):
            raise BenchmarkValidationError(f"{location} must be an object")
        query_id = query.get("id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise BenchmarkValidationError(f"{location}.id must be non-empty")
        if query_id in query_ids:
            raise BenchmarkValidationError(f"Duplicate query id: {query_id}")
        query_ids.add(query_id)
        if not isinstance(query.get("query"), str) or not query["query"].strip():
            raise BenchmarkValidationError(f"{query_id}.query must be non-empty")
        if query.get("query_type") not in ALLOWED_QUERY_TYPES:
            raise BenchmarkValidationError(f"{query_id}.query_type is invalid")
        if query.get("answerable") is not True:
            raise BenchmarkValidationError(f"{query_id}.answerable must be true in V1")
        verification = query.get("verification")
        if not isinstance(verification, dict) or verification.get("status") not in ALLOWED_VERIFICATION:
            raise BenchmarkValidationError(f"{query_id}.verification.status is invalid")
        if require_human_verified and verification["status"] != "human_verified":
            raise BenchmarkValidationError(f"{query_id} has not been human verified")
        groups = query.get("evidence_groups")
        if not isinstance(groups, list) or not groups:
            raise BenchmarkValidationError(f"{query_id} requires at least one evidence group")
        group_ids: set[str] = set()
        declared_documents = set(query.get("source_documents") or ())
        observed_documents: set[str] = set()
        for group in groups:
            if not isinstance(group, dict):
                raise BenchmarkValidationError(f"{query_id} evidence group must be an object")
            group_id = group.get("id")
            if not isinstance(group_id, str) or not group_id or group_id in group_ids:
                raise BenchmarkValidationError(f"{query_id} has an invalid/duplicate evidence-group id")
            group_ids.add(group_id)
            ids = group.get("chunk_ids")
            if not isinstance(ids, list) or not ids:
                raise BenchmarkValidationError(f"{query_id}/{group_id} has no chunk_ids")
            if len(ids) != len(set(ids)) or any(not isinstance(item, str) or not item for item in ids):
                raise BenchmarkValidationError(f"{query_id}/{group_id} has invalid/duplicate chunk_ids")
            unknown = sorted(set(ids) - set(chunks))
            if unknown:
                raise BenchmarkValidationError(f"{query_id}/{group_id} references unknown chunks: {unknown}")
            ground_truth_ids.update(ids)
            observed_documents.update(chunks[item]["document_id"] for item in ids)
            evidence_group_count += 1
        if declared_documents != observed_documents:
            raise BenchmarkValidationError(
                f"{query_id}.source_documents mismatch: declared={sorted(declared_documents)} "
                f"observed={sorted(observed_documents)}"
            )
    return {
        "status": "PASS",
        "query_count": len(queries),
        "evidence_group_count": evidence_group_count,
        "unique_ground_truth_chunks": len(ground_truth_ids),
        "human_verified": sum(q["verification"]["status"] == "human_verified" for q in queries),
    }


def export_runtime_queries(benchmark_path: Path, output_path: Path) -> None:
    """Export only runtime-safe fields; evidence labels never cross this boundary."""

    value = load_json(benchmark_path)
    rows = [{"id": row["id"], "query": row["query"]} for row in value["queries"]]
    atomic_json(output_path, {"schema_version": "s2g-runtime-queries-v1", "queries": rows})


def write_verification_worksheet(
    benchmark_path: Path, corpus_path: Path, output_path: Path
) -> None:
    """Write source evidence plus blank reviewer fields; make no human judgments."""

    benchmark = load_json(benchmark_path)
    chunks = corpus_index(corpus_path)
    fields = [
        "query_id", "query_type", "query", "evidence_group", "group_description",
        "chunk_id", "document_title", "pages", "structure", "source_spans",
        "source_text", "human_query_natural", "human_evidence_sufficient",
        "human_chunk_relevant", "human_pdf_verified", "human_notes",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for query in benchmark["queries"]:
            for group in query["evidence_groups"]:
                for chunk_id in group["chunk_ids"]:
                    row = chunks[chunk_id]
                    structure = {
                        key: row.get(key)
                        for key in ("chapter", "section", "article", "clause", "point")
                    }
                    writer.writerow({
                        "query_id": query["id"],
                        "query_type": query["query_type"],
                        "query": query["query"],
                        "evidence_group": group["id"],
                        "group_description": group["description"],
                        "chunk_id": chunk_id,
                        "document_title": row["document_title"],
                        "pages": f"{row['page_start']}-{row['page_end']}",
                        "structure": json.dumps(structure, ensure_ascii=False),
                        "source_spans": json.dumps(row["source_spans"], ensure_ascii=False),
                        "source_text": row["source_text"],
                        "human_query_natural": "",
                        "human_evidence_sufficient": "",
                        "human_chunk_relevant": "",
                        "human_pdf_verified": "",
                        "human_notes": "",
                    })
