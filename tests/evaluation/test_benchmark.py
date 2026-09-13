from __future__ import annotations

import json

import pytest

from src.evaluation.benchmark import BenchmarkValidationError, validate_benchmark


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _benchmark(chunk_id="C1"):
    return {
        "schema_version": "s2g-retrieval-benchmark-v1",
        "queries": [{
            "id": "Q1", "query": "Câu hỏi?", "query_type": "direct_lookup", "answerable": True,
            "source_documents": ["D1"],
            "evidence_groups": [{"id": "g1", "description": "x", "chunk_ids": [chunk_id]}],
            "verification": {"status": "agent_verified_pending_human", "notes": "fixture"},
        }],
    }


def test_validator_accepts_draft_and_blocks_official_without_human_verification(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text(json.dumps({"chunk_id": "C1", "document_id": "D1"}) + "\n")
    benchmark = tmp_path / "queries.json"
    _write(benchmark, _benchmark())
    assert validate_benchmark(benchmark, corpus)["status"] == "PASS"
    with pytest.raises(BenchmarkValidationError, match="human verified"):
        validate_benchmark(benchmark, corpus, require_human_verified=True)


def test_validator_rejects_unknown_ground_truth_chunk(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text(json.dumps({"chunk_id": "C1", "document_id": "D1"}) + "\n")
    benchmark = tmp_path / "queries.json"
    _write(benchmark, _benchmark("MISSING"))
    with pytest.raises(BenchmarkValidationError, match="unknown chunks"):
        validate_benchmark(benchmark, corpus)


def test_validator_rejects_duplicate_query_ids(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text(json.dumps({"chunk_id": "C1", "document_id": "D1"}) + "\n")
    value = _benchmark()
    value["queries"].append(dict(value["queries"][0]))
    benchmark = tmp_path / "queries.json"
    _write(benchmark, value)
    with pytest.raises(BenchmarkValidationError, match="Duplicate query"):
        validate_benchmark(benchmark, corpus)
