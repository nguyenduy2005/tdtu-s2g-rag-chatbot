import json
from pathlib import Path

import pytest

from src.evaluation.evaluator import evaluate_ranking_artifact
from src.evaluation.runner import load_runtime_queries
from src.evaluation.telemetry import summarize_calls, usage_cost_usd


def _benchmark():
    return {
        "queries": [
            {
                "id": "Q1",
                "query_type": "direct_lookup",
                "evidence_groups": [{"id": "g1", "chunk_ids": ["a"]}],
            },
            {
                "id": "Q2",
                "query_type": "multi_evidence",
                "evidence_groups": [
                    {"id": "g1", "chunk_ids": ["b"]},
                    {"id": "g2", "chunk_ids": ["c"]},
                ],
            },
        ]
    }


def test_evaluate_multiple_ranking_stages():
    artifact = {
        "method_id": "demo",
        "queries": [
            {"id": "Q1", "rankings": {"bm25": ["x", "a"], "hybrid": ["a"]}, "latency_seconds": {"total": 0.2}},
            {"id": "Q2", "rankings": {"bm25": ["b"], "hybrid": ["b", "c"]}, "latency_seconds": {"total": 0.4}},
        ],
    }
    result = evaluate_ranking_artifact(_benchmark(), artifact)
    assert result["retrieval_metrics"]["hybrid"]["aggregate"]["complete_evidence_queries@3"] == 2
    assert result["retrieval_metrics"]["bm25"]["aggregate"]["complete_evidence_queries@3"] == 1
    assert result["retrieval_latency_seconds"]["total"]["median"] == pytest.approx(0.3)


def test_openai_cached_input_cost_is_not_double_counted():
    usage = {
        "input_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 200},
        "output_tokens": 100,
    }
    assert usage_cost_usd(usage) == pytest.approx((800 * 0.05 + 200 * 0.005 + 100 * 0.40) / 1_000_000)
    summary = summarize_calls([{"stage": "judge", "usage_metadata": usage, "latency_seconds": 1.25}])
    assert summary["cached_input_tokens"] == 200
    assert summary["provider_latency_seconds"] == 1.25


def test_runtime_query_gt_firewall(tmp_path: Path):
    safe = tmp_path / "safe.json"
    safe.write_text(json.dumps({"schema_version": "s2g-runtime-queries-v1", "queries": [{"id": "Q1", "query": "Question"}]}))
    assert load_runtime_queries(safe) == [{"id": "Q1", "query": "Question"}]
    unsafe = tmp_path / "unsafe.json"
    unsafe.write_text(json.dumps({"schema_version": "s2g-runtime-queries-v1", "queries": [{"id": "Q1", "query": "Question", "chunk_ids": ["secret"]}]}))
    with pytest.raises(ValueError, match="GT firewall FAIL"):
        load_runtime_queries(unsafe)
