from __future__ import annotations

import pytest

from src.evaluation.metrics import evaluate_rankings, query_metrics
from src.evaluation.telemetry import summarize_calls, usage_cost_usd


def _query():
    return {
        "id": "Q1", "query_type": "multi_evidence",
        "evidence_groups": [
            {"id": "g1", "chunk_ids": ["A", "A2"]},
            {"id": "g2", "chunk_ids": ["B"]},
        ],
    }


def test_recall_mrr_and_partial_evidence_group_coverage():
    row = query_metrics(_query(), ["X", "A", "Y", "B"])
    assert row["recall@1"] == 0
    assert row["recall@3"] == pytest.approx(1 / 3)
    assert row["recall@5"] == pytest.approx(2 / 3)
    assert row["mrr@10"] == 0.5
    assert row["evidence_group_coverage@3"] == 0.5
    assert row["evidence_group_coverage@5"] == 1.0
    assert row["complete_evidence@5"] is True


def test_macro_and_grouped_aggregation():
    queries = [_query(), {**_query(), "id": "Q2", "query_type": "direct_lookup"}]
    result = evaluate_rankings(queries, {"Q1": ["A", "B"], "Q2": ["X"]})
    assert result["aggregate"]["query_count"] == 2
    assert result["aggregate"]["complete_evidence_queries@3"] == 1
    assert result["by_query_type"]["multi_evidence"]["evidence_group_coverage@3"] == 1.0


def test_usage_cost_and_stage_aggregation():
    usage = {
        "input_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 200},
        "output_tokens": 100,
        "output_tokens_details": {"reasoning_tokens": 40},
    }
    expected = (800 * 0.05 + 200 * 0.005 + 100 * 0.40) / 1_000_000
    assert usage_cost_usd(usage) == pytest.approx(expected)
    result = summarize_calls([{"stage": "s2g_judge", "usage_metadata": usage, "latency_seconds": 1.25}])
    assert result["provider_calls"] == 1
    assert result["reasoning_tokens"] == 40
    assert result["estimated_cost_usd"] == pytest.approx(expected)
