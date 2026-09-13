"""Offline retrieval metrics with evidence-group semantics."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Iterable


DEFAULT_KS = (1, 3, 5, 10)


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def query_metrics(query: dict[str, Any], ranking: Iterable[str], ks: tuple[int, ...] = DEFAULT_KS) -> dict[str, Any]:
    ranked = _unique(ranking)
    groups = query["evidence_groups"]
    gt = set(item for group in groups for item in group["chunk_ids"])
    first_rank = next((index for index, chunk_id in enumerate(ranked[:10], start=1) if chunk_id in gt), None)
    result: dict[str, Any] = {
        "query_id": query["id"],
        "query_type": query["query_type"],
        "ground_truth_chunks": len(gt),
        "evidence_groups": len(groups),
        "mrr@10": 0.0 if first_rank is None else 1.0 / first_rank,
        "first_relevant_rank": first_rank,
    }
    for k in ks:
        top = set(ranked[:k])
        covered_groups = sum(bool(top.intersection(group["chunk_ids"])) for group in groups)
        result[f"recall@{k}"] = len(top.intersection(gt)) / len(gt)
        result[f"evidence_group_coverage@{k}"] = covered_groups / len(groups)
        result[f"complete_evidence@{k}"] = covered_groups == len(groups)
    return result


def evaluate_rankings(
    queries: list[dict[str, Any]],
    rankings: dict[str, list[str]],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[str, Any]:
    missing = sorted({query["id"] for query in queries} - set(rankings))
    extra = sorted(set(rankings) - {query["id"] for query in queries})
    if missing or extra:
        raise ValueError(f"Ranking/query mismatch: missing={missing}, extra={extra}")
    rows = [query_metrics(query, rankings[query["id"]], ks) for query in queries]

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        result = {"query_count": len(items), "mrr@10": mean(row["mrr@10"] for row in items)}
        for k in ks:
            result[f"recall@{k}"] = mean(row[f"recall@{k}"] for row in items)
            result[f"evidence_group_coverage@{k}"] = mean(
                row[f"evidence_group_coverage@{k}"] for row in items
            )
            result[f"complete_evidence_queries@{k}"] = sum(row[f"complete_evidence@{k}"] for row in items)
        return result

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["query_type"]].append(row)
    return {
        "schema_version": "s2g-retrieval-metrics-v1",
        "definitions": {
            "recall@k": "retrieved unique GT chunks / all unique GT chunks, macro averaged",
            "mrr@10": "reciprocal rank of first GT chunk within top 10, macro averaged",
            "evidence_group_coverage@k": "covered required groups / all required groups, macro averaged",
        },
        "aggregate": aggregate(rows),
        "by_query_type": {key: aggregate(value) for key, value in sorted(grouped.items())},
        "per_query": rows,
    }
