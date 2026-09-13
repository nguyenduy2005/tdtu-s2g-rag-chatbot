"""Validation and aggregation for explicit answer-quality review records."""

from __future__ import annotations

from collections import Counter
from typing import Any


CRITERIA = ("correctness", "completeness", "faithfulness", "citation_adequacy", "clarity")
ALLOWED = {"PASS", "FAIL", "REVIEW"}


def summarize_answer_reviews(value: dict[str, Any], expected_ids: set[str]) -> dict[str, Any]:
    reviews = value.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("Answer reviews must be a list")
    identifiers = [row.get("query_id") for row in reviews]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != expected_ids:
        raise ValueError("Answer reviews must contain every query exactly once")
    summary: dict[str, Any] = {
        "schema_version": "s2g-answer-quality-summary-v1",
        "query_count": len(reviews),
        "independent_human_review": value.get("independent_human_review") is True,
        "criteria": {},
    }
    for criterion in CRITERIA:
        statuses = [row.get(criterion) for row in reviews]
        if any(status not in ALLOWED for status in statuses):
            raise ValueError(f"Invalid {criterion} status")
        counts = Counter(statuses)
        decided = counts["PASS"] + counts["FAIL"]
        summary["criteria"][criterion] = {
            "pass": counts["PASS"],
            "fail": counts["FAIL"],
            "review": counts["REVIEW"],
            "pass_rate_all_queries": counts["PASS"] / len(reviews),
            "pass_rate_decided": counts["PASS"] / decided if decided else None,
        }
    core = CRITERIA[:4]
    summary["all_core_criteria_pass"] = sum(
        all(row[criterion] == "PASS" for criterion in core) for row in reviews
    )
    summary["queries_requiring_review_or_with_failure"] = [
        row["query_id"]
        for row in reviews
        if any(row[criterion] != "PASS" for criterion in CRITERIA)
    ]
    return summary
