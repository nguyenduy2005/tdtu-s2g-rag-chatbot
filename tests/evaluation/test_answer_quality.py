import pytest

from src.evaluation.answer_quality import summarize_answer_reviews


def test_answer_quality_summary_keeps_review_separate():
    value = {
        "independent_human_review": False,
        "reviews": [
            {"query_id": "Q1", "correctness": "PASS", "completeness": "PASS", "faithfulness": "PASS", "citation_adequacy": "PASS", "clarity": "PASS"},
            {"query_id": "Q2", "correctness": "REVIEW", "completeness": "FAIL", "faithfulness": "PASS", "citation_adequacy": "FAIL", "clarity": "PASS"},
        ],
    }
    result = summarize_answer_reviews(value, {"Q1", "Q2"})
    assert result["criteria"]["correctness"]["review"] == 1
    assert result["criteria"]["correctness"]["pass_rate_all_queries"] == 0.5
    assert result["all_core_criteria_pass"] == 1


def test_answer_quality_requires_exact_query_set():
    with pytest.raises(ValueError, match="every query"):
        summarize_answer_reviews({"reviews": []}, {"Q1"})
