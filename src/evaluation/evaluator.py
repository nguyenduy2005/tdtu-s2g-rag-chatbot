"""Offline-only evaluation of frozen rankings and completed S2G traces."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from src.evaluation.metrics import evaluate_rankings
from src.evaluation.telemetry import summarize_calls
from src.product_s2g.io import atomic_json


def _deduplicate(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(values: Iterable[float]) -> dict[str, float]:
    rows = list(values)
    return {
        "mean": mean(rows) if rows else 0.0,
        "median": median(rows) if rows else 0.0,
        "p95": _percentile(rows, 0.95),
        "max": max(rows, default=0.0),
    }


def evaluate_ranking_artifact(
    benchmark: dict[str, Any], artifact: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate every named stage in a GT-free ranking artifact."""

    queries = benchmark["queries"]
    rows = artifact.get("queries")
    if not isinstance(rows, list):
        raise ValueError("Ranking artifact queries must be a list")
    expected = {query["id"] for query in queries}
    if {row.get("id") for row in rows} != expected:
        raise ValueError("Ranking artifact does not contain exactly the benchmark query IDs")
    stage_names = {stage for row in rows for stage in (row.get("rankings") or {})}
    if not stage_names:
        raise ValueError("Ranking artifact contains no stages")
    stage_results = {}
    for stage in sorted(stage_names):
        rankings = {}
        for row in rows:
            stage_ranking = (row.get("rankings") or {}).get(stage)
            if not isinstance(stage_ranking, list):
                raise ValueError(f"{row.get('id')} is missing ranking stage {stage}")
            rankings[row["id"]] = stage_ranking
        stage_results[stage] = evaluate_rankings(queries, rankings)
    latency_rows = [row.get("latency_seconds") or {} for row in rows]
    latency_stages = {key for row in latency_rows for key in row}
    return {
        "schema_version": "s2g-offline-evaluation-v1",
        "method_id": artifact.get("method_id"),
        "query_count": len(rows),
        "retrieval_metrics": stage_results,
        "retrieval_latency_seconds": {
            stage: _summary(float(row.get(stage) or 0.0) for row in latency_rows)
            for stage in sorted(latency_stages)
        },
    }


def _completed_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    core = value.get("s2g_core") or {}
    response = value.get("response") or {}
    if core.get("closed") is not True or core.get("error") is not None:
        raise ValueError(f"Incomplete S2G result: {path}")
    if response.get("request_id") != core.get("query_id"):
        raise ValueError(f"Request/query ID mismatch: {path}")
    return value


def s2g_ranking_artifact(
    benchmark: dict[str, Any], runtime_root: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Extract GT-free candidate/final rankings and resource telemetry."""

    rows: list[dict[str, Any]] = []
    telemetry_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    for query in benchmark["queries"]:
        result_path = runtime_root / "runtime" / query["id"] / "result.json"
        if not result_path.is_file():
            raise ValueError(f"Missing completed S2G result: {query['id']}")
        value = _completed_result(result_path)
        core = value["s2g_core"]
        response = value["response"]
        candidate_pool: list[str] = []
        presented_pool: list[str] = []
        calls: list[dict[str, Any]] = []
        retrieval_seconds = 0.0
        for turn in core.get("turns") or ():
            retrieval = turn.get("retrieval") or {}
            candidate_pool.extend(
                row["chunk_id"] for row in retrieval.get("items") or () if row.get("chunk_id")
            )
            audit = retrieval.get("audit") or {}
            presented_pool.extend(
                row["chunk_id"] for row in audit.get("presented") or () if row.get("chunk_id")
            )
            retrieval_seconds += float(retrieval.get("latency_seconds") or 0.0)
            for stage in ("judge", "extractor"):
                if isinstance(turn.get(stage), dict):
                    calls.append(turn[stage])
        answer_provider = value.get("answer_provider") or {}
        if answer_provider.get("provider_called"):
            calls.append({"stage": "answer_reasoner", **answer_provider})
        call_summary = summarize_calls(calls)
        total_latency = float(core.get("latency_seconds") or 0.0)
        citations = [row["chunk_id"] for row in response.get("citations") or ()]
        final_evidence = list(core.get("final_parent_chunk_ids") or ())
        rows.append({
            "id": query["id"],
            "rankings": {
                "iterative_candidate_pool": _deduplicate(candidate_pool),
                "iterative_presented_pool": _deduplicate(presented_pool),
                "s2g_selected_evidence": _deduplicate(final_evidence),
                "answer_citations": _deduplicate(citations),
            },
            "latency_seconds": {
                "retrieval": retrieval_seconds,
                "provider": call_summary["provider_latency_seconds"],
                "end_to_end": total_latency,
            },
        })
        telemetry_rows.append({
            "query_id": query["id"],
            "query_type": query["query_type"],
            "stop_reason": core.get("stop_reason"),
            "retrieval_rounds": (core.get("budget") or {}).get("retrieval_calls", 0),
            "end_to_end_latency_seconds": total_latency,
            **call_summary,
        })
        final_ids = set(final_evidence)
        answer_rows.append({
            "query_id": query["id"],
            "query_type": query["query_type"],
            "question": query["query"],
            "answer": response.get("answer", ""),
            "insufficient": response.get("insufficient"),
            "confidence": response.get("confidence"),
            "citation_chunk_ids": citations,
            "final_evidence_chunk_ids": final_evidence,
            "citation_contract_valid": bool(citations) or bool(response.get("insufficient")),
            "citations_within_final_evidence": set(citations).issubset(final_ids),
        })
    ranking = {
        "schema_version": "s2g-retrieval-rankings-v1",
        "method_id": "s2g_judge_first",
        "queries": rows,
    }
    pricing = telemetry_rows[0]["pricing"] if telemetry_rows else {}
    telemetry = {
        "schema_version": "s2g-resource-evaluation-v1",
        "query_count": len(telemetry_rows),
        "provider_calls": sum(row["provider_calls"] for row in telemetry_rows),
        "input_tokens": sum(row["input_tokens"] for row in telemetry_rows),
        "cached_input_tokens": sum(row["cached_input_tokens"] for row in telemetry_rows),
        "output_tokens": sum(row["output_tokens"] for row in telemetry_rows),
        "reasoning_tokens": sum(row["reasoning_tokens"] for row in telemetry_rows),
        "estimated_cost_usd": sum(row["estimated_cost_usd"] for row in telemetry_rows),
        "latency_seconds": _summary(row["end_to_end_latency_seconds"] for row in telemetry_rows),
        "retrieval_rounds": _summary(float(row["retrieval_rounds"]) for row in telemetry_rows),
        "stop_reasons": dict(Counter(row["stop_reason"] for row in telemetry_rows)),
        "pricing": pricing,
        "per_query": telemetry_rows,
    }
    return ranking, telemetry, answer_rows


def write_answer_review(path: Path, answer_rows: list[dict[str, Any]]) -> None:
    """Create a human worksheet; never infer semantic answer judgments."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id", "query_type", "question", "answer", "insufficient", "confidence",
        "citation_chunk_ids", "final_evidence_chunk_ids", "citation_contract_valid",
        "citations_within_final_evidence", "human_correct", "human_complete",
        "human_faithful", "human_citations_adequate", "human_clear", "human_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in answer_rows:
            output = dict(row)
            output["citation_chunk_ids"] = json.dumps(output["citation_chunk_ids"], ensure_ascii=False)
            output["final_evidence_chunk_ids"] = json.dumps(output["final_evidence_chunk_ids"], ensure_ascii=False)
            for field in fieldnames[10:]:
                output[field] = ""
            writer.writerow(output)


def write_s2g_evaluation(
    benchmark: dict[str, Any], runtime_root: Path, output_root: Path
) -> dict[str, Any]:
    ranking, telemetry, answers = s2g_ranking_artifact(benchmark, runtime_root)
    evaluation = evaluate_ranking_artifact(benchmark, ranking)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "s2g_rankings.json", ranking)
    atomic_json(output_root / "s2g_retrieval_metrics.json", evaluation)
    atomic_json(output_root / "s2g_resource_metrics.json", telemetry)
    write_answer_review(output_root / "answer_quality_review.csv", answers)
    return {"retrieval": evaluation, "resources": telemetry}
