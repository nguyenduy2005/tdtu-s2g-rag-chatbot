"""Reciprocal Rank Fusion without raw-score normalization."""

from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    rankings: dict[str, list[dict[str, Any]]], *, k_rrf: int = 60, top_k: int | None = None
) -> list[dict[str, Any]]:
    if k_rrf < 0:
        raise ValueError("k_rrf must be non-negative")
    if not rankings:
        return []
    fused: dict[str, dict[str, Any]] = {}
    for system_name in sorted(rankings):
        seen: set[str] = set()
        for fallback_rank, item in enumerate(rankings[system_name], start=1):
            chunk_id = item["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank = int(item.get("rank", fallback_rank))
            row = fused.setdefault(chunk_id, {"chunk_id": chunk_id, "rrf_score": 0.0})
            row["rrf_score"] += 1.0 / (k_rrf + rank)
            row[f"{system_name}_rank"] = rank
            score = item.get(f"{system_name}_score", item.get("score"))
            if score is not None:
                row[f"{system_name}_score"] = float(score)
    ordered = sorted(fused.values(), key=lambda row: (-row["rrf_score"], row["chunk_id"]))
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        ordered = ordered[:top_k]
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
        row["rrf_rank"] = rank
        row["score"] = row["rrf_score"]
    return ordered
