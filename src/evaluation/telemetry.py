"""Latency and GPT-5 nano token-cost aggregation for completed S2G artifacts."""

from __future__ import annotations

from typing import Any, Iterable


GPT5_NANO_USD_PER_MILLION = {
    "input": 0.05,
    "cached_input": 0.005,
    "output": 0.40,
}
PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-5-nano"
PRICING_OBSERVED_DATE = "2026-09-13"


def usage_cost_usd(usage: dict[str, Any] | None) -> float:
    usage = usage or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    uncached = max(0, input_tokens - cached)
    return (
        uncached * GPT5_NANO_USD_PER_MILLION["input"]
        + cached * GPT5_NANO_USD_PER_MILLION["cached_input"]
        + output * GPT5_NANO_USD_PER_MILLION["output"]
    ) / 1_000_000


def summarize_calls(calls: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(calls)
    input_tokens = output_tokens = cached_tokens = reasoning_tokens = 0
    latency = cost = 0.0
    by_stage: dict[str, dict[str, float | int]] = {}
    for call in rows:
        usage = call.get("usage_metadata") or call.get("usage") or {}
        stage = str(call.get("stage") or "answer_reasoner")
        inp = int(usage.get("input_tokens") or 0)
        out = int(usage.get("output_tokens") or 0)
        cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
        reasoning = int((usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)
        call_latency = float(call.get("latency_seconds") or 0.0)
        call_cost = usage_cost_usd(usage)
        input_tokens += inp
        output_tokens += out
        cached_tokens += cached
        reasoning_tokens += reasoning
        latency += call_latency
        cost += call_cost
        bucket = by_stage.setdefault(stage, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_seconds": 0.0, "cost_usd": 0.0})
        bucket["calls"] += 1
        bucket["input_tokens"] += inp
        bucket["output_tokens"] += out
        bucket["latency_seconds"] += call_latency
        bucket["cost_usd"] += call_cost
    return {
        "provider_calls": len(rows),
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "provider_latency_seconds": latency,
        "estimated_cost_usd": cost,
        "pricing": {**GPT5_NANO_USD_PER_MILLION, "unit": "USD per 1M tokens", "source": PRICING_SOURCE, "observed_date": PRICING_OBSERVED_DATE},
        "by_stage": by_stage,
    }
