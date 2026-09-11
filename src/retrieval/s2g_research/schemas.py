"""Strict provider contracts matching the judge-first S2G specification."""

from __future__ import annotations

import json
from typing import Any, Iterable

from src.retrieval.s2g.schemas import GAP_CATEGORIES, JudgeDecision, SelectorDecision


class ResearchSchemaError(ValueError):
    pass


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sufficient", "gap_items"],
    "properties": {
        "sufficient": {"type": "boolean"},
        "gap_items": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["category", "target", "slot", "description"],
                "properties": {
                    "category": {"type": "string", "enum": list(GAP_CATEGORIES)},
                    "target": {"type": ["string", "null"]},
                    "slot": {"type": ["string", "null"]},
                    "description": {"type": "string", "minLength": 1, "maxLength": 300},
                },
            },
        },
    },
}


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ResearchSchemaError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicates)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ResearchSchemaError(str(exc)) from exc
    if not isinstance(value, dict):
        raise ResearchSchemaError("Top-level response must be an object")
    return value


def parse_judge(raw: str) -> JudgeDecision:
    value = _object(raw)
    if set(value) != {"sufficient", "gap_items"} or type(value["sufficient"]) is not bool:
        raise ResearchSchemaError("Judge requires exactly sufficient and gap_items")
    rows = value["gap_items"]
    if not isinstance(rows, list) or len(rows) > 3:
        raise ResearchSchemaError("gap_items must contain at most three items")
    gaps = []
    fingerprints = set()
    from src.retrieval.s2g.schemas import GapItem

    for row in rows:
        if not isinstance(row, dict) or set(row) != {"category", "target", "slot", "description"}:
            raise ResearchSchemaError("Invalid gap item fields")
        if row["category"] not in GAP_CATEGORIES:
            raise ResearchSchemaError("Unknown gap category")
        target = row["target"]
        slot = row["slot"]
        description = row["description"]
        if target is not None and not isinstance(target, str):
            raise ResearchSchemaError("target must be string or null")
        if slot is not None and not isinstance(slot, str):
            raise ResearchSchemaError("slot must be string or null")
        if not isinstance(description, str) or not 1 <= len(description.strip()) <= 300:
            raise ResearchSchemaError("description must contain 1-300 characters")
        item = GapItem(
            row["category"],
            target.strip() or None if isinstance(target, str) else None,
            slot.strip() or None if isinstance(slot, str) else None,
            description.strip(),
        )
        fingerprint = (item.category, item.target, item.slot, item.description.casefold())
        if fingerprint in fingerprints:
            raise ResearchSchemaError("Duplicate gap item")
        fingerprints.add(fingerprint)
        gaps.append(item)
    if value["sufficient"] and gaps:
        raise ResearchSchemaError("sufficient=true requires gap_items=[]")
    if not value["sufficient"] and not gaps:
        raise ResearchSchemaError("sufficient=false requires at least one gap item")
    return JudgeDecision(value["sufficient"], tuple(gaps))


def selector_schema(sentence_ids: Iterable[str]) -> dict[str, Any]:
    identifiers = tuple(sentence_ids)
    if any(not isinstance(item, str) or not item for item in identifiers):
        raise ValueError("Evidence IDs must be non-empty strings")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Sentence pool contains duplicate evidence IDs")
    items: dict[str, Any] = {"type": "string"}
    max_items = min(10, len(identifiers))
    if identifiers:
        items["enum"] = list(identifiers)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["evidence_global_ids"],
        "properties": {
            "evidence_global_ids": {
                "type": "array",
                "maxItems": max_items,
                "items": items,
            }
        },
    }


def parse_selector(raw: str) -> SelectorDecision:
    value = _object(raw)
    if set(value) != {"evidence_global_ids"}:
        raise ResearchSchemaError("Extractor requires exactly evidence_global_ids")
    identifiers = value["evidence_global_ids"]
    if not isinstance(identifiers, list) or len(identifiers) > 10:
        raise ResearchSchemaError("evidence_global_ids must be an array of at most ten IDs")
    if any(not isinstance(item, str) or not item for item in identifiers):
        raise ResearchSchemaError("Every evidence ID must be a non-empty string")
    if len(identifiers) != len(set(identifiers)):
        raise ResearchSchemaError("Duplicate evidence ID")
    return SelectorDecision(tuple(identifiers))
