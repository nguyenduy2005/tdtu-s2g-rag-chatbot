"""Strict method-intrinsic schemas for the adapted S2G controller."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


GAP_CATEGORIES = ("bridge_entity", "attribute", "relation", "evidence_span", "other")

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sufficient", "gaps"],
    "properties": {
        "sufficient": {"type": "boolean"},
        "gaps": {
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

SELECTOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sentence_ids"],
    "properties": {
        "sentence_ids": {
            "type": "array",
            "maxItems": 10,
            "items": {"type": "string", "minLength": 1},
        }
    },
}


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class GapItem:
    category: str
    target: str | None
    slot: str | None
    description: str


@dataclass(frozen=True)
class JudgeDecision:
    sufficient: bool
    gaps: tuple[GapItem, ...]


@dataclass(frozen=True)
class SelectorDecision:
    sentence_ids: tuple[str, ...]


def canonical_schema_sha256(schema: dict[str, Any]) -> str:
    raw = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SchemaError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _parse(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_object_no_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(SchemaError(f"Invalid constant: {item}")),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SchemaError(str(exc)) from exc
    if not isinstance(value, dict):
        raise SchemaError("Top-level response must be an object")
    return value


def _nullable_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SchemaError(f"{field} must be a string or null")
    trimmed = value.strip()
    return trimmed or None


def parse_judge(raw: str) -> JudgeDecision:
    value = _parse(raw)
    if set(value) != {"sufficient", "gaps"} or type(value["sufficient"]) is not bool:
        raise SchemaError("Judge response must contain exactly boolean sufficient and array gaps")
    rows = value["gaps"]
    if not isinstance(rows, list) or len(rows) > 3:
        raise SchemaError("gaps must contain at most three items")
    gaps: list[GapItem] = []
    fingerprints: set[tuple[str, str | None, str | None, str]] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"category", "target", "slot", "description"}:
            raise SchemaError("Each gap must contain exactly category, target, slot, description")
        if row["category"] not in GAP_CATEGORIES:
            raise SchemaError("Unknown gap category")
        target = _nullable_text(row["target"], "target")
        slot = _nullable_text(row["slot"], "slot")
        description = row["description"]
        if not isinstance(description, str) or not 1 <= len(description.strip()) <= 300:
            raise SchemaError("description must contain 1-300 Unicode code points")
        item = GapItem(row["category"], target, slot, description.strip())
        key = (item.category, item.target, item.slot, item.description.casefold())
        if key in fingerprints:
            raise SchemaError("Duplicate structured gap")
        fingerprints.add(key); gaps.append(item)
    if value["sufficient"] and gaps:
        raise SchemaError("sufficient=true requires gaps=[]")
    if not value["sufficient"] and not gaps:
        raise SchemaError("sufficient=false requires at least one structured gap")
    return JudgeDecision(value["sufficient"], tuple(gaps))


def parse_selector(raw: str) -> SelectorDecision:
    value = _parse(raw)
    if set(value) != {"sentence_ids"} or not isinstance(value["sentence_ids"], list):
        raise SchemaError("Selector response must contain exactly sentence_ids")
    ids = value["sentence_ids"]
    if len(ids) > 10 or any(not isinstance(item, str) or not item.strip() for item in ids):
        raise SchemaError("sentence_ids must contain at most ten non-empty strings")
    if len(ids) != len(set(ids)):
        raise SchemaError("sentence_ids must be unique")
    return SelectorDecision(tuple(ids))
