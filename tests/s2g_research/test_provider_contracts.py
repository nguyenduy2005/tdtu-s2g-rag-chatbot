from __future__ import annotations

import json

import pytest

from src.retrieval.s2g_research.prompts import build_judge_prompt
from src.retrieval.s2g_research.schemas import (
    JUDGE_SCHEMA,
    ResearchSchemaError,
    parse_judge,
    parse_selector,
    selector_schema,
)


def test_judge_contract_uses_gap_items_and_context_only_empty_input():
    assert JUDGE_SCHEMA["required"] == ["sufficient", "gap_items"]
    prompt, hashes = build_judge_prompt("Câu hỏi?", ())
    assert '"evidence_context":[]' in prompt
    assert "parametric knowledge" in prompt
    assert hashes == ()


def test_judge_parser_enforces_sufficiency_invariants():
    decision = parse_judge(json.dumps({
        "sufficient": False,
        "gap_items": [{
            "category": "attribute",
            "target": "học bổng",
            "slot": "điều kiện",
            "description": "Điều kiện nhận học bổng",
        }],
    }))
    assert decision.sufficient is False
    assert decision.gaps[0].target == "học bổng"
    with pytest.raises(ResearchSchemaError, match="requires at least one gap item"):
        parse_judge('{"sufficient":false,"gap_items":[]}')
    with pytest.raises(ResearchSchemaError, match="requires gap_items"):
        parse_judge('{"sufficient":true,"gap_items":[{"category":"other","target":null,"slot":null,"description":"x"}]}')


def test_pointer_contract_uses_exact_global_id_vocabulary():
    schema = selector_schema(["S1", "S2"])
    field = schema["properties"]["evidence_global_ids"]
    assert field["items"]["enum"] == ["S1", "S2"]
    assert parse_selector('{"evidence_global_ids":["S2"]}').sentence_ids == ("S2",)
    with pytest.raises(ResearchSchemaError, match="Duplicate evidence ID"):
        parse_selector('{"evidence_global_ids":["S1","S1"]}')
