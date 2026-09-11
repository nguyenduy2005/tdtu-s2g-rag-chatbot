"""OpenAI strict-output adapters for the research S2G contracts."""

from __future__ import annotations

from src.s2g_runtime.openai_adapter import OpenAICaller
from src.retrieval.s2g.schemas import GapItem
from src.retrieval.s2g.sentence_context import SentenceRecord

from .prompts import PROMPT_SHA256, build_extractor_prompt, build_judge_prompt
from .schemas import JUDGE_SCHEMA, parse_judge, parse_selector, selector_schema


class OpenAIResearchJudge:
    def __init__(self, caller: OpenAICaller) -> None:
        self.caller = caller

    def judge(self, q_0: str, context: tuple[SentenceRecord, ...]):
        prompt, hashes = build_judge_prompt(q_0, context)
        return self.caller.request(
            stage="s2g_judge",
            prompt=prompt,
            template_sha256=PROMPT_SHA256["judge"],
            parser=parse_judge,
            max_output_tokens=4096,
            schema=JUDGE_SCHEMA,
            source_text_sha256=hashes,
        )


class OpenAIResearchEvidenceExtractor:
    def __init__(self, caller: OpenAICaller) -> None:
        self.caller = caller

    def select(
        self,
        q_0: str,
        gaps: tuple[GapItem, ...],
        sentence_pool: tuple[SentenceRecord, ...],
    ):
        prompt, hashes = build_extractor_prompt(q_0, gaps, sentence_pool)
        return self.caller.request(
            stage="s2g_sentence_selector",
            prompt=prompt,
            template_sha256=PROMPT_SHA256["extractor"],
            parser=parse_selector,
            max_output_tokens=4096,
            schema=selector_schema(item.sentence_id for item in sentence_pool),
            source_text_sha256=hashes,
        )
