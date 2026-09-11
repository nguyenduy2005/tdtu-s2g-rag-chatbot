"""Deterministic Vietnamese sentence view and append-only Evidence Context."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from src.s2g_runtime.runtime_types import EvidenceItem


ABBREVIATIONS = {"pgs.", "ts.", "ths.", "gs.", "tp.", "q.", "p.", "đh.", "tr.", "số."}
LIST_MARKER = re.compile(
    r"(?mi)^[ \t]*(?:[-•+]|[a-zđ][.)]|\d+[.)]|\d+(?:\.\d+)+(?:[.)])?)[ \t]+"
)
BLANK_LINE = re.compile(r"\n[ \t]*\n+")
CLOSERS = '\"\'”’»)]}'


@dataclass(frozen=True)
class SentenceRecord:
    sentence_id: str
    parent_chunk_id: str
    source_document_identifier: str
    page_start: int
    page_end: int
    parent_source_spans: tuple[dict[str, Any], ...]
    char_start: int
    char_end: int
    text: str


def _trimmed_interval(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace(): start += 1
    while end > start and text[end - 1].isspace(): end -= 1
    return (start, end) if start < end else None


def _is_abbreviation(text: str, period: int) -> bool:
    begin = period
    while begin > 0 and not text[begin - 1].isspace(): begin -= 1
    return text[begin:period + 1].casefold() in ABBREVIATIONS


def sentence_intervals(text: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(text, str):
        raise TypeError("source_text must be a string")
    cuts = {0, len(text)}
    for match in BLANK_LINE.finditer(text):
        cuts.add(match.start() + 1); cuts.add(match.end())
    for match in LIST_MARKER.finditer(text):
        if match.start() > 0:
            cuts.add(match.start())
    index = 0
    while index < len(text):
        char = text[index]
        if char in ".!?":
            if char == ".":
                before = text[index - 1] if index else ""
                after = text[index + 1] if index + 1 < len(text) else ""
                if (before.isdigit() and after.isdigit()) or _is_abbreviation(text, index):
                    index += 1; continue
            end = index + 1
            while end < len(text) and text[end] in CLOSERS:
                end += 1
            if end == len(text) or (text[end].isspace() and any(not c.isspace() for c in text[end:])):
                cuts.add(end)
            index = end; continue
        index += 1
    ordered = sorted(cuts)
    result = []
    for start, end in zip(ordered, ordered[1:]):
        interval = _trimmed_interval(text, start, end)
        if interval:
            result.append(interval)
    # No non-whitespace source character may disappear or appear twice.
    accounted = set()
    for start, end in result:
        accounted.update(i for i in range(start, end) if not text[i].isspace())
    expected = {i for i, char in enumerate(text) if not char.isspace()}
    if accounted != expected:
        raise RuntimeError("Sentence segmentation lost or duplicated non-whitespace source text")
    return tuple(result)


def segment_chunk(evidence: EvidenceItem) -> tuple[SentenceRecord, ...]:
    rows = []
    for start, end in sentence_intervals(evidence.source_text):
        sentence = evidence.source_text[start:end]
        digest = hashlib.sha256(sentence.encode("utf-8")).hexdigest()[:16]
        rows.append(SentenceRecord(
            sentence_id=f"s2g:{evidence.chunk_id}:{start}:{end}:{digest}",
            parent_chunk_id=evidence.chunk_id,
            source_document_identifier=evidence.source_document_identifier,
            page_start=evidence.page_start,
            page_end=evidence.page_end,
            parent_source_spans=evidence.source_spans,
            char_start=start,
            char_end=end,
            text=sentence,
        ))
    return tuple(rows)


@dataclass
class EvidenceContext:
    sentence_limit: int = 30
    parent_limit: int = 30
    sentences: list[SentenceRecord] = field(default_factory=list)
    _ids: set[str] = field(default_factory=set, repr=False)

    def admit(self, selected_ids: Iterable[str], pool: tuple[SentenceRecord, ...]) -> tuple[SentenceRecord, ...]:
        by_id = {row.sentence_id: row for row in pool}
        ids = tuple(selected_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate sentence pointer")
        if any(pointer not in by_id for pointer in ids):
            raise ValueError("Invalid or hallucinated sentence pointer")
        admitted = []
        parents = {row.parent_chunk_id for row in self.sentences}
        for pointer in ids:
            if pointer in self._ids:
                continue
            row = by_id[pointer]
            if len(self.sentences) >= self.sentence_limit:
                break
            if row.parent_chunk_id not in parents and len(parents) >= self.parent_limit:
                continue
            self.sentences.append(row); self._ids.add(pointer); parents.add(row.parent_chunk_id); admitted.append(row)
        return tuple(admitted)

    @property
    def parent_chunk_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(row.parent_chunk_id for row in self.sentences))
