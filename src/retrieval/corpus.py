"""Frozen retrieval-corpus loading and traceability metadata."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RetrievalDocument:
    chunk_id: str
    text: str
    document_id: str
    document_title: str
    source_file: str
    page_start: int
    page_end: int
    source_spans: tuple[dict[str, Any], ...]
    chapter: str | None
    section: str | None
    article: str | None
    clause: str | None
    point: str | None
    unit_type: str
    structure_source: str
    retrieval_role: str

    def traceability(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_title": self.document_title,
            "source_file": self.source_file,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "source_spans": list(self.source_spans),
            "chapter": self.chapter,
            "section": self.section,
            "article": self.article,
            "clause": self.clause,
            "point": self.point,
            "unit_type": self.unit_type,
            "structure_source": self.structure_source,
        }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_file(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"Frozen hash mismatch for {path}: expected {expected_sha256}; observed {actual}")


def _to_document(chunk: dict[str, Any], text_field: str) -> RetrievalDocument:
    text = chunk.get(text_field)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{chunk.get('chunk_id')}: retrieval field {text_field!r} is empty or invalid")
    chunk_id = chunk.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError("Chunk lacks a non-empty chunk_id")
    return RetrievalDocument(
        chunk_id=chunk_id,
        text=text,
        document_id=chunk["document_id"],
        document_title=chunk["document_title"],
        source_file=chunk["source_file"],
        page_start=int(chunk["page_start"]),
        page_end=int(chunk["page_end"]),
        source_spans=tuple(dict(span) for span in chunk["source_spans"]),
        chapter=chunk.get("chapter"),
        section=chunk.get("section"),
        article=chunk.get("article"),
        clause=chunk.get("clause"),
        point=chunk.get("point"),
        unit_type=chunk["unit_type"],
        structure_source=chunk["structure_source"],
        retrieval_role=chunk["retrieval_role"],
    )


def load_retrieval_corpus(
    chunks_path: Path,
    *,
    indexed_roles: Iterable[str] = ("content",),
    text_field: str = "source_text",
) -> tuple[list[RetrievalDocument], dict[str, Any]]:
    """Load only eligible frozen chunks without changing IDs or text."""

    roles = set(indexed_roles)
    if not roles:
        raise ValueError("indexed_roles must contain at least one role")
    documents: list[RetrievalDocument] = []
    seen: set[str] = set()
    total = 0
    excluded: Counter[str] = Counter()
    for line_number, line in enumerate(chunks_path.open(encoding="utf-8"), start=1):
        if not line.strip():
            continue
        total += 1
        chunk = json.loads(line)
        chunk_id = chunk.get("chunk_id")
        if chunk_id in seen:
            raise ValueError(f"Duplicate chunk_id at {chunks_path}:{line_number}: {chunk_id}")
        seen.add(chunk_id)
        role = chunk.get("retrieval_role")
        if role not in roles:
            excluded[f"retrieval_role={role}"] += 1
            continue
        documents.append(_to_document(chunk, text_field))
    if len({doc.chunk_id for doc in documents}) != len(documents):
        raise ValueError("Indexed corpus contains duplicate chunk IDs")
    stats = {
        "total_chunks": total,
        "indexed_chunks": len(documents),
        "excluded_chunks": total - len(documents),
        "indexed_roles": sorted(roles),
        "text_field": text_field,
        "exclusion_reasons": dict(sorted(excluded.items())),
    }
    return documents, stats
