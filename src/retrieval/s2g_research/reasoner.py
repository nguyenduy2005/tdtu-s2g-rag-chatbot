"""Evidence-only adapter for the decoupled answer reasoner."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.retrieval.s2g.sentence_context import SentenceRecord


class EvidenceOnlyAnswerReasoner:
    """Call the product answer generator with exactly selected evidence text.

    Full parent chunk text is deliberately not exposed to the answer model.
    Metadata is retained only for citations and provenance.
    """

    def __init__(self, generator: Any, chunks: dict[str, dict[str, Any]]) -> None:
        self.generator = generator
        self.chunks = chunks

    def reason(self, question: str, context: tuple[SentenceRecord, ...]) -> Any:
        selected: defaultdict[str, list[str]] = defaultdict(list)
        for sentence in context:
            selected[sentence.parent_chunk_id].append(sentence.text)
        evidence = []
        for chunk_id, sentences in selected.items():
            if chunk_id not in self.chunks:
                raise ValueError(f"Evidence Context references unknown chunk: {chunk_id}")
            row = self.chunks[chunk_id]
            selected_text = "\n".join(sentences)
            evidence.append({
                "chunk_id": chunk_id,
                "document_title": row["document_title"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
                "retrieval_role": row["retrieval_role"],
                "quality_flags": list(row.get("quality_flags") or ()),
                "context_prefix": None,
                "source_text": selected_text,
                "selected_sentences": list(sentences),
            })
        return self.generator.generate(question, evidence)
