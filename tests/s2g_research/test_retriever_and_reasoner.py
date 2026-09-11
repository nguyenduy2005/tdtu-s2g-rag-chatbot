from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.retrieval.corpus import RetrievalDocument
from src.retrieval.s2g.sentence_context import segment_chunk
from src.retrieval.s2g_research.reasoner import EvidenceOnlyAnswerReasoner
from src.retrieval.s2g_research.retrievers import RankedRetrieverAdapter


def _document():
    return RetrievalDocument(
        chunk_id="C1",
        document_id="D1",
        text="Câu bằng chứng được chọn. Câu không được chọn.",
        document_title="Synthetic",
        source_file="synthetic.pdf",
        page_start=1,
        page_end=1,
        source_spans=(),
        chapter=None,
        section=None,
        article=None,
        clause=None,
        point=None,
        unit_type="clause",
        structure_source="processed_structure",
        retrieval_role="content",
    )


class RankedBackend:
    def retrieve(self, query, top_k):
        return [{"chunk_id": "C1", "rank": 1, "score": 2.0}][:top_k]


def test_ranked_retriever_adapter_has_common_contract():
    batch = RankedRetrieverAdapter("bm25", RankedBackend(), [_document()]).retrieve(
        "query", top_k=6, turn_index=0
    )
    assert batch.mode == "bm25"
    assert batch.query == "query"
    assert [row["chunk_id"] for row in batch.items] == ["C1"]
    assert batch.evidence["C1"].source_text == _document().text


class CaptureGenerator:
    def __init__(self):
        self.evidence = None

    def generate(self, question, evidence):
        self.evidence = evidence
        return {"answer": "ok"}


def test_answer_reasoner_receives_selected_sentences_not_full_parent_text():
    document = _document()
    evidence_item = SimpleNamespace(
        chunk_id=document.chunk_id,
        source_text=document.text,
        source_document_identifier=document.document_id,
        page_start=document.page_start,
        page_end=document.page_end,
        source_spans=document.source_spans,
    )
    selected = segment_chunk(evidence_item)[:1]
    generator = CaptureGenerator()
    chunks = {
        "C1": {
            "document_title": "Synthetic",
            "page_start": 1,
            "page_end": 1,
            "retrieval_role": "content",
            "quality_flags": [],
        }
    }
    EvidenceOnlyAnswerReasoner(generator, chunks).reason("question", selected)

    assert generator.evidence[0]["selected_sentences"] == ["Câu bằng chứng được chọn."]
    assert generator.evidence[0]["source_text"] == "Câu bằng chứng được chọn."
    assert "Câu không được chọn" not in generator.evidence[0]["source_text"]
    assert generator.evidence[0]["context_prefix"] is None


def test_ranked_retriever_rejects_unknown_chunk():
    class BadBackend:
        def retrieve(self, query, top_k):
            return [{"chunk_id": "UNKNOWN", "rank": 1}]

    with pytest.raises(ValueError, match="unknown chunk IDs"):
        RankedRetrieverAdapter("dense", BadBackend(), [_document()]).retrieve(
            "query", top_k=6, turn_index=0
        )
