from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.product_s2g.answer import AnswerGenerator
from src.product_s2g.config import ProductConfig
from src.product_s2g.corpus import FullCorpusCatalog, sha256_file
from src.product_s2g.service import S2GChatService
from src.retrieval.s2g_research.models import S2GResearchResult, S2GResearchStopReason


@pytest.fixture(scope="module")
def config():
    return ProductConfig.load()


@pytest.fixture(scope="module")
def catalog(config):
    return FullCorpusCatalog.load(config)


def test_full_corpus_indexes_every_nonempty_chunk_and_all_pdfs(config, catalog):
    before = sha256_file(config.corpus_path)
    assert len(catalog.documents) == 3482
    assert len(catalog.chunks) == 3482
    assert len(catalog.document_ids) == 27
    assert len(catalog.source_files) == 27
    assert sum(catalog.document_pages.values()) == 339
    assert set(catalog.role_counts) == {"content", "review", "context_only"}
    assert sum(catalog.role_counts.values()) == 3482
    assert all(path.is_file() for path in catalog.source_files.values())
    assert sha256_file(config.corpus_path) == before


def test_unavailable_pages_are_explicit_and_not_invented(catalog):
    assert {(row["page"], row["reason"]) for row in catalog.unavailable_pages} == {
        (4, "empty_source_page"),
        (76, "empty_source_page"),
    }


class _Responses:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _evidence(chunk_id="C1"):
    return [{
        "chunk_id": chunk_id, "document_title": "Synthetic", "page_start": 1,
        "page_end": 1, "retrieval_role": "content", "quality_flags": [],
        "context_prefix": "Document: Synthetic", "source_text": "Nội dung thử nghiệm.",
        "selected_sentences": ["Nội dung thử nghiệm."],
    }]


def test_answer_generator_requires_grounded_dynamic_citations(config):
    response = SimpleNamespace(
        status="completed", model="gpt-5-nano-2025-08-07", id="synthetic-answer",
        output_text=json.dumps({"answer": "Nội dung thử nghiệm [1].", "citations": ["C1"], "insufficient": False, "confidence": "high"}),
        usage={"input_tokens": 10, "output_tokens": 5},
    )
    client = SimpleNamespace(responses=_Responses(response))
    result = AnswerGenerator(client, config.answer, sleeper=lambda _: None).generate("Câu hỏi?", _evidence())
    assert result.citation_chunk_ids == ("C1",)
    assert client.responses.calls[0]["reasoning"] == {"effort": "low"}
    prompt = client.responses.calls[0]["input"]
    assert "parent_source_text" not in prompt
    assert "context_prefix" not in prompt
    assert "selected_sentences" in prompt
    assert "Tuyệt đối không biến nghĩa vụ của cán bộ coi thi" in prompt
    schema = client.responses.calls[0]["text"]["format"]["schema"]
    assert schema["properties"]["citations"]["items"]["enum"] == ["C1"]


def test_answer_generator_rejects_hallucinated_citation(config):
    response = SimpleNamespace(
        status="completed", model="gpt-5-nano-2025-08-07", id="bad",
        output_text='{"answer":"bad","citations":["C999"],"insufficient":false,"confidence":"high"}', usage=None,
    )
    generator = AnswerGenerator(SimpleNamespace(responses=_Responses(response)), config.answer, sleeper=lambda _: None)
    with pytest.raises(Exception, match="invalid citations"):
        generator.generate("Câu hỏi?", _evidence())


class _FakeController:
    def __init__(self, chunk_id, sentence):
        self.chunk_id = chunk_id
        self.sentence = sentence

    def run(self, query_id, question):
        return S2GResearchResult(
            method_id="synthetic", query_id=query_id, original_question=question,
            query_history=[question], turns=[],
            evidence_context=[{
                "sentence_id": "S1", "parent_chunk_id": self.chunk_id,
                "source_document_identifier": "doc", "page_start": 1, "page_end": 1,
                "parent_source_spans": (), "char_start": 0, "char_end": len(self.sentence), "text": self.sentence,
            }],
            final_parent_chunk_ids=[self.chunk_id], stop_reason=S2GResearchStopReason.SUFFICIENT,
            closed=True,
            budget={
                "judge_calls": 2, "retrieval_calls": 1, "extractor_calls": 1,
                "answer_calls": 1, "presented_chunks": 6, "selected_sentences": 1,
            },
            answer={
                "answer": "Câu trả lời có căn cứ.",
                "citation_chunk_ids": (self.chunk_id,),
                "insufficient": False,
                "confidence": "high",
                "provider": {"provider_called": False},
            },
        )


def test_end_to_end_service_contract_with_fake_models(tmp_path, config, catalog):
    chunk = next(row for row in catalog.chunks.values() if row["retrieval_role"] == "content")
    local_config = replace(config, output_root=tmp_path / "product")
    service = S2GChatService(
        local_config, catalog, _FakeController(chunk["chunk_id"], chunk["source_text"][:120]),
        {"synthetic": True},
    )
    response = service.chat("Câu hỏi thử nghiệm", request_id="test-request")
    assert response["answer"] == "Câu trả lời có căn cứ."
    assert response["citations"][0]["chunk_id"] == chunk["chunk_id"]
    assert response["coverage_notice"]["source_pdfs"] == 27
    assert response["coverage_notice"]["indexed_chunks"] == 3482
    assert (tmp_path / "product/runtime/test-request/result.json").is_file()
