from __future__ import annotations

from dataclasses import replace

from src.s2g_runtime.runtime_types import EvidenceItem, ModelCallRecord, ModelResult
from src.retrieval.s2g.schemas import GapItem, JudgeDecision, SelectorDecision
from src.retrieval.s2g_research.config import S2GResearchConfig
from src.retrieval.s2g_research.models import RetrievalBatch, S2GResearchStopReason
from src.retrieval.s2g_research.pipeline import S2GResearchPipeline
from src.retrieval.s2g_research.trace import validate_trace


def _call(stage: str, value):
    return ModelResult(
        value=value,
        call=ModelCallRecord(
            stage=stage,
            raw_response="{}",
            parsed_response={},
            response_id=f"synthetic-{stage}",
            model_name="synthetic",
            model_version="v1",
        ),
    )


def _gap(index: int) -> GapItem:
    return GapItem("attribute", f"target-{index}", f"slot-{index}", f"missing-{index}")


class FakeJudge:
    def __init__(self, decisions, events):
        self.decisions = list(decisions)
        self.events = events

    def judge(self, question, context):
        self.events.append(("judge", len(context)))
        return _call("s2g_judge", self.decisions.pop(0))


class FakeExtractor:
    def __init__(self, events, *, empty=False):
        self.events = events
        self.empty = empty

    def select(self, question, gaps, sentence_pool):
        self.events.append(("extract", len(sentence_pool), len(gaps)))
        pointers = () if self.empty else (sentence_pool[0].sentence_id,)
        return _call("s2g_sentence_selector", SelectorDecision(pointers))


class FakeRetriever:
    def __init__(self, events, *, empty=False, repeated=False, failure=None):
        self.events = events
        self.empty = empty
        self.repeated = repeated
        self.failure = failure
        self.calls = []

    def retrieve(self, query, top_k, turn_index):
        self.events.append(("retrieve", turn_index, top_k))
        self.calls.append((query, top_k, turn_index))
        if self.failure:
            raise self.failure
        if self.empty:
            return RetrievalBatch("hybrid_rrf_reranker", query, turn_index, (), {}, {})
        suffix = 0 if self.repeated else turn_index
        chunk_id = f"C{suffix}"
        text = "Quy định thử nghiệm có đầy đủ nội dung."
        item = {"chunk_id": chunk_id, "rank": 1, "score": 1.0}
        evidence = EvidenceItem(chunk_id, text, "DOC", 1, 1, ())
        return RetrievalBatch(
            "hybrid_rrf_reranker",
            query,
            turn_index,
            (item,),
            {chunk_id: evidence},
            {"synthetic": True},
        )


class FakeReasoner:
    def __init__(self, events):
        self.events = events
        self.contexts = []

    def reason(self, question, context):
        self.events.append(("reason", len(context)))
        self.contexts.append(context)
        return {"answer": "synthetic", "evidence_count": len(context)}


def _config():
    return S2GResearchConfig.load()


def _pipeline(decisions, *, retriever=None, extractor=None):
    events = []
    retriever = retriever or FakeRetriever(events)
    extractor = extractor or FakeExtractor(events)
    reasoner = FakeReasoner(events)
    return (
        S2GResearchPipeline(retriever, FakeJudge(decisions, events), extractor, reasoner, _config()),
        events,
        retriever,
        reasoner,
    )


def test_configuration_freezes_required_s2g_architecture():
    config = _config()
    assert config.max_turns == 4
    assert config.top_k == 6
    assert config.max_gaps_for_query == 1
    assert config.retrieval_mode == "hybrid_rrf_reranker"


def test_judge_runs_on_empty_context_before_first_retrieval():
    pipeline, events, retriever, reasoner = _pipeline([
        JudgeDecision(False, (_gap(0),)),
        JudgeDecision(True, ()),
    ])
    result = pipeline.run("Q1", "Câu hỏi gốc?")

    assert events == [
        ("judge", 0),
        ("retrieve", 0, 6),
        ("extract", 1, 1),
        ("judge", 1),
        ("reason", 1),
    ]
    assert result.stop_reason is S2GResearchStopReason.SUFFICIENT
    assert result.budget == {
        "judge_calls": 2,
        "retrieval_calls": 1,
        "extractor_calls": 1,
        "answer_calls": 1,
        "presented_chunks": 1,
        "selected_sentences": 1,
    }
    assert len(retriever.calls) == 1
    assert len(reasoner.contexts[0]) == 1
    assert result.turns[0]["evidence_context_before"] == []
    assert result.turns[1]["evidence_context_before"] == result.turns[0]["evidence_context_after"]
    validate_trace(result)


def test_max_turns_means_four_retrievals_and_final_judge():
    decisions = [JudgeDecision(False, (_gap(index),)) for index in range(5)]
    pipeline, events, retriever, _ = _pipeline(decisions)
    result = pipeline.run("Q2", "Câu hỏi nhiều bước?")

    assert result.stop_reason is S2GResearchStopReason.MAX_TURNS
    assert result.closed is True
    assert result.budget["retrieval_calls"] == 4
    assert result.budget["extractor_calls"] == 4
    assert result.budget["judge_calls"] == 5
    assert result.budget["answer_calls"] == 1
    assert all(call[1] == 6 for call in retriever.calls)
    validate_trace(result)


def test_repeated_pointer_is_not_duplicated_in_accumulated_context():
    events = []
    retriever = FakeRetriever(events, repeated=True)
    extractor = FakeExtractor(events)
    pipeline = S2GResearchPipeline(
        retriever,
        FakeJudge([
            JudgeDecision(False, (_gap(0),)),
            JudgeDecision(False, (_gap(1),)),
        ], events),
        extractor,
        FakeReasoner(events),
        _config(),
    )
    result = pipeline.run("Q3", "Câu hỏi?")

    assert result.stop_reason is S2GResearchStopReason.NO_NEW_EVIDENCE
    assert len(result.evidence_context) == 1
    assert result.budget["retrieval_calls"] == 2
    validate_trace(result)


def test_empty_retrieval_is_a_semantic_stop_not_an_infrastructure_error():
    events = []
    result = S2GResearchPipeline(
        FakeRetriever(events, empty=True),
        FakeJudge([JudgeDecision(False, (_gap(0),))], events),
        FakeExtractor(events),
        FakeReasoner(events),
        _config(),
    ).run("Q4", "Câu hỏi?")

    assert result.stop_reason is S2GResearchStopReason.EMPTY_RETRIEVAL
    assert result.closed is True
    assert result.error is None
    assert result.budget["extractor_calls"] == 0
    validate_trace(result)


def test_retriever_failure_stays_separate_and_skips_answer_reasoner():
    events = []
    result = S2GResearchPipeline(
        FakeRetriever(events, failure=ConnectionError("network unavailable")),
        FakeJudge([JudgeDecision(False, (_gap(0),))], events),
        FakeExtractor(events),
        FakeReasoner(events),
        _config(),
    ).run("Q5", "Câu hỏi?")

    assert result.stop_reason is S2GResearchStopReason.ERROR
    assert result.closed is False
    assert result.error["category"] == "ConnectionError"
    assert result.budget["answer_calls"] == 0


def test_invalid_judge_contract_hard_fails_without_retrieval():
    events = []
    result = S2GResearchPipeline(
        FakeRetriever(events),
        FakeJudge([{"not": "a judge result"}], events),
        FakeExtractor(events),
        FakeReasoner(events),
        _config(),
    ).run("Q6", "Câu hỏi?")

    assert result.stop_reason is S2GResearchStopReason.ERROR
    assert result.error["category"] == "TypeError"
    assert result.budget["retrieval_calls"] == 0


def test_duplicate_gap_query_stops_without_repeating_retrieval():
    gap = _gap(0)
    pipeline, _, retriever, _ = _pipeline([
        JudgeDecision(False, (gap,)),
        JudgeDecision(False, (gap,)),
    ])
    result = pipeline.run("Q7", "Câu hỏi?")

    assert result.stop_reason is S2GResearchStopReason.NO_VALID_NEW_QUERY
    assert result.budget["retrieval_calls"] == 1
    assert len(retriever.calls) == 1
    validate_trace(result)
