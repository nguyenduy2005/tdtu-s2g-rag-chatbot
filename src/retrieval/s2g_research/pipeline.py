"""Judge-first S2G-RAG state machine with complete, GT-free turn traces."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import time
from typing import Any, Protocol

from src.s2g_runtime.runtime_types import ModelResult
from src.retrieval.s2g.gap_mapping import build_next_query, normalize_for_duplicate
from src.retrieval.s2g.schemas import GapItem, JudgeDecision, SelectorDecision
from src.retrieval.s2g.sentence_context import EvidenceContext, SentenceRecord, segment_chunk

from .config import S2GResearchConfig
from .models import RetrievalBatch, S2GResearchResult, S2GResearchStopReason
from .retrievers import S2GRetriever


class Judge(Protocol):
    def judge(self, q_0: str, context: tuple[SentenceRecord, ...]) -> ModelResult: ...


class EvidenceExtractor(Protocol):
    def select(
        self,
        q_0: str,
        gaps: tuple[GapItem, ...],
        sentence_pool: tuple[SentenceRecord, ...],
    ) -> ModelResult: ...


class AnswerReasoner(Protocol):
    def reason(self, question: str, context: tuple[SentenceRecord, ...]) -> Any: ...


def _context_rows(context: EvidenceContext) -> list[dict[str, Any]]:
    return [asdict(item) for item in context.sentences]


def _call_record(result: ModelResult) -> dict[str, Any]:
    return asdict(result.call)


def _answer_value(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (str, int, float, bool, type(None), list, dict, tuple)):
        return value
    raise TypeError("Answer Reasoner returned a non-serializable result")


class S2GResearchPipeline:
    """Execute S2G as J0, then at most four retrieve/extract/update turns.

    With ``max_turns=4``, the maximum semantic-call budget is five Judge
    calls, four Evidence Extractor calls, and one Answer Reasoner call.
    Ground-truth data is absent from every runtime interface.
    """

    def __init__(
        self,
        retriever: S2GRetriever,
        judge: Judge,
        extractor: EvidenceExtractor,
        reasoner: AnswerReasoner,
        config: S2GResearchConfig,
    ) -> None:
        self.retriever = retriever
        self.judge = judge
        self.extractor = extractor
        self.reasoner = reasoner
        self.config = config

    def run(self, query_id: str, question: str) -> S2GResearchResult:
        pipeline_started = time.monotonic()
        query_id = query_id.strip()
        question = question.strip()
        if not query_id:
            raise ValueError("query_id must be non-empty")
        if not question:
            raise ValueError("question must be non-empty")

        context = EvidenceContext(
            sentence_limit=self.config.sentence_limit,
            parent_limit=self.config.parent_limit,
        )
        query_history = [question]
        turns: list[dict[str, Any]] = []
        stop = S2GResearchStopReason.ERROR
        error: dict[str, Any] | None = None
        answer: Any = None
        budget = {
            "judge_calls": 0,
            "retrieval_calls": 0,
            "extractor_calls": 0,
            "answer_calls": 0,
            "presented_chunks": 0,
            "selected_sentences": 0,
        }

        for turn_index in range(self.config.max_turns + 1):
            trace: dict[str, Any] = {
                "turn_index": turn_index,
                "original_question": question,
                "evidence_context_before": _context_rows(context),
            }
            turns.append(trace)
            try:
                judged = self.judge.judge(question, tuple(context.sentences))
                budget["judge_calls"] += 1
                if not isinstance(judged.value, JudgeDecision):
                    raise TypeError("Judge returned an invalid decision type")
                decision = judged.value
                trace["judge"] = _call_record(judged)
                trace["sufficient"] = decision.sufficient
                trace["gap_items"] = [asdict(item) for item in decision.gaps]

                if decision.sufficient:
                    if not context.sentences:
                        raise ValueError("sufficient=true is invalid with empty Evidence Context")
                    stop = S2GResearchStopReason.SUFFICIENT
                    trace["evidence_context_after"] = _context_rows(context)
                    trace["stop_reason"] = stop.value
                    break

                if turn_index == self.config.max_turns:
                    stop = S2GResearchStopReason.MAX_TURNS
                    trace["evidence_context_after"] = _context_rows(context)
                    trace["stop_reason"] = stop.value
                    break

                active_gaps = decision.gaps[: self.config.max_gaps_for_query]
                trace["active_gap_items"] = [asdict(item) for item in active_gaps]
                retrieval_query = build_next_query(question, active_gaps)
                trace["retrieval_query"] = retrieval_query
                normalized_history = {normalize_for_duplicate(item) for item in query_history}
                if not retrieval_query or normalize_for_duplicate(retrieval_query) in normalized_history:
                    stop = S2GResearchStopReason.NO_VALID_NEW_QUERY
                    trace["evidence_context_after"] = _context_rows(context)
                    trace["stop_reason"] = stop.value
                    break
                query_history.append(retrieval_query)

                retrieval_started = time.monotonic()
                batch = self.retriever.retrieve(
                    retrieval_query,
                    top_k=self.config.top_k,
                    turn_index=turn_index,
                )
                retrieval_latency = time.monotonic() - retrieval_started
                budget["retrieval_calls"] += 1
                self._validate_batch(batch)
                budget["presented_chunks"] += len(batch.items)
                trace["retrieval"] = {
                    "mode": batch.mode,
                    "query": batch.query,
                    "top_k": self.config.top_k,
                    "items": [dict(item) for item in batch.items],
                    "audit": batch.audit,
                    "latency_seconds": retrieval_latency,
                }
                if not batch.items:
                    stop = S2GResearchStopReason.EMPTY_RETRIEVAL
                    trace["sentence_pool"] = []
                    trace["evidence_context_after"] = _context_rows(context)
                    trace["stop_reason"] = stop.value
                    break

                sentence_pool = tuple(
                    sentence
                    for item in batch.items
                    for sentence in segment_chunk(batch.evidence[item["chunk_id"]])
                )
                trace["sentence_pool"] = [asdict(item) for item in sentence_pool]
                if not sentence_pool:
                    stop = S2GResearchStopReason.NO_NEW_EVIDENCE
                    trace["evidence_context_after"] = _context_rows(context)
                    trace["stop_reason"] = stop.value
                    break

                selected = self.extractor.select(question, active_gaps, sentence_pool)
                budget["extractor_calls"] += 1
                if not isinstance(selected.value, SelectorDecision):
                    raise TypeError("Evidence Extractor returned an invalid decision type")
                admitted = context.admit(selected.value.sentence_ids, sentence_pool)
                budget["selected_sentences"] += len(admitted)
                trace["extractor"] = _call_record(selected)
                trace["selected_sentence_ids"] = list(selected.value.sentence_ids)
                trace["admitted_evidence"] = [asdict(item) for item in admitted]
                trace["evidence_context_after"] = _context_rows(context)
                if not admitted:
                    stop = S2GResearchStopReason.NO_NEW_EVIDENCE
                    trace["stop_reason"] = stop.value
                    break
            except Exception as exc:
                stop = S2GResearchStopReason.ERROR
                error = {
                    "category": type(exc).__name__,
                    "message": str(exc),
                    "turn_index": turn_index,
                }
                if getattr(exc, "audit", None) is not None:
                    error["provider_audit"] = exc.audit
                trace.setdefault("evidence_context_after", _context_rows(context))
                trace["stop_reason"] = stop.value
                trace["error"] = error
                break

        closed = stop is not S2GResearchStopReason.ERROR
        if closed:
            try:
                answer = _answer_value(self.reasoner.reason(question, tuple(context.sentences)))
                budget["answer_calls"] += 1
            except Exception as exc:
                stop = S2GResearchStopReason.ERROR
                closed = False
                error = {
                    "category": type(exc).__name__,
                    "message": str(exc),
                    "turn_index": len(turns) - 1,
                    "phase": "answer_reasoner",
                }

        return S2GResearchResult(
            method_id=self.config.method_id,
            query_id=query_id,
            original_question=question,
            query_history=query_history,
            turns=turns,
            evidence_context=_context_rows(context),
            final_parent_chunk_ids=list(context.parent_chunk_ids),
            stop_reason=stop,
            closed=closed,
            budget=budget,
            latency_seconds=time.monotonic() - pipeline_started,
            answer=answer,
            error=error,
        )

    def _validate_batch(self, batch: RetrievalBatch) -> None:
        if batch.mode != self.config.retrieval_mode:
            raise ValueError("Retriever mode differs from the frozen research configuration")
        if len(batch.items) > self.config.top_k:
            raise ValueError("Retriever returned more than top_k items")
        identifiers = tuple(item.get("chunk_id") for item in batch.items)
        if any(not isinstance(item, str) or not item for item in identifiers):
            raise ValueError("Every retrieved item must contain a chunk_id")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Retriever returned duplicate chunk IDs")
        if set(batch.evidence) != set(identifiers):
            raise ValueError("Retrieved items and evidence mapping differ")
        if batch.query.strip() == "":
            raise ValueError("Retriever audit query is empty")
