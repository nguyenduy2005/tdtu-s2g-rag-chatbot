"""End-to-end orchestration: full-corpus retrieval, S2G, answer, citations."""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from typing import Any
from urllib.parse import quote

from src.s2g_runtime.attempt_audit import controller_attempt_audit_scope
from src.product_s2g.io import atomic_json
from src.retrieval.s2g_research.config import S2GResearchConfig
from src.retrieval.s2g_research.factory import build_research_pipeline
from src.retrieval.s2g_research.models import S2GResearchStopReason
from src.retrieval.s2g_research.pipeline import S2GResearchPipeline

from .config import ProductConfig
from .corpus import FullCorpusCatalog


class ProductRuntimeError(RuntimeError):
    pass


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)[:100]


class S2GChatService:
    def __init__(
        self,
        config: ProductConfig,
        catalog: FullCorpusCatalog,
        controller: S2GResearchPipeline,
        component_info: dict[str, Any],
    ) -> None:
        self.config = config
        self.catalog = catalog
        self.controller = controller
        self.component_info = component_info

    @classmethod
    def build(cls, config: ProductConfig | None = None) -> "S2GChatService":
        config = config or ProductConfig.load()
        research_config = S2GResearchConfig.load(config.research_controller_config)
        components = build_research_pipeline(research_config, config)
        components.catalog.write_manifest(config)
        return cls(config, components.catalog, components.pipeline, components.component_info)

    def chat(self, question: str, *, request_id: str | None = None) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("Câu hỏi không được để trống")
        if len(question) > 2000:
            raise ValueError("Câu hỏi vượt quá 2.000 ký tự")
        request_id = _safe_id(request_id or uuid.uuid4().hex)
        runtime_root = self.config.output_root / "runtime" / request_id
        if runtime_root.exists():
            raise FileExistsError(f"Request ID đã tồn tại: {request_id}")
        runtime_root.mkdir(parents=True, exist_ok=False)
        with controller_attempt_audit_scope(
            runtime_root / "controller_attempts",
            self.config.product_id,
            "s2g_research",
            request_id,
        ):
            result = self.controller.run(request_id, question)
        if result.error or result.stop_reason is S2GResearchStopReason.ERROR:
            failure = {"request_id": request_id, "question": question, "s2g": result.core_dict()}
            atomic_json(runtime_root / "failure.json", failure)
            raise ProductRuntimeError(f"S2G không hoàn thành: {result.error}")

        if not isinstance(result.answer, dict):
            raise ProductRuntimeError("Answer Reasoner returned an invalid result")
        final_ids = list(result.final_parent_chunk_ids)
        sentences_by_parent: defaultdict[str, list[str]] = defaultdict(list)
        for sentence in result.evidence_context:
            if sentence["parent_chunk_id"] in final_ids:
                sentences_by_parent[sentence["parent_chunk_id"]].append(sentence["text"])
        evidence = []
        for chunk_id in final_ids:
            row = self.catalog.chunks[chunk_id]
            evidence.append({
                "chunk_id": chunk_id,
                "document_id": row["document_id"],
                "document_title": row["document_title"],
                "source_file": row["source_file"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
                "retrieval_role": row["retrieval_role"],
                "quality_flags": list(row.get("quality_flags") or ()),
                "context_prefix": row.get("context_prefix"),
                "selected_sentences": sentences_by_parent[chunk_id],
                "source_text": row["source_text"],
                "unit_type": row["unit_type"],
                "structure_source": row["structure_source"],
            })
        by_id = {row["chunk_id"]: row for row in evidence}
        citation_ids = tuple(result.answer.get("citation_chunk_ids") or ())
        if any(chunk_id not in by_id for chunk_id in citation_ids):
            raise ProductRuntimeError("Answer Reasoner returned a citation outside Evidence Context")
        citations = []
        for chunk_id in citation_ids:
            row = by_id[chunk_id]
            low_trust = row["retrieval_role"] != "content" or bool(
                {"ocr_table_alignment_ambiguous", "table_relation_insufficient_v1_1", "structure_alignment_low_confidence"}
                & set(row["quality_flags"])
            )
            citations.append({
                "chunk_id": chunk_id,
                "document_id": row["document_id"],
                "document_title": row["document_title"],
                "source_file": row["source_file"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
                "retrieval_role": row["retrieval_role"],
                "quality_flags": row["quality_flags"],
                "low_trust": low_trust,
                "excerpt": " ".join(row["selected_sentences"])[:900],
                "pdf_url": f"/api/documents/{quote(row['document_id'], safe='')}/pdf#page={row['page_start']}",
            })
        response = {
            "request_id": request_id,
            "answer": str(result.answer["answer"]),
            "insufficient": bool(result.answer["insufficient"]),
            "confidence": str(result.answer["confidence"]),
            "citations": citations,
            "retrieval": {
                "method": "S2G-RAG judge-first — full-readable-corpus deployment",
                "rounds": result.budget["retrieval_calls"],
                "stop_reason": result.stop_reason.value,
                "queries": result.query_history,
                "selected_sentence_count": len(result.evidence_context),
                "final_parent_chunk_count": len(final_ids),
            },
            "coverage_notice": {
                "source_pdfs": 27,
                "indexed_chunks": len(self.catalog.documents),
                "unavailable_pages": list(self.catalog.unavailable_pages),
            },
        }
        audit = {
            "response": response,
            "answer_provider": result.answer.get("provider"),
            "s2g_core": result.core_dict(),
            "final_evidence": evidence,
        }
        atomic_json(runtime_root / "result.json", audit)
        return response

    def document_summary(self) -> list[dict[str, Any]]:
        counts: defaultdict[str, int] = defaultdict(int)
        roles: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in self.catalog.chunks.values():
            counts[row["document_id"]] += 1
            roles[row["document_id"]][row["retrieval_role"]] += 1
        unavailable_by_doc: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.catalog.unavailable_pages:
            unavailable_by_doc[row["document_id"]].append(row)
        return [
            {
                "document_id": document_id,
                "title": next(row["document_title"] for row in self.catalog.chunks.values() if row["document_id"] == document_id),
                "source_file": self.catalog.source_files[document_id].name,
                "chunk_count": counts[document_id],
                "role_counts": dict(roles[document_id]),
                "unavailable_pages": unavailable_by_doc[document_id],
            }
            for document_id in self.catalog.document_ids
        ]
