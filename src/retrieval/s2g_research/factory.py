"""Build the research S2G pipeline from the existing full-corpus product assets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.product_s2g.answer import AnswerGenerator
from src.product_s2g.config import ProductConfig
from src.product_s2g.corpus import FullCorpusCatalog
from src.product_s2g.openai_runtime import build_product_s2g_caller
from src.product_s2g.retrieval import build_full_corpus_backbone
from src.product_s2g.semantic_wrappers import S2GJudgeValidationWrapper, S2GSelectorValidationWrapper

from .config import S2GResearchConfig
from .openai_adapter import OpenAIResearchEvidenceExtractor, OpenAIResearchJudge
from .pipeline import S2GResearchPipeline
from .reasoner import EvidenceOnlyAnswerReasoner
from .retrievers import BackboneRetrieverAdapter, RankedRetrieverAdapter


@dataclass(frozen=True)
class ResearchPipelineComponents:
    pipeline: S2GResearchPipeline
    catalog: FullCorpusCatalog
    component_info: dict[str, Any]


def build_research_pipeline(
    research_config: S2GResearchConfig | None = None,
    product_config: ProductConfig | None = None,
) -> ResearchPipelineComponents:
    research_config = research_config or S2GResearchConfig.load()
    product_config = product_config or ProductConfig.load()
    if research_config.corpus_path.resolve() != product_config.corpus_path.resolve():
        raise ValueError("Research and product configurations reference different corpora")

    catalog = FullCorpusCatalog.load(product_config)
    backbone, _reranker, component_info = build_full_corpus_backbone(catalog, product_config)
    documents = list(catalog.documents)
    if research_config.retrieval_mode == "bm25":
        retriever = RankedRetrieverAdapter("bm25", backbone.bm25, documents)
    elif research_config.retrieval_mode == "dense":
        retriever = RankedRetrieverAdapter("dense", backbone.dense, documents)
    else:
        retriever = BackboneRetrieverAdapter(backbone, mode=research_config.retrieval_mode)

    caller = build_product_s2g_caller()
    judge = S2GJudgeValidationWrapper(OpenAIResearchJudge(caller))
    extractor = S2GSelectorValidationWrapper(OpenAIResearchEvidenceExtractor(caller))
    answer_generator = AnswerGenerator(caller.client, product_config.answer)
    reasoner = EvidenceOnlyAnswerReasoner(answer_generator, catalog.chunks)
    pipeline = S2GResearchPipeline(retriever, judge, extractor, reasoner, research_config)
    return ResearchPipelineComponents(
        pipeline=pipeline,
        catalog=catalog,
        component_info={
            **component_info,
            "research_method_id": research_config.method_id,
            "retrieval_mode": research_config.retrieval_mode,
            "presented_top_k": research_config.top_k,
            "max_turns": research_config.max_turns,
            "judge_first": True,
        },
    )
