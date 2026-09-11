"""Judge-first, research-grade S2G-RAG orchestration.

The package deliberately reuses the qualified S2G schemas and sentence
pointer implementation without changing the retired/frozen controller.
"""

from .config import S2GResearchConfig
from .models import RetrievalBatch, S2GResearchResult, S2GResearchStopReason
from .pipeline import S2GResearchPipeline

__all__ = [
    "RetrievalBatch",
    "S2GResearchConfig",
    "S2GResearchPipeline",
    "S2GResearchResult",
    "S2GResearchStopReason",
]
