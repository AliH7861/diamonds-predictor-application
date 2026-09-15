"""Local diamond assistant: structured search, RAG, memory, and ML enrichment."""

from .dataset_search import DiamondCatalog
from .schemas import DiamondQueryPlan, EvidenceRoute, MemoryCandidate, RetrievalCheck
from .service import DiamondAssistant

__all__ = [
    "DiamondAssistant", "DiamondCatalog", "DiamondQueryPlan", "EvidenceRoute",
    "MemoryCandidate", "RetrievalCheck",
]
