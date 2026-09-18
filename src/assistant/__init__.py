"""Public assistant types with lazy imports for the lightweight HTTP frontend."""

from .schemas import DiamondQueryPlan, EvidenceRoute, MemoryCandidate, RetrievalCheck

__all__ = [
    "DiamondAssistant",
    "DiamondCatalog",
    "DiamondQueryPlan",
    "EvidenceRoute",
    "MemoryCandidate",
    "RetrievalCheck",
]


def __getattr__(name):
    """Load backend classes only when callers explicitly request them."""
    if name == "DiamondAssistant":
        from .service import DiamondAssistant

        return DiamondAssistant
    if name == "DiamondCatalog":
        from .dataset_search import DiamondCatalog

        return DiamondCatalog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
