"""Small validated objects exchanged between the LLM and application code."""

from dataclasses import dataclass, field

from .clarity import clarity_family


@dataclass(frozen=True)
class EvidenceRoute:
    """Intent and evidence sources selected before any expensive work runs."""

    intent: str
    use_dataset: bool = False
    use_similarity: bool = False
    use_knowledge: bool = False
    use_models: bool = False
    use_memory: bool = True


@dataclass
class DiamondQueryPlan:
    """Structured interpretation of a natural-language buying question."""

    search_dataset: bool = False
    min_price: float | None = None
    max_price: float | None = None
    target_price: float | None = None
    target_carat: float | None = None
    carat_tolerance: float = 0.10
    depth: float | None = None
    table: float | None = None
    x: float | None = None
    y: float | None = None
    z: float | None = None
    cut: str | None = None
    color: str | None = None
    clarity: str | None = None
    priorities: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarifying_question: str | None = None
    knowledge_queries: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict) -> "DiamondQueryPlan":
        """Keep supported fields and reject impossible LLM-produced filters."""
        allowed = cls.__dataclass_fields__
        clean = {key: item for key, item in value.items() if key in allowed}
        clean["priorities"] = list(clean.get("priorities") or [])
        clean["knowledge_queries"] = list(clean.get("knowledge_queries") or [])
        for key in (
            "min_price", "max_price", "target_price", "target_carat", "carat_tolerance",
            "depth", "table", "x", "y", "z",
        ):
            item = clean.get(key)
            if item is not None:
                try:
                    clean[key] = max(0.0, float(item))
                except (TypeError, ValueError):
                    clean[key] = None
        valid_categories = {
            "cut": {"fair", "good", "very good", "premium", "ideal"},
            "color": set("defghij"),
            "clarity": {"i", "i1", "si", "si1", "si2", "vs", "vs1", "vs2", "vvs", "vvs1", "vvs2", "if"},
        }
        for key, choices in valid_categories.items():
            item = clean.get(key)
            if item is not None and str(item).casefold() not in choices:
                clean[key] = None
        if clean.get("clarity"):
            clean["clarity"] = clarity_family(clean["clarity"])
        if clean.get("min_price") is not None and clean.get("max_price") is not None:
            if clean["min_price"] > clean["max_price"]:
                clean["min_price"], clean["max_price"] = clean["max_price"], clean["min_price"]
        return cls(**clean)


@dataclass
class RetrievalCheck:
    """Decision describing whether the first RAG search needs another pass."""

    needs_more_context: bool = False
    extra_queries: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict) -> "RetrievalCheck":
        return cls(bool(value.get("needs_more_context")), list(value.get("extra_queries") or []))


@dataclass
class MemoryCandidate:
    """An explicit, reusable preference that may be stored for later questions."""

    should_save: bool = False
    memory_text: str | None = None

    @classmethod
    def from_dict(cls, value: dict) -> "MemoryCandidate":
        text = value.get("memory_text")
        return cls(bool(value.get("should_save")), str(text).strip() if text else None)
