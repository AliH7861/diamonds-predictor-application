"""Retrieve and save durable user preferences separately from domain knowledge."""

from .schemas import MemoryCandidate


class PreferenceMemory:
    """Keep preference-memory decisions out of the main assistant orchestrator."""

    MARKERS = (
        "i prefer",
        "i care",
        "i prioritize",
        "i usually",
        "my budget",
        "matters most",
        "matters to me",
    )

    def __init__(self, stores, llm, limit: int = 4):
        self.stores = stores
        self.llm = llm
        self.limit = limit

    def retrieve(self, question: str) -> list[str]:
        """Return semantically related durable preferences."""
        return self.stores.search_memory(question, limit=self.limit)

    def save_explicit(self, text: str) -> str | None:
        """Save only statements that contain an explicit reusable preference."""
        if not any(marker in text.casefold() for marker in self.MARKERS):
            return None
        candidate = MemoryCandidate.from_dict(
            self.llm.structured(
                "Return JSON with should_save and memory_text. Save only an explicit reusable user preference.",
                text,
            )
        )
        if candidate.should_save and candidate.memory_text:
            self.stores.add_memory(candidate.memory_text)
            return candidate.memory_text
        return None
