"""Runtime settings for the local assistant and its evidence limits."""

import os
from dataclasses import dataclass
from pathlib import Path

from config import PROJECT_ROOT, RAW_DATA_PATH


@dataclass(frozen=True)
class AssistantSettings:
    """Centralize local model names, paths, and compact-context limits."""

    project_root: Path = PROJECT_ROOT
    data_path: Path = RAW_DATA_PATH
    chat_model: str = "qwen3.5:4b"
    embedding_model: str = "nomic-embed-text"
    ollama_url: str = "http://127.0.0.1:11434"
    top_diamonds: int = 5
    knowledge_per_query: int = 2
    memory_limit: int = 4

    @classmethod
    def from_environment(cls) -> "AssistantSettings":
        """Read optional local overrides without changing source files."""
        return cls(
            data_path=Path(os.getenv("DIAMOND_DATA_PATH", str(RAW_DATA_PATH))),
            chat_model=os.getenv("DIAMOND_CHAT_MODEL", "qwen3.5:4b"),
            embedding_model=os.getenv("DIAMOND_EMBEDDING_MODEL", "nomic-embed-text"),
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        )
