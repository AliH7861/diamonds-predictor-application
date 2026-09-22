"""Wire production assistant stages together once at application startup."""

import json

from src.clustering.rag import load_profile_registry

from .config import AssistantSettings
from .dataset_search import DiamondCatalog
from .generation import OllamaClient
from .model_evidence import ModelEvidenceProvider
from .service import DiamondAssistant
from .vector_store import ChromaStores


def create_assistant(include_models: bool = True) -> DiamondAssistant:
    """Build and index the assistant once at application startup."""
    settings = AssistantSettings.from_environment()
    llm = OllamaClient(
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        base_url=settings.ollama_url,
    )
    stores = ChromaStores(settings.project_root / "vector_db" / "chroma_v2", llm)
    stores.index_knowledge(settings.project_root / "knowledge")
    records_path = settings.project_root / "models" / "clustering" / "rag_records.json"
    if records_path.exists():
        stores.index_segmentation_records(json.loads(records_path.read_text(encoding="utf-8")))
    profiles = load_profile_registry(
        settings.project_root / "models" / "clustering" / "profile_registry.json"
    )
    catalog = DiamondCatalog.from_csv(settings.data_path)
    models = ModelEvidenceProvider(settings.project_root) if include_models else None
    # Warm Qwen last so the chat model remains the most recently used Ollama
    # model when the first question arrives.
    llm.warmup()
    return DiamondAssistant(
        llm,
        catalog,
        stores,
        models,
        top_diamonds=settings.top_diamonds,
        knowledge_per_query=settings.knowledge_per_query,
        memory_limit=settings.memory_limit,
        profile_registry=profiles,
    )
