from src.assistant.retrieval import KnowledgeRetriever
from src.assistant.vector_store import chunk_markdown


class RecordingStore:
    def __init__(self):
        self.queries = []

    def search_knowledge_details(self, queries, limit):
        self.queries.extend(queries)
        return [{
            "query": queries[0],
            "source": "02_quality_grades.md",
            "section": "Clarity",
            "similarity": 0.8,
            "retrieval_score": 0.85,
            "document": "Diamond Quality Grades > Clarity\nVS describes clarity.",
        }]


class NoPlanningLLM:
    def structured(self, *_args, **_kwargs):
        raise AssertionError("Retrieval should not make a planning LLM call.")


def test_markdown_chunks_retain_title_and_section():
    chunks = chunk_markdown(
        "# Diamond Quality\n\n## Clarity\n\nVS is a clarity family."
    )
    assert chunks == [{
        "title": "Diamond Quality",
        "section": "Clarity",
        "document": "Diamond Quality > Clarity\nVS is a clarity family.",
    }]


def test_retrieval_expands_topic_without_second_llm_call():
    store = RecordingStore()
    retriever = KnowledgeRetriever(store, NoPlanningLLM(), per_query=2)
    result = retriever.retrieve("What does VS clarity mean?", [])

    assert store.queries == [
        "What does VS clarity mean?",
        "diamond clarity families inclusions grading",
    ]
    assert result["details"][0]["section"] == "Clarity"
    assert not result["needed_second_retrieval"]
