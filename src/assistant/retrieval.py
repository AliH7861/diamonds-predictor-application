"""Retrieve a compact set of relevant domain facts from the knowledge store."""

from .schemas import RetrievalCheck


class KnowledgeRetriever:
    """Run scored semantic retrieval and one optional focused second pass."""

    def __init__(self, stores, llm, per_query: int = 2):
        self.stores = stores
        self.llm = llm
        self.per_query = per_query

    def _search(self, queries: list[str]) -> list[dict]:
        if hasattr(self.stores, "search_knowledge_details"):
            return self.stores.search_knowledge_details(queries, limit=self.per_query)
        documents = self.stores.search_knowledge(queries, limit=self.per_query)
        return [
            {"query": query, "source": "test-store", "similarity": None, "document": text}
            for query, text in zip(queries, documents)
        ]

    def retrieve(self, question: str, queries: list[str]) -> dict:
        """Return deduplicated scored chunks and the searches that produced them."""
        initial_queries = queries or [question]
        details = self._search(initial_queries)
        check = RetrievalCheck()
        if len(details) < 2:
            check = RetrievalCheck.from_dict(self.llm.structured(
                "Return JSON with needs_more_context and at most two extra_queries.",
                f"Question: {question}\nRetrieved knowledge: {details}",
            ))
        if check.needs_more_context and check.extra_queries:
            extra = self._search(check.extra_queries[:2])
            by_document = {item["document"]: item for item in details + extra}
            details = list(by_document.values())
        return {
            "details": details[:6],
            "initial_queries": initial_queries,
            "extra_queries": check.extra_queries[:2],
            "needed_second_retrieval": check.needs_more_context,
        }
