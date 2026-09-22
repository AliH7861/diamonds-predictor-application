"""Retrieve a compact, auditable set of relevant domain facts."""


class KnowledgeRetriever:
    """Run hybrid retrieval with deterministic topic-aware query expansion."""

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

    @staticmethod
    def _queries(question: str, planned: list[str]) -> list[str]:
        """Build at most two focused searches without another language-model call."""
        queries = [item.strip() for item in planned if item and item.strip()]
        if not queries:
            queries = [question.strip()]
        lowered = question.casefold()
        topic_queries = (
            (
                ("clarity", "inclusion", "vs", "vvs", "if"),
                "diamond clarity families inclusions grading",
            ),
            (("cut", "sparkle", "brilliance"), "diamond cut brightness sparkle grades"),
            (("color", "colour"), "diamond color grades D through J appearance"),
            (("price", "cost", "value", "budget"), "diamond price value carat trade-offs"),
            (("cluster", "segment", "buyer"), "buyer segmentation clustering method limitations"),
            (
                ("model", "ann", "xgboost", "random forest", "feature"),
                "project model evidence inputs estimates",
            ),
            (
                ("dataset", "available", "contain", "origin", "certificate"),
                "dataset fields coverage limitations",
            ),
        )
        for terms, expanded in topic_queries:
            if any(term in lowered for term in terms):
                queries.append(expanded)
                break
        return list(dict.fromkeys(queries))[:2]

    def retrieve(self, question: str, queries: list[str]) -> dict:
        """Return deduplicated scored chunks and the searches that produced them."""
        initial_queries = self._queries(question, queries)
        details = self._search(initial_queries)
        profile_request = any(
            term in question.casefold()
            for term in (
                "customer segment",
                "buyer segment",
                "customer profile",
                "buyer profile",
                "clustering",
            )
        )
        if not profile_request:
            details = [
                item
                for item in details
                if not any(
                    marker
                    in (str(item.get("source", "")) + " " + str(item.get("section", ""))).casefold()
                    for marker in ("segment", "cluster", "customer profile", "buyer profile")
                )
            ]
        return {
            "details": details[:6],
            "initial_queries": initial_queries,
            "extra_queries": [],
            "needed_second_retrieval": False,
        }
