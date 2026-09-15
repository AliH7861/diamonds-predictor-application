"""Main orchestration from natural-language question to grounded answer."""

from dataclasses import asdict

import pandas as pd

from .clarification import build_buying_plan, parse_model_inputs
from .memory import PreferenceMemory
from .prompt_builder import build_evidence_payload, build_generation_prompt
from .retrieval import KnowledgeRetriever
from .routing import route_question
from .schemas import DiamondQueryPlan
from .similarity_search import StructuredSimilaritySearch


class DiamondAssistant:
    """Run each assistant stage in visible execution order."""

    def __init__(
        self,
        llm,
        catalog,
        stores,
        enricher=None,
        similarity=None,
        top_diamonds=5,
        knowledge_per_query=2,
        memory_limit=4,
    ):
        self.llm = llm
        self.catalog = catalog
        self.stores = stores
        self.enricher = enricher
        self.similarity = similarity or StructuredSimilaritySearch(catalog.diamonds)
        self.top_diamonds = top_diamonds
        self.retriever = KnowledgeRetriever(stores, llm, knowledge_per_query)
        self.memory = PreferenceMemory(stores, llm, memory_limit)

    @staticmethod
    def _conversation_text(conversation: list[dict] | None) -> str:
        """Pass at most eight recent visible turns and 4,000 characters onward."""
        if not conversation:
            return "No earlier conversation."
        text = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}"
            for item in conversation[-8:]
        )
        return text[-4000:]

    @staticmethod
    def _reference_from_conversation(conversation: list[dict] | None):
        """Reuse the latest displayed diamond as the structured similarity reference."""
        for message in reversed(conversation or []):
            result = message.get("result") or {}
            for key in ("similar_matches", "matches"):
                frame = result.get(key)
                if isinstance(frame, pd.DataFrame) and not frame.empty:
                    return frame.iloc[0]
                if isinstance(frame, list) and frame:
                    return pd.Series(frame[0])
        return None

    def _empty_result(self, plan: DiamondQueryPlan, route, answer: str, memory: list[str]) -> dict:
        """Return the same response shape while waiting for clarification."""
        empty = self.catalog.diamonds.head(0).copy()
        trace = [
            {"stage": "routing", "result": asdict(route)},
            {"stage": "memory", "result": memory},
            {"stage": "structured_plan", "result": asdict(plan)},
            {"stage": "clarification", "result": answer},
        ]
        return {
            "status": "needs_clarification",
            "answer": answer,
            "matches": empty,
            "similar_matches": empty,
            "route": asdict(route),
            "plan": asdict(plan),
            "initial_queries": [],
            "needed_second_retrieval": False,
            "extra_queries": [],
            "knowledge": [],
            "knowledge_details": [],
            "retrieved_memory": memory,
            "saved_memory": None,
            "evidence": {},
            "trace": trace,
        }

    def ask(
        self,
        question: str,
        conversation: list[dict] | None = None,
        on_token=None,
    ) -> dict:
        """Route, clarify, retrieve only needed evidence, compact it, and answer."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must contain text.")

        conversation_text = self._conversation_text(conversation)
        route = route_question(question, conversation_text)
        model_inputs = {}
        if route.intent in {"price_prediction", "clarity_prediction"}:
            model_inputs, missing = parse_model_inputs(
                f"{conversation_text}\n{question}", route.intent
            )
            if missing:
                labels = {"x": "x dimension", "y": "y dimension", "z": "z dimension"}
                requested = ", ".join(labels.get(item, item) for item in missing)
                return self._empty_result(
                    DiamondQueryPlan(), route,
                    f"To run the saved ANN, please provide: {requested}.", []
                )
        buying_plan = build_buying_plan(question, conversation_text) if route.use_dataset else None

        # Clarification precedes memory lookup, embeddings, dataset search, and inference.
        if buying_plan is not None and buying_plan.needs_clarification:
            answer = buying_plan.clarifying_question or (
                "What budget, carat size, and quality factor matter most to you?"
            )
            return self._empty_result(buying_plan, route, answer, [])

        memory = self.memory.retrieve(question) if route.use_memory else []
        if route.intent in {"price_prediction", "clarity_prediction"}:
            plan = DiamondQueryPlan(knowledge_queries=[question])
        elif buying_plan is not None:
            plan = buying_plan
        else:
            plan_data = self.llm.structured(
                "Return JSON for a diamond search plan. Set search_dataset false for factual "
                "questions and create semantic knowledge_queries that match the question.",
                f"Previous preferences: {memory}\nQuestion: {question}",
            )
            plan = DiamondQueryPlan.from_dict(plan_data)
        plan.search_dataset = bool(route.use_dataset and plan.search_dataset)
        if not plan.search_dataset:
            # A language model may invent search filters while answering a factual
            # question. Remove them so the developer trace reflects the route that
            # the deterministic application code will actually execute.
            for field_name in (
                "min_price", "max_price", "target_carat", "depth", "table",
                "x", "y", "z", "cut", "color", "clarity",
            ):
                setattr(plan, field_name, None)

        matches = (
            self.catalog.search(plan, self.top_diamonds)
            if route.use_dataset else self.catalog.diamonds.head(0).copy()
        )

        similar = self.catalog.diamonds.head(0).copy()
        if route.use_similarity:
            reference = self._reference_from_conversation(conversation)
            if reference is None and not matches.empty:
                reference = matches.iloc[0]
            if reference is None:
                return self._empty_result(
                    plan,
                    route,
                    "Which diamond or earlier recommendation should I use as the comparison point?",
                    memory,
                )
            similar = self.similarity.find(reference, plan, question, self.top_diamonds)

        # Saved ANNs are loaded at startup and used only for routed inference.
        model_evidence = {}
        if (
            route.intent in {"price_prediction", "clarity_prediction"}
            and self.enricher is not None
        ):
            model_evidence = self.enricher.predict(route.intent, model_inputs)
        if route.use_models and self.enricher is not None:
            if not matches.empty:
                matches = self.enricher.enrich(matches)
            if not similar.empty:
                similar = self.enricher.enrich(similar)

        retrieval = {
            "details": [], "initial_queries": [], "extra_queries": [],
            "needed_second_retrieval": False,
        }
        if route.use_knowledge:
            retrieval = self.retriever.retrieve(question, plan.knowledge_queries)

        evidence = build_evidence_payload(
            route, plan, matches, similar, retrieval["details"], memory, model_evidence
        )
        prompt = build_generation_prompt(question, conversation_text, evidence)
        system_prompt = (
            "Answer as a concise diamond adviser. Use only the supplied compact evidence. "
            "Explain relevant trade-offs, name supplied knowledge sources when useful, and "
            "label saved-model outputs as estimates."
        )
        if on_token is None:
            answer = self.llm.complete(system_prompt, prompt)
        else:
            answer = self.llm.complete(system_prompt, prompt, on_token=on_token)
        saved_memory = self.memory.save_explicit(f"{conversation_text}\n{question}")
        model_status = (
            self.enricher.status()
            if self.enricher is not None and hasattr(self.enricher, "status")
            else {"enabled": self.enricher is not None}
        )
        trace = [
            {"stage": "routing", "result": asdict(route)},
            {"stage": "clarification", "result": "complete"},
            {"stage": "structured_plan", "result": asdict(plan)},
            {"stage": "dataset_search", "result": len(matches)},
            {"stage": "similarity_search", "result": len(similar)},
            {"stage": "knowledge_retrieval", "result": retrieval["details"]},
            {"stage": "model_evidence", "result": model_status},
            {"stage": "memory", "result": memory},
            {"stage": "compact_evidence", "result": evidence},
            {"stage": "generation", "result": "completed"},
        ]
        return {
            "status": "answered",
            "answer": answer,
            "matches": matches,
            "similar_matches": similar,
            "route": asdict(route),
            "plan": asdict(plan),
            "initial_queries": retrieval["initial_queries"],
            "needed_second_retrieval": retrieval["needed_second_retrieval"],
            "extra_queries": retrieval["extra_queries"],
            "knowledge": [item["document"] for item in retrieval["details"]],
            "knowledge_details": retrieval["details"],
            "retrieved_memory": memory,
            "saved_memory": saved_memory,
            "evidence": evidence,
            "trace": trace,
        }
