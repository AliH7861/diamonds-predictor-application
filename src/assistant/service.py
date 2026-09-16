"""Main orchestration from natural-language question to grounded answer."""

from dataclasses import asdict

import pandas as pd

from .clarification import build_buying_plan, normalize_user_text, parse_model_inputs
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
        """Pass only the latest exchange; durable constraints live in compact state."""
        if not conversation:
            return "No earlier conversation."
        text = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}"
            for item in conversation[-2:]
        )
        return text[-1600:]

    @staticmethod
    def _state_text(state: dict | None) -> str:
        """Express compact structured preferences for deterministic parsing."""
        if not state:
            return "No saved search preferences."
        parts = []
        if state.get("max_price") is not None:
            parts.append(f"maximum budget ${state['max_price']}")
        if state.get("target_price") is not None:
            parts.append(f"target price ${state['target_price']}")
        target = state.get("target_carat")
        tolerance = state.get("carat_tolerance")
        if target is not None and tolerance:
            parts.append(
                f"between {max(0, target - tolerance):.4g} and "
                f"{target + tolerance:.4g} carats"
            )
        elif target is not None:
            parts.append(f"{target} carats")
        for field in ("cut", "color", "clarity"):
            if state.get(field):
                parts.append(f"{field} {state[field]}")
        if state.get("priorities"):
            priorities = state["priorities"]
            parts.append(f"{priorities[0]} matters most")
            parts.extend(f"additional priority {item}" for item in priorities[1:])
        for field in ("clarity", "cut", "color"):
            if state.get(f"no_{field}_preference"):
                parts.append(f"I don't care about {field}")
        return "Saved search preferences: " + "; ".join(parts) + "."

    @staticmethod
    def _next_state(plan: DiamondQueryPlan, state: dict | None, question: str) -> dict:
        """Store only reusable filters instead of repeatedly sending full chat history."""
        result = dict(state or {})
        for field in (
            "min_price", "max_price", "target_price", "target_carat", "carat_tolerance",
            "depth", "table", "x", "y", "z", "cut", "color", "clarity",
            "priorities",
        ):
            result[field] = getattr(plan, field)
        current = normalize_user_text(question).casefold().replace("’", "'")
        for field in ("clarity", "cut", "color"):
            removes = (
                f"don't care about {field}" in current
                or f"dont care about {field}" in current
                or f"remove the {field}" in current
                or f"remove {field}" in current
                or f"forget {field}" in current
            )
            if removes:
                result[f"no_{field}_preference"] = True
            elif getattr(plan, field):
                result[f"no_{field}_preference"] = False
        result["pending_buying"] = plan.needs_clarification
        return result

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

    def _empty_result(
        self,
        plan: DiamondQueryPlan,
        route,
        answer: str,
        memory: list[str],
        state: dict | None = None,
    ) -> dict:
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
            "conversation_state": state or {},
        }

    def ask(
        self,
        question: str,
        conversation: list[dict] | None = None,
        on_token=None,
        state: dict | None = None,
    ) -> dict:
        """Route, clarify, retrieve only needed evidence, compact it, and answer."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must contain text.")

        conversation_text = self._conversation_text(conversation)
        state_text = self._state_text(state)
        context_text = f"{state_text}\n{conversation_text}"
        route = route_question(question, context_text)
        if route.intent == "small_talk":
            current = normalize_user_text(question).casefold()
            if "thank" in current or current.strip(" .!") in {"got it", "okay thanks"}:
                answer = "You're welcome. Ask whenever you want to compare diamonds."
            elif any(phrase in current for phrase in ("help", "what can you do", "who are you", "how do you work")):
                answer = (
                    "I can find diamonds by budget, size, cut, color, or clarity; compare "
                    "real dataset examples; explain quality trade-offs; and use the saved "
                    "models for price or clarity estimates."
                )
            else:
                answer = (
                    "Hi. Tell me your budget or the diamond qualities that matter to you, "
                    "and I can suggest matching options."
                )
            empty = self.catalog.diamonds.head(0).copy()
            plan = DiamondQueryPlan()
            return {
                "status": "answered", "answer": answer, "matches": empty,
                "similar_matches": empty, "route": asdict(route), "plan": asdict(plan),
                "initial_queries": [], "needed_second_retrieval": False,
                "extra_queries": [], "knowledge": [], "knowledge_details": [],
                "retrieved_memory": [], "saved_memory": None, "evidence": {},
                "trace": [
                    {"stage": "routing", "result": asdict(route)},
                    {"stage": "generation", "result": "skipped; conversational response"},
                ],
                "conversation_state": dict(state or {}),
            }
        if route.intent == "unsupported_dataset_field":
            empty = self.catalog.diamonds.head(0).copy()
            answer = (
                "The dataset does not contain certification, mine origin, inclusion type, "
                "customer demographics, sales channel, or sale date, so this project cannot "
                "determine that information."
            )
            return {
                "status": "answered", "answer": answer, "matches": empty,
                "similar_matches": empty, "route": asdict(route),
                "plan": asdict(DiamondQueryPlan()), "initial_queries": [],
                "needed_second_retrieval": False, "extra_queries": [],
                "knowledge": [], "knowledge_details": [], "retrieved_memory": [],
                "saved_memory": None, "evidence": {},
                "trace": [
                    {"stage": "routing", "result": asdict(route)},
                    {"stage": "generation", "result": "skipped; field unavailable"},
                ],
                "conversation_state": dict(state or {}),
            }
        model_inputs = {}
        if route.intent in {"price_prediction", "clarity_prediction"}:
            model_inputs, missing = parse_model_inputs(
                f"{context_text}\n{question}", route.intent
            )
            if missing:
                labels = {"x": "x dimension", "y": "y dimension", "z": "z dimension"}
                requested = ", ".join(labels.get(item, item) for item in missing)
                return self._empty_result(
                    DiamondQueryPlan(), route,
                    f"To run the saved ANN, please provide: {requested}.", [], state
                )
        buying_plan = (
            build_buying_plan(
                question,
                context_text,
                require_recommendation_details=route.intent != "dataset_count",
            )
            if route.use_dataset else None
        )
        next_state = (
            self._next_state(buying_plan, state, question)
            if buying_plan is not None else dict(state or {})
        )

        # Clarification precedes memory lookup, embeddings, dataset search, and inference.
        if buying_plan is not None and buying_plan.needs_clarification:
            answer = buying_plan.clarifying_question or (
                "What budget, carat size, and quality factor matter most to you?"
            )
            return self._empty_result(buying_plan, route, answer, [], next_state)

        # Exact count questions are answered by Pandas alone. They do not need
        # embeddings or generative inference.
        if route.intent == "dataset_count" and buying_plan is not None:
            all_matches = self.catalog.search(buying_plan, limit=None)
            matches = all_matches.head(self.top_diamonds).copy()
            count = len(all_matches)
            answer = f"There are {count:,} matching diamonds in the dataset."
            empty = self.catalog.diamonds.head(0).copy()
            trace = [
                {"stage": "routing", "result": asdict(route)},
                {"stage": "structured_plan", "result": asdict(buying_plan)},
                {"stage": "dataset_count", "result": count},
                {"stage": "generation", "result": "skipped; deterministic answer"},
            ]
            return {
                "status": "answered", "answer": answer, "matches": matches,
                "similar_matches": empty, "route": asdict(route),
                "plan": asdict(buying_plan), "initial_queries": [],
                "needed_second_retrieval": False, "extra_queries": [],
                "knowledge": [], "knowledge_details": [], "retrieved_memory": [],
                "saved_memory": None, "evidence": {"matching_count": count},
                "trace": trace, "conversation_state": next_state,
            }

        memory = self.memory.retrieve(question) if route.use_memory else []
        if route.intent in {"price_prediction", "clarity_prediction"}:
            plan = DiamondQueryPlan(knowledge_queries=[question])
        elif buying_plan is not None:
            plan = buying_plan
        elif route.intent == "general_knowledge":
            # The user's question is already the best semantic query. Avoid a
            # separate LLM planning call before retrieval.
            plan = DiamondQueryPlan(knowledge_queries=[question])
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

        if route.intent == "recommendation" and plan.search_dataset and matches.empty:
            empty = self.catalog.diamonds.head(0).copy()
            answer = (
                "No diamonds in the dataset match all of those constraints. "
                "Change the budget, size, cut, color, or clarity requirement and I can search again."
            )
            return {
                "status": "answered", "answer": answer, "matches": empty,
                "similar_matches": empty, "route": asdict(route), "plan": asdict(plan),
                "initial_queries": [], "needed_second_retrieval": False,
                "extra_queries": [], "knowledge": [], "knowledge_details": [],
                "retrieved_memory": memory, "saved_memory": None,
                "evidence": {"matching_count": 0},
                "trace": [
                    {"stage": "routing", "result": asdict(route)},
                    {"stage": "dataset_search", "result": 0},
                    {"stage": "generation", "result": "skipped; no exact matches"},
                ],
                "conversation_state": next_state,
            }

        if (
            route.intent == "recommendation"
            and not route.use_knowledge
            and not route.use_models
        ):
            prices = matches["price"]
            carats = matches["carat"]
            answer = (
                f"I found {len(matches)} close dataset matches from "
                f"${prices.min():,.0f} to ${prices.max():,.0f}, ranging from "
                f"{carats.min():.2f} to {carats.max():.2f} carats. "
                "The closest options are described below."
            )
            empty = self.catalog.diamonds.head(0).copy()
            return {
                "status": "answered", "answer": answer, "matches": matches,
                "similar_matches": empty, "route": asdict(route), "plan": asdict(plan),
                "initial_queries": [], "needed_second_retrieval": False,
                "extra_queries": [], "knowledge": [], "knowledge_details": [],
                "retrieved_memory": memory, "saved_memory": None,
                "evidence": {"matching_count": len(matches)},
                "trace": [
                    {"stage": "routing", "result": asdict(route)},
                    {"stage": "dataset_search", "result": len(matches)},
                    {"stage": "generation", "result": "skipped; direct dataset answer"},
                ],
                "conversation_state": next_state,
            }

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
                    next_state,
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
            "Answer the current question directly before adding useful detail. Do not infer "
            "preferences that the user did not state. Explain relevant trade-offs, name "
            "supplied knowledge sources when useful, and "
            "label saved-model outputs as estimates. Describe clarity only with the five "
            "project families I, SI, VS, VVS, and IF; never display numbered subgrades."
        )
        if on_token is None:
            answer = self.llm.complete(system_prompt, prompt)
        else:
            answer = self.llm.complete(system_prompt, prompt, on_token=on_token)
        saved_memory = self.memory.save_explicit(f"{context_text}\n{question}")
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
            "conversation_state": next_state,
        }
