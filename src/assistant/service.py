"""Simple orchestration for search, comparison, analysis, knowledge, and prediction."""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
import re
from time import perf_counter
from uuid import uuid4

import pandas as pd

from src.clustering.rag import answer_profile_question

from .clarification import SemanticUnderstanding, parse_model_inputs
from .comparison import compare_rows, displayed_rows
from .data_analysis import execute_analysis
from .domain_answers import answer_domain_catalog
from .prompt_builder import build_evidence_payload, build_generation_prompt, compact_rows
from .retrieval import KnowledgeRetriever
from .routing import route_question
from .schemas import DiamondQueryPlan
from .search_planning import STATE_FIELDS, build_search_plan
from .tool_executor import execute_model_plan, format_model_answer


LOGGER = logging.getLogger(__name__)


class DiamondAssistant:
    """Run one visible evidence path for each of the seven supported intents."""

    def __init__(
        self,
        llm,
        catalog,
        stores,
        enricher=None,
        top_diamonds=5,
        knowledge_per_query=2,
        memory_limit=4,
        profile_registry=None,
    ):
        self.llm = llm
        self.catalog = catalog
        self.stores = stores
        self.enricher = enricher
        self.top_diamonds = max(1, int(top_diamonds))
        self.retriever = KnowledgeRetriever(stores, llm, knowledge_per_query)
        self.profile_registry = list(profile_registry or [])
        self.understanding = SemanticUnderstanding(llm)

    @staticmethod
    def _conversation_text(conversation: list[dict] | None) -> str:
        if not conversation:
            return "No earlier conversation."
        return "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}" for item in conversation[-4:]
        )[-2400:]

    @staticmethod
    def _state_text(state: dict | None) -> str:
        clean = {
            key: (state or {}).get(key)
            for key in (*STATE_FIELDS, "last_result_ids")
            if (state or {}).get(key) not in (None, [], "")
        }
        if not clean:
            return "No active search preferences."
        return "Saved search preferences: " + json.dumps(clean, default=str)

    @staticmethod
    def _recover_search_state(conversation: list[dict] | None, state: dict | None) -> dict:
        """Rebuild basic search state when a frontend forgot to echo conversation_state."""
        if state:
            return dict(state)
        recovered: dict = {}
        for item in (conversation or [])[-10:]:
            if not isinstance(item, dict) or item.get("role") != "user":
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            route = route_question(content, state=recovered)
            if route.intent != "search":
                continue
            interpreted = build_search_plan(content, recovered)
            recovered = dict(interpreted.state)
        return recovered

    def _base_result(
        self,
        *,
        status: str,
        answer: str,
        route,
        state: dict | None = None,
        matches: pd.DataFrame | None = None,
        plan: DiamondQueryPlan | None = None,
        evidence: dict | None = None,
        knowledge_details: list[dict] | None = None,
        debug: dict | None = None,
        prompt: str | None = None,
        llm_calls: int = 0,
        embedding_calls: int = 0,
        tools: list[str] | None = None,
    ) -> dict:
        frame = matches if matches is not None else self.catalog.diamonds.head(0).copy()
        details = list(knowledge_details or [])
        return {
            "status": status,
            "answer": answer,
            "matches": frame,
            "similar_matches": self.catalog.diamonds.head(0).copy(),
            "route": asdict(route),
            "plan": asdict(plan or DiamondQueryPlan()),
            "initial_queries": list((debug or {}).get("rag_queries", [])),
            "needed_second_retrieval": False,
            "extra_queries": [],
            "knowledge": [item.get("document", "") for item in details],
            "knowledge_details": details,
            "retrieved_memory": [],
            "saved_memory": None,
            "evidence": evidence or {},
            "control": (debug or {}).get("control", {}),
            "trace": (debug or {}).get("trace", []),
            "conversation_state": dict(state or {}),
            "_debug": debug or {},
            "_generation_prompt": prompt,
            "_llm_calls": llm_calls,
            "_embedding_calls": embedding_calls,
            "_tools": list(tools or []),
        }

    def ask(
        self,
        question: str,
        conversation: list[dict] | None = None,
        on_token=None,
        state: dict | None = None,
    ) -> dict:
        """Answer through the production path and attach an eight-stage trace."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must contain text.")
        request_id = str(uuid4())
        started = perf_counter()
        result = self._ask(question.strip(), conversation or [], on_token, state or {})
        debug = result.pop("_debug", {})
        prompt = result.pop("_generation_prompt", None)
        llm_calls = result.pop("_llm_calls", 0)
        embedding_calls = result.pop("_embedding_calls", 0)
        tools = result.pop("_tools", [])
        route = result["route"]
        diagnostics = {
            "request_id": request_id,
            "question": question,
            "route": route,
            "semantic_understanding": debug.get("semantic_understanding", {}),
            "raw_criteria": debug.get("raw_criteria", {}),
            "validated_criteria": debug.get("accepted_criteria", {}),
            "dropped_candidates": debug.get("dropped_candidates", {}),
            "validation_problems": debug.get("problems", []),
            "previous_state": debug.get("previous_state", dict(state or {})),
            "resulting_state": result.get("conversation_state", {}),
            "state_changes": {
                "previous_validated_state": debug.get("previous_state", dict(state or {})),
                "resulting_state": result.get("conversation_state", {}),
            },
            "dataset_filters": debug.get("hard_filters", result.get("plan", {})),
            "dataset_qualifying_rows": debug.get("qualifying_rows", 0),
            "dataset_ranking_strategy": debug.get("ranking_strategy"),
            "selected_result_ids": debug.get("selected_result_ids", []),
            "dataset_preview": compact_rows(result.get("matches", pd.DataFrame())),
            "rag_queries": debug.get("rag_queries", []),
            "rag_chunks": result.get("knowledge_details", []),
            "evidence_payload": result.get("evidence", {}),
            "generation_prompt": prompt,
            "final_answer": result.get("answer"),
            "tools_requested": tools,
            "tools_executed": tools,
            "tools_skipped": [],
            "answer_sources": result.get("evidence", {}).get("answer_sources", {}),
            "model_versions": result.get("evidence", {}).get("model_metadata", {}),
            "llm_calls": llm_calls,
            "embedding_calls": embedding_calls,
            "generation_calls": llm_calls,
            "total_ms": round((perf_counter() - started) * 1000, 3),
            "time_to_first_token_ms": None,
            "warnings": [],
            "errors": [],
            "final_status": result.get("status"),
        }
        result["request_id"] = request_id
        result["diagnostics"] = diagnostics
        result["trace"] = [
            {"stage": "question", "result": question},
            {"stage": "route", "result": route},
            {"stage": "raw_plan", "result": diagnostics["raw_criteria"]},
            {
                "stage": "validated_plan_state",
                "result": {
                    "accepted": diagnostics["validated_criteria"],
                    "dropped": diagnostics["dropped_candidates"],
                    "previous_state": diagnostics["previous_state"],
                    "resulting_state": diagnostics["resulting_state"],
                },
            },
            {
                "stage": "dataset_execution",
                "result": {
                    "filters": diagnostics["dataset_filters"],
                    "qualifying_rows": diagnostics["dataset_qualifying_rows"],
                    "ranking_strategy": diagnostics["dataset_ranking_strategy"],
                    "selected_result_ids": diagnostics["selected_result_ids"],
                    "selected_rows": diagnostics["dataset_preview"],
                },
            },
            {
                "stage": "rag",
                "result": {
                    "queries": diagnostics["rag_queries"],
                    "documents": diagnostics["rag_chunks"],
                },
            },
            {"stage": "generation_evidence", "result": diagnostics["evidence_payload"]},
            {"stage": "final_answer", "result": diagnostics["final_answer"]},
        ]
        LOGGER.info(
            "assistant_trace=%s",
            json.dumps(
                {
                    "request_id": request_id,
                    "intent": route["intent"],
                    "tools": tools,
                    "result_ids": diagnostics["selected_result_ids"],
                    "total_ms": diagnostics["total_ms"],
                }
            ),
        )
        return result

    def _ask(self, question: str, conversation: list[dict], on_token, state: dict) -> dict:
        state = self._recover_search_state(conversation, state)
        context = f"{self._state_text(state)}\n{self._conversation_text(conversation)}"
        meaning = self.understanding.interpret(question, conversation=conversation, state=state)
        route = route_question(question, context, state=state, understanding=meaning)

        if route.intent == "search":
            return self._search(question, conversation, on_token, state, route, meaning)
        if route.intent == "compare":
            return self._compare(question, conversation, on_token, state, route)
        if route.intent == "dataset_analysis":
            return self._analyze(question, conversation, on_token, state, route)
        if route.intent == "diamond_knowledge":
            return self._knowledge(question, conversation, on_token, state, route, meaning)
        if route.intent == "model_prediction":
            return self._predict(question, conversation, on_token, state, route, meaning)
        if route.intent == "small_talk":
            return self._small_talk(question, conversation, on_token, state, route)
        return self._out_of_scope(question, conversation, on_token, state, route)

    def _search(self, question, conversation, on_token, state, route, meaning):
        del conversation, on_token
        interpreted = build_search_plan(
            question,
            state,
            semantic_updates=meaning.get("search_updates") or {},
            relative_change=meaning.get("relative_change") or {},
        )
        audit = interpreted.audit(state)
        debug = {
            **audit,
            "semantic_understanding": meaning,
            "control": {
                "envelope": {
                    "candidate_criteria": audit["raw_criteria"],
                    "accepted_criteria": audit["accepted_criteria"],
                    "dropped_candidates": audit["dropped_candidates"],
                    "problems": audit["problems"],
                }
            },
        }
        if interpreted.clarification:
            return self._base_result(
                status="needs_clarification",
                answer=interpreted.clarification,
                route=route,
                state=interpreted.state,
                plan=interpreted.plan,
                debug=debug,
            )

        limit = meaning.get("result_count")
        if not isinstance(limit, int) or not 1 <= limit <= 20:
            requested = re.search(
                r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
                r"(?:different\s+)?(?:diamonds?|options?|matches?)\b",
                question,
                re.I,
            )
            numbers = {
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
                "six": 6,
                "seven": 7,
                "eight": 8,
                "nine": 9,
                "ten": 10,
            }
            limit = self.top_diamonds
            if requested:
                token = requested.group(1).casefold()
                limit = int(token) if token.isdigit() else numbers[token]
        limit = max(1, min(int(limit), 20))

        try:
            matches, execution = self.catalog.search_with_trace(
                interpreted.plan,
                limit=limit,
                strategy=interpreted.strategy,
                exclude_ids=interpreted.exclude_ids,
            )
        except TypeError:
            # Compatibility with an older catalog signature.
            matches, execution = self.catalog.search_with_trace(interpreted.plan, limit=limit)
        if not isinstance(execution, dict):
            execution = {}

        selected_ids = execution.get("selected_result_ids", [])
        if not selected_ids and not matches.empty:
            for column in ("_row_id", "id", "diamond_id", "index"):
                if column in matches.columns:
                    selected_ids = [
                        int(value)
                        for value in matches[column].tolist()
                        if str(value).lstrip("-").isdigit()
                    ]
                    break

        next_state = dict(interpreted.state)
        next_state["last_result_ids"] = selected_ids
        next_state["search_active"] = True
        next_state.pop("pending_field", None)

        if matches.empty:
            answer = (
                "No dataset rows match those constraints. Try widening the price or size "
                "range, or relaxing one quality preference."
            )
        else:
            priority = (next_state.get("priorities") or [None])[-1]
            priority_text = f" while prioritizing {priority}" if priority else ""
            answer = (
                f"I found {len(matches)} matching diamonds{priority_text}. "
                "The table shows the closest useful trade-offs within your constraints."
            )

        execution.setdefault("qualifying_rows", len(matches))
        execution.setdefault("selected_result_ids", selected_ids)
        execution.setdefault("ranking_strategy", interpreted.strategy)
        debug.update(execution)
        evidence = {
            "dataset_results": compact_rows(matches),
            "dataset_statistics": {
                "qualifying_rows": execution.get("qualifying_rows", len(matches)),
                "returned_rows": len(matches),
            },
            "current_search_state": next_state,
            "answer_sources": {"matches": "diamond_dataset"},
        }
        return self._base_result(
            status="answered",
            answer=answer,
            route=route,
            state=next_state,
            matches=matches,
            plan=interpreted.plan,
            evidence=evidence,
            debug=debug,
            tools=["dataframe"],
        )

    def _compare(self, question, conversation, on_token, state, route):
        rows = displayed_rows(conversation)
        answer, selected, evidence = compare_rows(rows, question)
        debug = {
            "previous_state": dict(state),
            "accepted_criteria": {},
            "selected_result_ids": selected.get("_row_id", pd.Series(dtype=int))
            .astype(int)
            .tolist(),
            "qualifying_rows": len(rows),
            "ranking_strategy": "displayed_row_comparison",
        }
        return self._base_result(
            status="answered" if not selected.empty else "needs_clarification",
            answer=answer,
            route=route,
            state=state,
            matches=selected.drop(columns=["_display_position"], errors="ignore"),
            evidence={**evidence, "answer_sources": {"comparison": "displayed_dataframe_rows"}},
            debug=debug,
            tools=["dataframe"],
        )

    def _analyze(self, question, conversation, on_token, state, route):
        answer, analysis = execute_analysis(self.catalog.diamonds, question, route.level_3)
        return self._base_result(
            status="answered",
            answer=answer,
            route=route,
            state=state,
            evidence={"dataset_statistics": analysis, "answer_sources": {"analysis": "pandas"}},
            debug={
                "previous_state": dict(state),
                "qualifying_rows": analysis.get("row_count", len(self.catalog.diamonds)),
                "ranking_strategy": "pandas_aggregation",
            },
            tools=["pandas"],
        )

    def _knowledge(self, question, conversation, on_token, state, route, meaning=None):
        lower = question.casefold()
        semantic_topic = str((meaning or {}).get("knowledge_topic") or "").casefold()
        topic = None
        if semantic_topic in {"cut", "cut_grades"}:
            topic = "cut"
        elif semantic_topic in {"clarity", "clarity_grades"}:
            topic = "clarity"
        elif semantic_topic in {"color", "colour", "color_grades"}:
            topic = "color"
        elif semantic_topic in {"customer_profiles", "buyer_profiles", "clustering_profiles"}:
            topic = "customer_profiles"
        elif re.search(
            r"\b(?:types?|grades?)\b.{0,20}\bcuts?\b|\bcuts?\b.{0,20}\b(?:types?|grades?)\b", lower
        ):
            topic = "cut"
        elif re.search(r"\b(?:types?|grades?)\b.{0,20}\bclarity\b", lower):
            topic = "clarity"
        elif re.search(r"\b(?:types?|grades?)\b.{0,20}\bcolors?\b", lower):
            topic = "color"
        elif route.level_3 == "CUSTOMER_PROFILES":
            topic = "customer_profiles"
        vocabulary_request = bool(
            re.search(
                r"\b(?:types?|grades?|levels?|list|different types?)\b",
                lower,
            )
        )
        fixed = (
            answer_domain_catalog(topic or "", self.profile_registry)
            if topic and (vocabulary_request or topic == "customer_profiles")
            else None
        )
        if fixed is None and route.level_3 == "CUSTOMER_PROFILES":
            fixed = answer_profile_question(question, self.profile_registry)
        if fixed is not None:
            return self._base_result(
                status="answered",
                answer=fixed,
                route=route,
                state=state,
                evidence={"answer_sources": {"explanation": "fixed_project_vocabulary"}},
                debug={"previous_state": dict(state)},
            )
        knowledge_query = (
            str((meaning or {}).get("knowledge_query") or question).strip() or question
        )
        retrieval = self.retriever.retrieve(question, [knowledge_query])
        plan = DiamondQueryPlan(knowledge_queries=retrieval["initial_queries"])
        evidence = build_evidence_payload(
            route,
            plan,
            self.catalog.diamonds.head(0),
            self.catalog.diamonds.head(0),
            retrieval["details"],
            [],
            {},
        )
        prompt = build_generation_prompt(question, self._conversation_text(conversation), evidence)
        system = (
            "Answer the diamond question in at most 100 words using only RAG KNOWLEDGE. "
            "Never invent diamond attributes, prices, model outputs, or customer profiles. "
            "Do not redefine standard clarity terms. If the evidence is insufficient, say so."
        )
        answer = (
            self.llm.complete(system, prompt)
            if on_token is None
            else self.llm.complete(system, prompt, on_token=on_token)
        )
        return self._base_result(
            status="answered",
            answer=answer,
            route=route,
            state=state,
            plan=plan,
            evidence=evidence,
            knowledge_details=retrieval["details"],
            debug={
                "previous_state": dict(state),
                "rag_queries": retrieval["initial_queries"],
            },
            prompt=prompt,
            llm_calls=1,
            embedding_calls=1,
            tools=["rag"],
        )

    def _predict(self, question, conversation, on_token, state, route, meaning=None):
        action = route.level_3
        target_map = {
            "PRICE_PREDICTION": "price_prediction",
            "CLARITY_PREDICTION": "clarity_prediction",
            "CLUSTER_PREDICTION": "cluster_prediction",
        }
        if action == "MULTI_MODEL":
            requested = []
            semantic_targets = list((meaning or {}).get("model_targets") or [])
            if "clarity" in semantic_targets:
                requested.append("classification")
            if "price" in semantic_targets:
                requested.append("regression")
            if "cluster" in semantic_targets:
                requested.append("clustering")
            reason = route.reason.casefold()
            if "clarity" in reason and "classification" not in requested:
                requested.append("classification")
            if "price" in reason and "regression" not in requested:
                requested.append("regression")
            if "cluster" in reason and "clustering" not in requested:
                requested.append("clustering")
            inputs, _ = parse_model_inputs(
                f"{self._conversation_text(conversation)}\n{question}", "classification"
            )
            package = (
                execute_model_plan(self.enricher, requested, inputs)
                if self.enricher is not None
                else {
                    "tools_executed": [],
                    "tools_skipped": [
                        {"tool": item, "reason": "model unavailable"} for item in requested
                    ],
                }
            )
            package.setdefault("rag_context", [])
            package.setdefault("dataset_results", [])
            package.setdefault("dataset_stats", {})
            return self._base_result(
                status="answered",
                answer=format_model_answer(package),
                route=route,
                state=state,
                evidence=package,
                debug={"previous_state": dict(state)},
                tools=package.get("tools_executed", []),
            )
        if action not in target_map:
            return self._base_result(
                status="needs_clarification",
                answer="Which saved model should I run: price, clarity, or cluster?",
                route=route,
                state=state,
                debug={"previous_state": dict(state)},
            )
        legacy_intent = target_map[action]
        inputs, missing = parse_model_inputs(
            f"{self._conversation_text(conversation)}\n{question}", legacy_intent
        )
        if missing:
            return self._base_result(
                status="needs_clarification",
                answer="To run the saved model, please provide: " + ", ".join(missing) + ".",
                route=route,
                state=state,
                evidence={"missing_model_inputs": missing},
                debug={"previous_state": dict(state), "raw_criteria": inputs},
            )
        model = self.enricher.predict(legacy_intent, inputs) if self.enricher is not None else {}
        matches = self.catalog.diamonds.head(0).copy()
        tools = ["saved_model"]
        if not model:
            answer = "The requested saved model is unavailable."
        elif model.get("error"):
            answer = str(model["error"])
        elif legacy_intent == "price_prediction":
            answer = f"The saved price model estimates ${float(model['predicted_price']):,.2f}."
        elif legacy_intent == "clarity_prediction":
            family = model["clarity_family"]
            confidence = float((model.get("probabilities") or {}).get(family, 0)) * 100
            answer = (
                f"The saved clarity model estimates {family} with {confidence:.1f}% confidence."
            )
        else:
            profile = model["cluster_profile"]
            answer = f"The saved clustering model assigns cluster {model['cluster_id']}: {profile['name']}. {profile['summary']}"
        compare_dataset = bool(
            re.search(
                r"\b(?:actual|observed|real)\s+dataset\b|\bcompare.+\bdataset\b", question, re.I
            )
        )
        dataset_stats = {}
        if (
            compare_dataset
            and legacy_intent == "price_prediction"
            and inputs.get("carat") is not None
        ):
            plan = DiamondQueryPlan(
                search_dataset=True,
                target_carat=float(inputs["carat"]),
                carat_tolerance=0.10,
                cut=inputs.get("cut"),
                color=inputs.get("color"),
                clarity=inputs.get("clarity"),
            )
            matches, _ = self.catalog.search_with_trace(plan, limit=self.top_diamonds)
            if not matches.empty:
                dataset_stats = {
                    "observed_price_min": float(matches["price"].min()),
                    "observed_price_median": float(matches["price"].median()),
                    "observed_price_max": float(matches["price"].max()),
                }
                answer += (
                    " Comparable dataset rows range from "
                    f"${dataset_stats['observed_price_min']:,.0f} to "
                    f"${dataset_stats['observed_price_max']:,.0f}."
                )
                tools.append("dataframe")
        answer_key = {
            "price_prediction": "predicted_price",
            "clarity_prediction": "predicted_clarity",
            "cluster_prediction": "cluster_profile",
        }.get(legacy_intent, "prediction")
        answer_source = {
            "price_prediction": "regression_model",
            "clarity_prediction": "classification_model",
            "cluster_prediction": "clustering_model",
        }.get(legacy_intent, "saved_model")
        answer_sources = {answer_key: answer_source}
        if dataset_stats:
            answer_sources["observed_price_summary"] = "dataframe"
        model_evidence = {
            "model_evidence": model,
            "dataset_results": compact_rows(matches),
            "dataset_stats": dataset_stats,
            "rag_context": [],
            "tools_executed": [
                "regression"
                if legacy_intent == "price_prediction"
                else "classification"
                if legacy_intent == "clarity_prediction"
                else "clustering",
                *(["dataframe"] if "dataframe" in tools else []),
            ],
            "answer_sources": answer_sources,
        }
        return self._base_result(
            status="answered",
            answer=answer,
            route=route,
            state=state,
            matches=matches,
            evidence=model_evidence,
            debug={"previous_state": dict(state), "accepted_criteria": inputs},
            tools=tools,
        )

    def _small_talk(self, question, conversation, on_token, state, route):
        lower = question.casefold()
        if "thank" in lower:
            answer = "You're welcome."
        elif "help" in lower or "what can you do" in lower:
            answer = (
                "I can find diamonds in natural language, refine an earlier search, compare shown "
                "results, calculate dataset statistics, explain diamond concepts, and run the saved "
                "price, clarity, or clustering models."
            )
        else:
            answer = "Hi. Tell me naturally what you want to find or understand about diamonds."
        return self._base_result(status="answered", answer=answer, route=route, state=state)

    def _out_of_scope(self, question, conversation, on_token, state, route):
        if route.level_3 == "UNAVAILABLE_FIELD":
            answer = "That information is not present in this diamond dataset."
        else:
            answer = "I can help with diamond searches, comparisons, dataset analysis, explanations, and saved-model predictions."
        return self._base_result(status="answered", answer=answer, route=route, state=state)
