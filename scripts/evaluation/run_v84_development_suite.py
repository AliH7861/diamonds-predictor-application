"""Run the frozen 880-case V8.4 development evaluation and save reports."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROJECT_ROOT, RAW_DATA_PATH
from src.assistant.clarification import build_buying_plan, parse_model_inputs
from src.assistant.clarity import clarity_family
from src.assistant.legacy.concepts import concepts_as_dicts
from src.assistant.legacy.control_plane import prepare_request
from src.assistant.data_analysis import execute_analysis
from src.assistant.dataset_search import DiamondCatalog
from src.assistant.legacy.execution_policy import execution_policy
from src.assistant.generation import OllamaClient
from src.assistant.model_evidence import ModelEvidenceProvider
from src.assistant.retrieval import KnowledgeRetriever
from src.assistant.routing import route_question
from src.assistant.tool_executor import execute_model_plan, format_model_answer
from src.assistant.vector_store import ChromaStores


BENCHMARK_DIR = PROJECT_ROOT / "benchmarks" / "v84_development"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "v84_development"

ACTION_ROUTE = {
    "search": ("DATA_TASK", "RETRIEVE", "SEARCH"),
    "stats": ("DATA_TASK", "ANALYZE", "STATS"),
    "trend_group": ("DATA_TASK", "ANALYZE", "TREND_GROUP"),
    "trend_relation": ("DATA_TASK", "ANALYZE", "TREND_RELATION"),
    "rank": ("DATA_TASK", "RETRIEVE", "RANK"),
    "compare": ("DATA_TASK", "COMPARE", "COMPARE_DIAMONDS"),
    "clarify": ("CONVERSATION_CONTROL", None, None),
    "rag": ("KNOWLEDGE_TASK", "EXPLAIN", "RAG_EXPLANATION"),
    "reject_or_unavailable": ("OUT_OF_SCOPE", None, None),
    "price_range": ("DATA_TASK", "PRICE", "PRICE_RANGE"),
    "price_estimate": ("DATA_TASK", "PRICE", "PRICE_ESTIMATE"),
    "mixed": ("DATA_TASK", "TOOL_PLAN", "MULTI_TASK"),
}

ACTION_TOOLS = {
    "search": ["dataframe"],
    "stats": ["dataframe"],
    "trend_group": ["dataframe"],
    "trend_relation": ["dataframe"],
    "rank": ["dataframe"],
    "compare": ["dataframe"],
    "rag": ["rag"],
    "price_range": ["dataframe"],
    "price_estimate": ["dataframe"],
    "mixed": ["dataframe"],
}

ROOT_CAUSE = {
    "concept": "concept_mapping_failure",
    "field": "field_evidence_failure",
    "state": "state_update_failure",
    "l1": "L1_route_failure",
    "l2": "L2_route_failure",
    "l3": "L3_route_failure",
    "tool_selection": "tool_selection_failure",
    "tool_execution": "tool_execution_failure",
    "grounding": "grounding_failure",
    "exception": "exception",
}


def _expected_concepts(case: dict) -> set[str]:
    action = case.get("expected_behavior") or case.get("expected_action")
    concepts = set()
    if action == "search" and not (case.get("conversation_id") and int(case.get("turn") or 1) > 1):
        concepts.add("SEARCH_FILTER")
    if action == "rank":
        concepts.add("RANK_RECOMMENDATION")
    if action == "stats":
        concepts.add("STATISTIC")
    if action in {"trend_group", "trend_relation"}:
        concepts.add("TREND" if action == "trend_group" else "RELATIONSHIP")
    if action == "compare":
        concepts.add("COMPARE")
    prompt = case["prompt"].casefold()
    criteria = case.get("criteria") or {}
    if re.search(r"\b(?:price|cost)\b|\$", prompt):
        concepts.add("PRICE")
    if (
        "clarity" in criteria
        or "clarity" in prompt
        or any(x in prompt for x in ("vs1", "vs2", "vvs", "si1"))
    ):
        concepts.add("CLARITY")
    if (
        any(key in criteria for key in ("budget_max", "min_price", "max_price"))
        or "budget" in prompt
    ):
        concepts.add("BUDGET")
    if any(key in criteria for key in ("carat_target", "carat_min", "carat_max")) or re.search(
        r"\b(?:carat|ct)\b", prompt
    ):
        concepts.add("CARAT_SIZE")
    return concepts


def _model_tools(route) -> list[str]:
    if route.intent == "price_prediction":
        return ["regression"]
    if route.intent == "clarity_prediction":
        return ["classification"]
    if route.intent == "cluster_prediction":
        return ["clustering"]
    if route.intent == "multi_model":
        return [
            name for name in ("classification", "regression", "clustering") if name in route.reason
        ]
    return []


def _expected_route(case: dict) -> tuple[str | None, str | None, str | None]:
    behavior = case.get("expected_behavior") or case.get("expected_action")
    if behavior in {"model", "provenance", "multi_model", "missing_features"} and case.get(
        "expected_tools"
    ):
        models = [
            x for x in case["expected_tools"] if x in {"classification", "regression", "clustering"}
        ]
        if len(models) > 1:
            return "DATA_TASK", "TOOL_PLAN", "MULTI_TASK"
        if models:
            level = {
                "classification": "CLARITY_PREDICTION",
                "regression": "PRICE_PREDICTION",
                "clustering": "CLUSTER_PREDICTION",
            }[models[0]]
            return "DATA_TASK", "MODEL_ANALYSIS", level
    if behavior == "out_of_scope":
        return "OUT_OF_SCOPE", None, None
    if behavior == "conversation":
        return "CONVERSATION_CONTROL", "CHAT", "SMALL_TALK"
    return ACTION_ROUTE.get(
        behavior, ACTION_ROUTE.get(case.get("expected_action"), (None, None, None))
    )


def _plan_for_case(case: dict, route, state: dict, conversation: str):
    plan = (
        build_buying_plan(
            case["prompt"],
            conversation,
            require_recommendation_details=route.level_2 not in {"ANALYZE", "PRICE"},
        )
        if route.use_dataset
        else None
    )
    return prepare_request(case["prompt"], route, plan, state)


def _field_accuracy(case: dict, audit: dict) -> bool | None:
    criteria = case.get("criteria") or {}
    if not criteria:
        return None
    accepted = audit.get("envelope", {}).get("accepted_criteria", {})
    aliases = {
        "budget_max": "max_price",
        "carat_target": "target_carat",
        "clarity_family": "clarity",
        "priority": "priorities",
    }
    checks = []
    for key, value in criteria.items():
        if key in {"diamond_ids", "comparison_focus", "clarity_or_better", "color_or_better"}:
            continue
        if key in {"carat_min", "carat_max"}:
            low, high = criteria.get("carat_min"), criteria.get("carat_max")
            target, tolerance = accepted.get("target_carat"), accepted.get("carat_tolerance")
            if (
                low is not None
                and high is not None
                and target is not None
                and tolerance is not None
            ):
                checks.append(
                    abs((target - tolerance) - low) < 0.011
                    and abs((target + tolerance) - high) < 0.011
                )
            elif key == "carat_min":
                checks.append(False)
            continue
        actual_key = aliases.get(key, key)
        if actual_key not in accepted:
            checks.append(False)
            continue
        actual = accepted[actual_key]
        if actual_key == "clarity":
            value = clarity_family(str(value))
        if actual_key == "priorities" and isinstance(actual, list):
            checks.append(str(value).casefold() in {str(item).casefold() for item in actual})
            continue
        checks.append(
            str(actual).casefold() == str(value).casefold()
            or (
                isinstance(actual, (int, float))
                and isinstance(value, (int, float))
                and abs(actual - value) < 0.011
            )
        )
    return all(checks) if checks else None


def _clean_state(state: dict) -> dict:
    return {
        key: value
        for key, value in state.items()
        if not key.startswith("_") and key != "pending_buying"
    }


def _semantic_state(case: dict, before: dict, after: dict, audit: dict) -> tuple[bool | None, dict]:
    """Score state meaning field by field instead of transaction completion."""
    expected = case.get("expected_full_criteria")
    if not expected:
        return None, {"applicable": False, "fields": {}}
    aliases = {"budget_max": "max_price", "carat_target": "target_carat", "priority": "priorities"}
    actual = _clean_state(after)
    field_results = {}
    for source_key, expected_value in expected.items():
        if source_key.endswith("_or_better"):
            # The current five-family product state does not represent raw-grade
            # threshold flags. Record this explicitly instead of silently passing it.
            field_results[source_key] = "MISSING_CAPABILITY"
            continue
        key = aliases.get(source_key, source_key)
        actual_value = actual.get(key)
        if key == "clarity" and actual_value is not None:
            expected_value = clarity_family(str(expected_value))
        if key == "priorities" and isinstance(actual_value, list):
            equal = str(expected_value).casefold() in {str(x).casefold() for x in actual_value}
        elif isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
            equal = abs(float(expected_value) - float(actual_value)) < 0.011
        else:
            equal = str(expected_value).casefold() == str(actual_value).casefold()
        if equal:
            prior_value = _clean_state(before).get(key)
            field_results[source_key] = (
                "PRESERVED_CORRECTLY" if prior_value == actual_value else "CHANGED_CORRECTLY"
            )
        elif key not in actual:
            field_results[source_key] = "INCORRECTLY_REMOVED_OR_MISSING"
        else:
            field_results[source_key] = "INCORRECTLY_CHANGED"
    expected_keys = {aliases.get(key, key) for key in expected if not key.endswith("_or_better")}
    for delta in audit.get("state", {}).get("accepted_delta", []):
        if delta.get("operation") == "REMOVE" and delta.get("field") not in actual:
            field_results[delta["field"]] = "REMOVED_CORRECTLY"
    for key in actual.keys() - expected_keys:
        if key not in _clean_state(before) or actual[key] != _clean_state(before).get(key):
            field_results[key] = "INVENTED"
    scored = [value for value in field_results.values() if value != "MISSING_CAPABILITY"]
    correct_labels = {"PRESERVED_CORRECTLY", "CHANGED_CORRECTLY", "REMOVED_CORRECTLY"}
    return bool(scored) and all(value in correct_labels for value in scored), {
        "applicable": True,
        "expected": expected,
        "actual": actual,
        "fields": field_results,
        "accepted_delta": audit.get("state", {}).get("accepted_delta", []),
    }


def _failure_category(case: dict, failures: list[str], route_ok: bool, state_detail: dict) -> str:
    """Assign one mutually exclusive diagnosis to every failed case."""
    if not failures:
        return "PASS"
    if state_detail.get("applicable") and any(
        value == "MISSING_CAPABILITY" for value in state_detail.get("fields", {}).values()
    ):
        return "MISSING_CAPABILITY"
    if case.get("family") == "ambiguity_contradictions":
        return "AMBIGUOUS_CASE"
    if route_ok and set(failures) <= {"concept_mapping_failure"}:
        return "EVALUATOR_BUG"
    if route_ok and any(
        item in failures
        for item in (
            "field_evidence_failure",
            "state_update_failure",
            "tool_execution_failure",
            "grounding_failure",
        )
    ):
        return "DOWNSTREAM_FAILURE"
    return "REAL_PRODUCT_BUG"


def _status(rate: float, tool_rate: float) -> str:
    if tool_rate < 0.5:
        return "BLOCKED"
    if rate >= 0.90:
        return "STRONG"
    if rate >= 0.75:
        return "NEEDS_WORK"
    return "WEAK"


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    cases = json.loads((BENCHMARK_DIR / "cases.json").read_text(encoding="utf-8"))
    catalog = DiamondCatalog.from_csv(RAW_DATA_PATH)
    models = ModelEvidenceProvider(PROJECT_ROOT)
    llm = OllamaClient("qwen3.5:0.8b", "nomic-embed-text", "http://127.0.0.1:11434")
    stores = ChromaStores(PROJECT_ROOT / "vector_db" / "chroma_v2", llm)
    retriever = KnowledgeRetriever(stores, llm, 2)

    rows = []
    states: dict[str, dict] = defaultdict(dict)
    histories: dict[str, list[str]] = defaultdict(list)
    for number, case in enumerate(cases, 1):
        started = perf_counter()
        failures = []
        error = None
        family = case["family"]
        conversation_id = case.get("conversation_id") or ""
        state = states[conversation_id] if conversation_id else {}
        before_state = dict(state)
        conversation = "\n".join(histories[conversation_id][-4:]) if conversation_id else ""
        try:
            route = route_question(case["prompt"], conversation)
            plan, audit = _plan_for_case(case, route, state, conversation)
            expected_l1, expected_l2, expected_l3 = _expected_route(case)
            l1 = expected_l1 is None or route.level_1 == expected_l1
            l2 = expected_l2 is None or route.level_2 == expected_l2
            l3 = expected_l3 is None or route.level_3 == expected_l3

            expected_concepts = _expected_concepts(case)
            actual_concepts = {item["concept"] for item in concepts_as_dicts(case["prompt"])}
            concept_ok = not expected_concepts or expected_concepts.issubset(actual_concepts)
            field_ok = _field_accuracy(case, audit)
            semantic_state_ok, state_detail = _semantic_state(
                case, before_state, audit.get("resulting_state", state), audit
            )
            state_ok = semantic_state_ok if semantic_state_ok is not None else True

            expected_tools = list(
                case.get("expected_tools")
                or ACTION_TOOLS.get(
                    case.get("expected_behavior") or case.get("expected_action"), []
                )
            )
            actual_tools = []
            if route.use_dataset:
                actual_tools.append("dataframe")
            if route.use_knowledge:
                actual_tools.append("rag")
            actual_tools.extend(_model_tools(route))
            tool_selection = set(expected_tools).issubset(set(actual_tools))

            executed = []
            model_evidence = {}
            if any(
                tool in expected_tools for tool in ("classification", "regression", "clustering")
            ):
                raw, _ = parse_model_inputs(case["prompt"], "classification")
                model_evidence = execute_model_plan(
                    models,
                    [
                        tool
                        for tool in expected_tools
                        if tool in {"classification", "regression", "clustering"}
                    ],
                    raw,
                )
                executed.extend(model_evidence["tools_executed"])
                format_model_answer(model_evidence)
            if "dataframe" in expected_tools and route.use_dataset:
                if route.level_3 in {
                    "STATS",
                    "TREND_GROUP",
                    "TREND_RELATION",
                    "PRICE_RANGE",
                    "PRICE_ESTIMATE",
                }:
                    execute_analysis(catalog.diamonds, case["prompt"], route.level_3)
                elif plan is not None and not plan.needs_clarification:
                    catalog.search(plan, 5)
                executed.append("dataframe")
            rag_details = []
            if "rag" in expected_tools and route.use_knowledge:
                rag_details = retriever.retrieve(case["prompt"], [case["prompt"]])["details"]
                if rag_details:
                    executed.append("rag")
            tool_execution = set(expected_tools).issubset(set(executed)) if expected_tools else True
            grounding = True
            if "rag" in expected_tools:
                grounding = bool(rag_details and all(item.get("document") for item in rag_details))
            if any(
                tool in expected_tools for tool in ("classification", "regression", "clustering")
            ):
                grounding = grounding and all(
                    source in model_evidence.get("answer_sources", {}).values()
                    for source in (
                        {
                            "classification": "classification_model",
                            "regression": "regression_model",
                            "clustering": "clustering_model",
                        }[tool]
                        for tool in expected_tools
                        if tool in {"classification", "regression", "clustering"}
                    )
                )

            for key, ok in (
                ("concept", concept_ok),
                ("field", field_ok if field_ok is not None else True),
                ("state", state_ok),
                ("l1", l1),
                ("l2", l2),
                ("l3", l3),
                ("tool_selection", tool_selection),
                ("tool_execution", tool_execution),
                ("grounding", grounding),
            ):
                if not ok:
                    failures.append(ROOT_CAUSE[key])
            passed = not failures
            policy = execution_policy(route)
            llm_calls = policy.llm_calls if route.use_knowledge else 0
            embedding_calls = 1 if "rag" in executed else 0
            generation_calls = 0

            if conversation_id:
                states[conversation_id] = dict(audit.get("resulting_state") or state)
                histories[conversation_id].append(case["prompt"])
        except Exception as exc:  # keep evaluating and classify every failure
            passed = False
            failures.append("exception")
            error = f"{type(exc).__name__}: {exc}"
            route = route_question(case["prompt"], conversation)
            expected_l1, expected_l2, expected_l3 = _expected_route(case)
            l1 = l2 = l3 = concept_ok = state_ok = tool_selection = tool_execution = grounding = (
                False
            )
            field_ok = None
            expected_tools = case.get("expected_tools") or []
            actual_tools = executed = []
            llm_calls = embedding_calls = generation_calls = 0
            audit = {}
            state_detail = {"applicable": bool(case.get("expected_full_criteria")), "fields": {}}

        route_ok = bool(l1 and l2 and l3)
        failure_category = _failure_category(case, failures, route_ok, state_detail)
        expected_concepts = _expected_concepts(case)
        actual_concepts = {item["concept"] for item in concepts_as_dicts(case["prompt"])}

        rows.append(
            {
                "case_number": number,
                "name": case["name"],
                "family": family,
                "difficulty": case.get("difficulty"),
                "prompt": case["prompt"],
                "passed": passed,
                "root_causes": "|".join(dict.fromkeys(failures)),
                "error": error,
                "normalized_message": audit.get("normalized_message", ""),
                "concept_accuracy": concept_ok,
                "field_validation_accuracy": field_ok,
                "state_update_accuracy": state_ok,
                "expected_l1": expected_l1,
                "semantic_state_applicable": state_detail.get("applicable", False),
                "semantic_state_detail": json.dumps(state_detail, default=str),
                "previous_state": json.dumps(_clean_state(before_state), default=str),
                "candidate_delta": json.dumps(
                    audit.get("state", {}).get("candidate_delta", []), default=str
                ),
                "accepted_delta": json.dumps(
                    audit.get("state", {}).get("accepted_delta", []), default=str
                ),
                "rejected_delta": json.dumps(
                    audit.get("state", {}).get("rejected_delta", []), default=str
                ),
                "final_state": json.dumps(
                    _clean_state(audit.get("resulting_state", state)), default=str
                ),
                "expected_concepts": "|".join(sorted(expected_concepts)),
                "detected_concepts": "|".join(sorted(actual_concepts)),
                "missed_concepts": "|".join(sorted(expected_concepts - actual_concepts)),
                "added_concepts": "|".join(sorted(actual_concepts - expected_concepts)),
                "failure_category": failure_category,
                "route_evidence": route.reason,
                "conflicting_evidence": "",
                "actual_l1": route.level_1,
                "l1_accuracy": l1,
                "expected_l2": expected_l2,
                "actual_l2": route.level_2,
                "l2_accuracy": l2,
                "expected_l3": expected_l3,
                "actual_l3": route.level_3,
                "l3_accuracy": l3,
                "expected_tools": "|".join(expected_tools),
                "selected_tools": "|".join(actual_tools),
                "executed_tools": "|".join(executed),
                "tool_selection_accuracy": tool_selection,
                "tools_skipped": "|".join(sorted(set(expected_tools) - set(executed))),
                "primary_failure_stage": failures[0] if failures else "",
                "tool_execution_accuracy": tool_execution,
                "grounding_accuracy": grounding,
                "execution_path": "multi_tool"
                if len(actual_tools) > 1
                else (
                    "rag"
                    if "rag" in actual_tools
                    else (
                        "model_inference"
                        if any(
                            x in actual_tools
                            for x in ("classification", "regression", "clustering")
                        )
                        else "deterministic"
                    )
                ),
                "llm_calls": llm_calls,
                "embedding_calls": embedding_calls,
                "generation_calls": generation_calls,
                "classification_calls": int("classification" in executed),
                "regression_calls": int("regression" in executed),
                "clustering_calls": int("clustering" in executed),
                "pandas_calls": int("dataframe" in executed),
                "rag_retrieval_calls": int("rag" in executed),
                "deterministic_bypass": llm_calls == 0
                and embedding_calls == 0
                and generation_calls == 0,
                "wall_ms": round((perf_counter() - started) * 1000, 3),
            }
        )

    frame = pd.DataFrame(rows)
    summaries = []
    for family, group in frame.groupby("family", sort=False):
        rate = float(group["passed"].mean())
        tool_rate = float(group["tool_execution_accuracy"].mean())
        summaries.append(
            {
                "family": family,
                "passed": int(group["passed"].sum()),
                "total": len(group),
                "pass_rate": rate,
                "concept_accuracy": float(group["concept_accuracy"].mean()),
                "field_validation_accuracy": float(
                    group["field_validation_accuracy"].dropna().mean()
                )
                if group["field_validation_accuracy"].notna().any()
                else None,
                "state_update_accuracy": float(group["state_update_accuracy"].mean()),
                "L1": float(group["l1_accuracy"].mean()),
                "L2": float(group["l2_accuracy"].mean()),
                "L3": float(group["l3_accuracy"].mean()),
                "tool_accuracy": float(group["tool_selection_accuracy"].mean()),
                "tool_execution": tool_rate,
                "grounding": float(group["grounding_accuracy"].mean()),
                "deterministic_bypass": float(group["deterministic_bypass"].mean()),
                "LLM_per_case": float(group["llm_calls"].mean()),
                "embedding_per_case": float(group["embedding_calls"].mean()),
                "generation_per_case": float(group["generation_calls"].mean()),
                "median_ms": float(group["wall_ms"].median()),
                "status": _status(rate, tool_rate),
            }
        )
    summary = pd.DataFrame(summaries)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_DIR / "case_results.csv", index=False)
    frame.to_json(RESULTS_DIR / "case_results.json", orient="records", indent=2)
    summary.to_csv(RESULTS_DIR / "family_summary.csv", index=False)
    failures = frame.loc[~frame["passed"]]
    failures.to_csv(RESULTS_DIR / "failure_report.csv", index=False)
    causes = Counter(
        cause for text in failures["root_causes"] for cause in text.split("|") if cause
    )
    pd.DataFrame(causes.most_common(), columns=["root_cause", "count"]).to_csv(
        RESULTS_DIR / "root_cause_summary.csv", index=False
    )
    usage = (
        frame[
            [
                "family",
                "llm_calls",
                "embedding_calls",
                "generation_calls",
                "classification_calls",
                "regression_calls",
                "clustering_calls",
                "pandas_calls",
                "rag_retrieval_calls",
            ]
        ]
        .groupby("family")
        .sum()
    )
    usage.to_csv(RESULTS_DIR / "tool_usage_summary.csv")
    pd.DataFrame(
        {
            "level": ["L1", "L2", "L3"],
            "accuracy": [
                frame.l1_accuracy.mean(),
                frame.l2_accuracy.mean(),
                frame.l3_accuracy.mean(),
            ],
        }
    ).to_csv(RESULTS_DIR / "routing_level_summary.csv", index=False)
    for level in ("l1", "l2", "l3"):
        matrix = pd.crosstab(
            frame[f"expected_{level}"].fillna("<unspecified>"),
            frame[f"actual_{level}"].fillna("<none>"),
        )
        matrix.to_csv(RESULTS_DIR / f"{level}_confusion_matrix.csv")
    concept_names = sorted(
        set().union(
            *(set(str(value).split("|")) - {""} for value in frame["expected_concepts"]),
            *(set(str(value).split("|")) - {""} for value in frame["detected_concepts"]),
        )
    )
    concept_rows = []
    for concept in concept_names:
        expected_mask = (
            frame.expected_concepts.fillna("").str.split("|").apply(lambda x: concept in x)
        )
        detected_mask = (
            frame.detected_concepts.fillna("").str.split("|").apply(lambda x: concept in x)
        )
        tp = int((expected_mask & detected_mask).sum())
        fp = int((~expected_mask & detected_mask).sum())
        fn = int((expected_mask & ~detected_mask).sum())
        concept_rows.append(
            {
                "concept": concept,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
            }
        )
    pd.DataFrame(concept_rows).to_csv(RESULTS_DIR / "concept_confusion_report.csv", index=False)
    category_summary = frame.groupby("failure_category", dropna=False).size().rename("count")
    category_summary.to_csv(RESULTS_DIR / "failure_category_summary.csv")
    category_examples = (
        frame.loc[
            frame.failure_category != "PASS",
            ["failure_category", "case_number", "family", "prompt", "root_causes"],
        ]
        .groupby("failure_category", sort=False)
        .head(5)
    )
    category_examples.to_csv(RESULTS_DIR / "failure_category_examples.csv", index=False)
    state_rows = frame[frame.semantic_state_applicable]
    state_rows[
        [
            "case_number",
            "name",
            "family",
            "prompt",
            "state_update_accuracy",
            "semantic_state_detail",
        ]
    ].to_csv(RESULTS_DIR / "semantic_state_report.csv", index=False)
    model_rows = frame[
        frame[["classification_calls", "regression_calls", "clustering_calls"]].sum(axis=1) > 0
    ]
    model_rows.to_csv(RESULTS_DIR / "model_integration_summary.csv", index=False)
    overall = {
        "suite": "V8.4 development",
        "cases": len(frame),
        "passed": int(frame.passed.sum()),
        "pass_rate": float(frame.passed.mean()),
        "L1": float(frame.l1_accuracy.mean()),
        "L2": float(frame.l2_accuracy.mean()),
        "L3": float(frame.l3_accuracy.mean()),
        "state_update_accuracy": float(state_rows.state_update_accuracy.mean())
        if len(state_rows)
        else None,
        "validation_accuracy": float(frame.field_validation_accuracy.dropna().mean())
        if frame.field_validation_accuracy.notna().any()
        else None,
        "deterministic_bypass": float(frame.deterministic_bypass.mean()),
        "generation_scope": "Route/tool/model/retrieval evaluation; full generated-answer grading not run for all 880 cases.",
        "official_1200_modified": False,
    }
    (RESULTS_DIR / "overall_summary.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\nOVERALL", json.dumps(overall, indent=2))
    return frame, summary


if __name__ == "__main__":
    run()
