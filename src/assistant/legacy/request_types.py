"""Typed requests passed from assistant routing to deterministic executors."""

from dataclasses import asdict, dataclass, field
from typing import Any

from ..schemas import DiamondQueryPlan, EvidenceRoute


@dataclass(frozen=True)
class SearchRequest:
    criteria: dict[str, Any]
    action: str = "search"


@dataclass(frozen=True)
class RankingRequest:
    criteria: dict[str, Any]
    priority: str | None = None
    action: str = "rank"


@dataclass(frozen=True)
class StatsRequest:
    criteria: dict[str, Any]
    action: str = "stats"


@dataclass(frozen=True)
class TrendRequest:
    criteria: dict[str, Any]
    kind: str = "group"
    action: str = "trend"


@dataclass(frozen=True)
class PriceAnalysisRequest:
    criteria: dict[str, Any]
    kind: str = "range"
    action: str = "price_analysis"


@dataclass(frozen=True)
class ComparisonRequest:
    criteria: dict[str, Any]
    focus: str = "balance"
    action: str = "compare"


@dataclass(frozen=True)
class PredictionRequest:
    criteria: dict[str, Any]
    target: str = "price"
    action: str = "predict"


@dataclass(frozen=True)
class KnowledgeRequest:
    criteria: dict[str, Any] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)
    action: str = "rag"


@dataclass(frozen=True)
class ToolPlan:
    tasks: list[dict[str, Any]]
    action: str = "multi_task"


TypedRequest = (
    SearchRequest
    | RankingRequest
    | StatsRequest
    | TrendRequest
    | PriceAnalysisRequest
    | ComparisonRequest
    | PredictionRequest
    | KnowledgeRequest
    | ToolPlan
)


def plan_criteria(plan: DiamondQueryPlan | None) -> dict[str, Any]:
    """Return only meaningful filters from a query plan."""
    if plan is None:
        return {}
    ignored = {"search_dataset", "needs_clarification", "clarifying_question", "knowledge_queries"}
    return {
        key: value
        for key, value in asdict(plan).items()
        if key not in ignored and value not in (None, "", [], {})
    }


def build_typed_request(route: EvidenceRoute, plan: DiamondQueryPlan | None) -> TypedRequest:
    """Choose one executor request from the validated route and plan."""
    criteria = plan_criteria(plan)
    if route.level_3 == "MULTI_TASK" and route.intent == "multi_model":
        tasks = []
        reason = route.reason.casefold()
        if "classification" in reason:
            tasks.append(
                {
                    "tool": "classification",
                    "inputs": criteria,
                    "dependencies": [],
                    "expected_output": "predicted clarity family and probabilities",
                    "provenance": "saved classification artifact",
                }
            )
        if "regression" in reason:
            tasks.append(
                {
                    "tool": "regression",
                    "inputs": criteria,
                    "dependencies": [],
                    "expected_output": "predicted price",
                    "provenance": "saved regression artifact",
                }
            )
        if "clustering" in reason:
            dependencies = ["regression"] if "regression" in reason else []
            tasks.append(
                {
                    "tool": "clustering",
                    "inputs": criteria,
                    "dependencies": dependencies,
                    "expected_output": "cluster id and interpreted product profile",
                    "provenance": "saved clustering artifact",
                }
            )
        if "dataframe" in reason:
            tasks.append(
                {
                    "tool": "dataframe",
                    "inputs": criteria,
                    "dependencies": [task["tool"] for task in tasks if task["tool"] != "dataframe"],
                    "expected_output": "observed similar rows and price summary",
                    "provenance": "diamond dataset",
                }
            )
        if route.use_knowledge:
            tasks.append(
                {
                    "tool": "rag",
                    "inputs": {"queries": [route.reason]},
                    "dependencies": [task["tool"] for task in tasks],
                    "expected_output": "grounded explanation of executed evidence",
                    "provenance": "retrieved project knowledge",
                }
            )
        return ToolPlan(tasks)
    if route.level_3 == "MULTI_TASK":
        second = (
            asdict(PredictionRequest(criteria, "price"))
            if "price_prediction" in route.reason
            else asdict(StatsRequest(criteria))
        )
        return ToolPlan([asdict(SearchRequest(criteria)), second])
    if route.level_3 == "RANK":
        priorities = criteria.get("priorities") or []
        return RankingRequest(criteria, priorities[0] if priorities else None)
    if route.level_3 == "STATS":
        return StatsRequest(criteria)
    if route.level_3 in {"TREND_GROUP", "TREND_RELATION"}:
        return TrendRequest(criteria, "group" if route.level_3 == "TREND_GROUP" else "relation")
    if route.level_3 in {"PRICE_RANGE", "PRICE_ESTIMATE"}:
        return PriceAnalysisRequest(criteria, route.level_3.removeprefix("PRICE_").casefold())
    if route.intent in {"recommendation", "dataset_count", "similarity_search"}:
        return SearchRequest(criteria)
    if route.intent == "comparison":
        priorities = criteria.get("priorities") or []
        return ComparisonRequest(criteria, priorities[0] if priorities else "balance")
    if route.intent in {"price_prediction", "clarity_prediction", "cluster_prediction"}:
        return PredictionRequest(criteria, route.intent.removesuffix("_prediction"))
    return KnowledgeRequest(criteria, list(plan.knowledge_queries if plan else []))
