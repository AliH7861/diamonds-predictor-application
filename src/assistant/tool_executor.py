"""Execute traced multi-model plans and build one normalized evidence package."""

from __future__ import annotations

from typing import Any


MODEL_ORDER = ("classification", "regression", "clustering")


def execute_model_plan(provider, requested: list[str], raw_inputs: dict[str, Any]) -> dict:
    """Run requested model tools in dependency order without inventing inputs."""
    ordered = [tool for tool in MODEL_ORDER if tool in requested]
    working_inputs = dict(raw_inputs)
    outputs: dict[str, dict] = {}
    traces: list[dict] = []
    executed: list[str] = []
    skipped: list[dict] = []

    for tool in ordered:
        # A regression estimate may serve as the price input to clustering when
        # both tools were explicitly requested. The trace records this dependency.
        if tool == "clustering" and "price" not in working_inputs:
            regression = outputs.get("regression")
            if regression and regression.get("prediction") is not None:
                working_inputs["price"] = regression["prediction"]

        output, trace = provider.execute(tool, working_inputs)
        trace["execution_order"] = len(traces) + 1
        if tool == "clustering" and "price" not in raw_inputs and "price" in working_inputs:
            trace["resolved_dependencies"] = {
                "price": "regression.prediction",
            }
        traces.append(trace)
        if output:
            outputs[tool] = output
            executed.append(tool)
        else:
            skipped.append(
                {
                    "tool": tool,
                    "reason": trace.get("skip_reason") or trace.get("error", "unknown failure"),
                    "missing_features": trace.get("missing_features", []),
                }
            )

    answer_sources = {}
    if "classification" in outputs:
        answer_sources["predicted_clarity"] = "classification_model"
    if "regression" in outputs:
        answer_sources["predicted_price"] = "regression_model"
    if "clustering" in outputs:
        answer_sources["cluster_profile"] = "clustering_model"

    return {
        "classification_output": outputs.get("classification", {}),
        "regression_output": outputs.get("regression", {}),
        "clustering_output": outputs.get("clustering", {}),
        "model_metadata": {tool: provider.metadata.get(tool, {}) for tool in ordered},
        "tool_trace": traces,
        "answer_sources": answer_sources,
        "tools_requested": ordered,
        "tools_executed": executed,
        "tools_skipped": skipped,
        "tool_execution_order": ordered,
    }


def format_model_answer(evidence: dict) -> str:
    """Render deterministic model results without an LLM inventing evidence."""
    parts = []
    classification = evidence.get("classification_output") or {}
    regression = evidence.get("regression_output") or {}
    clustering = evidence.get("clustering_output") or {}

    if classification:
        predicted = classification["predicted_class"]
        confidence = float(classification.get("class_probabilities", {}).get(predicted, 0))
        parts.append(
            f"The saved clarity classifier predicts the {predicted} family "
            f"with {confidence * 100:.1f}% model confidence."
        )
    if regression:
        parts.append(
            "The saved price regression model estimates a price of "
            f"${float(regression['prediction']):,.2f}."
        )
    if clustering:
        profile = clustering["cluster_profile"]
        typical = profile.get("typical_diamond", {})
        summary = profile.get("summary", "")
        parts.append(
            f"The diamond is assigned to cluster {clustering['cluster_id']}, "
            f"{profile.get('name', 'unnamed profile')}. {summary} "
            f"A typical diamond in this product-derived segment is about "
            f"{float(typical.get('carat', 0)):.2f} carats and "
            f"${float(typical.get('price', 0)):,.0f}."
        )
    skipped = evidence.get("tools_skipped") or []
    for item in skipped:
        missing = item.get("missing_features") or []
        if missing:
            parts.append(
                f"I could not run {item['tool']} because these required inputs are missing: "
                + ", ".join(missing)
                + "."
            )
        else:
            parts.append(f"I could not run {item['tool']}: {item['reason']}.")
    return " ".join(parts)
