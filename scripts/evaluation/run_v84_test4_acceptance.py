"""Run real-product V8.4 integration and warm-latency acceptance scenarios."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import PROJECT_ROOT
from src.assistant.runtime import create_assistant


DETAILS = "1.0 carat Ideal G VS2, depth 61.5, table 57, x 6.45, y 6.43, z 3.96"
SCENARIOS = [
    ("pandas_only", "How many Ideal diamonds are under 5000 dollars?", ["dataframe"], False),
    ("regression_only", f"Predict price for {DETAILS}.", ["regression"], False),
    ("classification_only", f"Predict clarity for {DETAILS}.", ["classification"], False),
    (
        "clustering_only",
        f"What segment does this diamond fit: {DETAILS}, price 6200?",
        ["clustering"],
        False,
    ),
    ("rag_only", "What does VS clarity mean?", ["rag"], True),
    (
        "pandas_regression",
        f"Predict price and compare it with actual dataset values: {DETAILS}.",
        ["regression", "dataframe"],
        False,
    ),
    (
        "classification_regression",
        f"Predict clarity and price for {DETAILS}.",
        ["classification", "regression"],
        False,
    ),
    (
        "regression_clustering",
        f"Estimate price and tell me the segment for {DETAILS}.",
        ["regression", "clustering"],
        False,
    ),
    (
        "pandas_rag",
        "Find Ideal diamonds under 5000 dollars and explain the value trade-offs.",
        ["dataframe", "rag"],
        True,
    ),
    (
        "all_models",
        f"Predict clarity, price and profile for {DETAILS}.",
        ["classification", "regression", "clustering"],
        False,
    ),
    (
        "multi_tool_explanation",
        f"Predict clarity, price and profile for {DETAILS}, then explain what the results mean.",
        ["classification", "regression", "clustering", "rag"],
        True,
    ),
]


def _actual_tools(result: dict) -> set[str]:
    evidence = result.get("evidence") or {}
    tools = set(evidence.get("tools_executed") or [])
    route = result.get("route") or {}
    if route.get("use_dataset") and not tools:
        tools.add("dataframe")
    if result.get("knowledge_details"):
        tools.add("rag")
    if evidence.get("model_evidence"):
        intent = route.get("intent")
        tools.add(
            {
                "price_prediction": "regression",
                "clarity_prediction": "classification",
                "cluster_prediction": "clustering",
            }.get(intent, "model")
        )
    return tools


def run() -> pd.DataFrame:
    cold_started = perf_counter()
    assistant = create_assistant()
    cold_ms = (perf_counter() - cold_started) * 1000
    rows = []
    for use_case, prompt, expected_tools, generates in SCENARIOS:
        repeats = 1 if generates else 4
        durations = []
        final = None
        for _ in range(repeats):
            started = perf_counter()
            final = assistant.ask(prompt)
            durations.append((perf_counter() - started) * 1000)
        actual = _actual_tools(final)
        expected = set(expected_tools)
        unnecessary = sorted(actual - expected - {"model"})
        missing = sorted(expected - actual)
        route = final.get("route") or {}
        diagnostics = final.get("diagnostics") or {}
        evidence = final.get("evidence") or {}
        sources = set((evidence.get("answer_sources") or {}).values())
        evidence_ok = True
        for tool, source in {
            "regression": "regression_model",
            "classification": "classification_model",
            "clustering": "clustering_model",
            "dataframe": "dataframe",
        }.items():
            if tool in expected and tool in {"regression", "classification", "clustering"}:
                evidence_ok = evidence_ok and source in sources
        rows.append(
            {
                "use_case": use_case,
                "tests_run": repeats,
                "passed": not missing and not unnecessary and evidence_ok,
                "routing_accuracy": route.get("level_1") in {"DATA_TASK", "KNOWLEDGE_TASK"},
                "tool_selection_accuracy": not missing and not unnecessary,
                "tool_execution_accuracy": not missing,
                "evidence_selection_accuracy": evidence_ok,
                "grounding_accuracy": bool(final.get("answer"))
                and ("rag" not in expected or bool(final.get("knowledge_details"))),
                "unnecessary_tool_rate": len(unnecessary) / max(1, len(actual)),
                "p50_latency_ms": statistics.median(durations),
                "p95_latency_ms": max(durations),
                "cold_start_ms": cold_ms,
                "warm_samples_ms": json.dumps([round(value, 3) for value in durations]),
                "time_to_first_token_ms": diagnostics.get("time_to_first_token_ms"),
                "expected_tools": "|".join(expected_tools),
                "actual_tools": "|".join(sorted(actual)),
                "missing_tools": "|".join(missing),
                "unnecessary_tools": "|".join(unnecessary),
                "route": f"{route.get('level_1')}/{route.get('level_2')}/{route.get('level_3')}",
                "request_id": final.get("request_id"),
                "status": "WORKING WELL"
                if not missing and not unnecessary and evidence_ok
                else "FAILING",
            }
        )
    frame = pd.DataFrame(rows)
    target = PROJECT_ROOT / "outputs" / "evaluation" / "v84_test4"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / "acceptance_dashboard.csv", index=False)
    frame.to_json(target / "acceptance_cases.json", orient="records", indent=2)
    summary = {
        "cold_start_ms": cold_ms,
        "passed": int(frame.passed.sum()),
        "total": len(frame),
        "pass_rate": float(frame.passed.mean()),
        "P50_ms": float(frame.p50_latency_ms.median()),
        "P95_ms": float(frame.p95_latency_ms.max()),
        "frontend_test": "separate browser acceptance",
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return frame


if __name__ == "__main__":
    run()
