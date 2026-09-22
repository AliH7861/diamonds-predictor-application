"""Run fresh held-out semantic cases that were not used to write V8.4 fixes."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROJECT_ROOT
from src.assistant.clarification import build_buying_plan
from src.assistant.legacy.control_plane import prepare_request
from src.assistant.routing import route_question


CASES = [
    ("search_filtering", "Pull stones near 0.68 ct with a $3,200 ceiling.", "SEARCH", ""),
    ("statistics_aggregation", "Compute the 75th percentile price for VS diamonds.", "STATS", ""),
    ("trends_group_analysis", "Which color group has the highest median price?", "TREND_GROUP", ""),
    ("ranking_recommendations", "Order the best-value choices below $4,200.", "RANK", ""),
    (
        "diamond_comparison",
        "Explain the practical difference between SI and VS.",
        "DOMAIN_EXPLANATION",
        "",
    ),
    (
        "multi_turn",
        "Only raise it to 0.85 carat.",
        "SEARCH",
        "Find a 0.75 carat diamond under $4000.",
    ),
    (
        "ambiguity_contradictions",
        "I am torn between Ideal and Premium; do not pick yet.",
        "AMBIGUOUS_CHANGE",
        "",
    ),
    ("rag_grounded_explanation", "Explain how clarity grading works.", "RAG_EXPLANATION", ""),
    (
        "dataframe_schema_variation",
        "What fields are available for each diamond?",
        "RAG_EXPLANATION",
        "",
    ),
    ("engineered_feature_usage", "Return the median face area proxy.", "STATS", ""),
    ("hallucination_bad_input", "Show the average aura_score.", "UNAVAILABLE_FIELD", ""),
    ("price_value_analysis", "What prices are observed for Premium VS stones?", "PRICE_RANGE", ""),
    (
        "mixed_intent_queries",
        "Find options below $3500 and calculate their median price.",
        "MULTI_TASK",
        "",
    ),
    ("out_of_scope_refusal", "Can you diagnose this rash?", "OUT_OF_SCOPE", ""),
    (
        "missing_data_empty_results",
        "Predict a price for a one-carat diamond.",
        "PRICE_PREDICTION",
        "",
    ),
    (
        "conflicting_multi_intent",
        "Find choices and also compute their average price.",
        "MULTI_TASK",
        "",
    ),
    (
        "explain_why_justification",
        "Why can carat change a diamond's price?",
        "DOMAIN_EXPLANATION",
        "",
    ),
    ("adversarial_phrasing", "yo show me rocks below two grand", "SEARCH", ""),
    (
        "long_context_memory",
        "Keep all of that and use Ideal cut.",
        "SEARCH",
        "Find 0.7 ct VS under $3500.",
    ),
    (
        "correction_chains",
        "Correction: use $4600 and preserve the rest.",
        "SEARCH",
        "Find 0.8 ct under $4000.",
    ),
    (
        "tool_boundary_tests",
        "What would a diamond with these measurements cost?",
        "PRICE_PREDICTION",
        "",
    ),
    (
        "grounding_audit",
        "According to the project, why did raw geometry remain useful?",
        "RAG_EXPLANATION",
        "",
    ),
]


def _state_chain() -> list[dict]:
    turns = [
        ("Find around 0.72 ct under $3,800.", {"target_carat": 0.72, "max_price": 3800.0}),
        (
            "Make clarity VS2 or better.",
            {"target_carat": 0.72, "max_price": 3800.0, "clarity": "VS"},
        ),
        (
            "G color or better too.",
            {"target_carat": 0.72, "max_price": 3800.0, "clarity": "VS", "color": "G"},
        ),
        (
            "Actually use $4,400.",
            {"target_carat": 0.72, "max_price": 4400.0, "clarity": "VS", "color": "G"},
        ),
        (
            "Remove the color requirement.",
            {"target_carat": 0.72, "max_price": 4400.0, "clarity": "VS"},
        ),
        ("Move it to 0.80 carat.", {"target_carat": 0.80, "max_price": 4400.0, "clarity": "VS"}),
    ]
    state: dict = {}
    history: list[str] = []
    rows = []
    for turn, (prompt, expected) in enumerate(turns, 1):
        context = "\n".join(history)
        route = route_question(prompt, context)
        plan = build_buying_plan(prompt, context, route.intent != "dataset_count")
        _, audit = prepare_request(prompt, route, plan, state)
        state = audit["resulting_state"]
        visible = {
            key: value
            for key, value in state.items()
            if not key.startswith("_") and key != "pending_buying"
        }
        passed = all(visible.get(key) == value for key, value in expected.items())
        unexpected = set(visible) - set(expected) - {"carat_tolerance"}
        rows.append(
            {
                "turn": turn,
                "prompt": prompt,
                "passed": passed and not unexpected,
                "expected": expected,
                "actual": visible,
                "unexpected": sorted(unexpected),
            }
        )
        history.append(prompt)
    return rows


def run() -> dict:
    rows = []
    for family, prompt, expected, context in CASES:
        route = route_question(prompt, context)
        rows.append(
            {
                "family": family,
                "prompt": prompt,
                "expected_l3": expected,
                "actual_l3": route.level_3,
                "passed": route.level_3 == expected,
                "deterministic": route.source == "deterministic",
            }
        )
    frame = pd.DataFrame(rows)
    state_rows = _state_chain()
    output = PROJECT_ROOT / "outputs" / "evaluation" / "v84_hardening"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "held_out_routes.csv", index=False)
    (output / "held_out_state.json").write_text(json.dumps(state_rows, indent=2), encoding="utf-8")
    summary = {
        "route_cases": len(frame),
        "route_passed": int(frame.passed.sum()),
        "route_accuracy": float(frame.passed.mean()),
        "state_turns": len(state_rows),
        "state_passed": sum(row["passed"] for row in state_rows),
        "state_accuracy": sum(row["passed"] for row in state_rows) / len(state_rows),
        "llm_calls": 0,
        "embedding_calls": 0,
    }
    (output / "held_out_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not frame.passed.all() or not all(row["passed"] for row in state_rows):
        print(frame.loc[~frame.passed].to_string(index=False))
        print([row for row in state_rows if not row["passed"]])
        raise SystemExit(1)
    return summary


if __name__ == "__main__":
    run()
