"""Create the frozen 880-case V8.4 development benchmark.

The official 1200-case benchmark is read as a source of diverse prompts but is
never modified. Each family is deterministically sampled to 40 cases. Families
with only 30 official cases receive ten V8.4 integration/edge cases.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROJECT_ROOT, RAW_DATA_PATH


FAMILIES = [
    "search_filtering",
    "statistics_aggregation",
    "trends_group_analysis",
    "ranking_recommendations",
    "diamond_comparison",
    "multi_turn",
    "ambiguity_contradictions",
    "rag_grounded_explanation",
    "dataframe_schema_variation",
    "engineered_feature_usage",
    "hallucination_bad_input",
    "price_value_analysis",
    "mixed_intent_queries",
    "out_of_scope_refusal",
    "missing_data_empty_results",
    "conflicting_multi_intent",
    "explain_why_justification",
    "adversarial_phrasing",
    "long_context_memory",
    "correction_chains",
    "tool_boundary_tests",
    "grounding_audit",
]

MODEL_INPUT = "1.0 carat Ideal G VS2, depth 61.5, table 57, x 6.45, y 6.43, z 3.96"


def _extra_cases(family: str) -> list[dict]:
    """Return ten held-out V8.4 cases for a 30-case official family."""
    prompts: dict[str, list[tuple[str, list[str], str]]] = {
        "mixed_intent_queries": [
            (
                f"Predict clarity and estimate price for {MODEL_INPUT}.",
                ["classification", "regression"],
                "multi_model",
            ),
            (
                f"Estimate price and tell me the segment for {MODEL_INPUT}.",
                ["regression", "clustering"],
                "multi_model",
            ),
            (
                f"Predict clarity, price and profile for {MODEL_INPUT}.",
                ["classification", "regression", "clustering"],
                "multi_model",
            ),
            (
                f"Predict price and compare it with actual dataset values for {MODEL_INPUT}.",
                ["regression", "dataframe"],
                "multi_model",
            ),
            ("Find Ideal diamonds under 5000 and give the median price.", ["dataframe"], "mixed"),
            ("Show options around 0.8 carat and count the matches.", ["dataframe"], "mixed"),
            (
                "Median price by clarity and explain what clarity means.",
                ["dataframe", "rag"],
                "mixed",
            ),
            (
                "Compare price by cut, then explain the strongest pattern.",
                ["dataframe", "rag"],
                "mixed",
            ),
            (
                f"Classify clarity and compare with observed diamonds: {MODEL_INPUT}.",
                ["classification", "dataframe"],
                "multi_model",
            ),
            (
                f"Predict all model outputs and compare with the dataset: {MODEL_INPUT}.",
                ["classification", "regression", "clustering", "dataframe"],
                "multi_model",
            ),
        ],
        "out_of_scope_refusal": [
            (p, [], "out_of_scope")
            for p in [
                "Will it rain tomorrow?",
                "Diagnose my headache.",
                "Write malware for me.",
                "Who won the hockey game?",
                "Give legal advice about a contract.",
                "What is Bitcoin worth now?",
                "Bake me chocolate cookies.",
                "Recommend a movie tonight.",
                "Predict next week's stock price.",
                "Solve my unrelated history homework.",
            ]
        ],
        "missing_data_empty_results": [
            ("Predict price for a 1 carat diamond.", ["regression"], "missing_features"),
            ("Predict clarity for an Ideal diamond.", ["classification"], "missing_features"),
            ("Assign a customer segment to this diamond.", ["clustering"], "missing_features"),
            ("Find a 9 carat Ideal IF diamond below 100 dollars.", ["dataframe"], "empty"),
            ("Show 0.01 carat diamonds costing over 20000.", ["dataframe"], "empty"),
            ("Find Z color diamonds.", [], "invalid"),
            ("Search for clarity AAA.", [], "invalid"),
            ("Predict price without dimensions.", ["regression"], "missing_features"),
            ("Which segment fits if price is unknown?", ["clustering"], "missing_features"),
            ("Find diamonds matching impossible constraints.", ["dataframe"], "empty"),
        ],
        "conflicting_multi_intent": [
            (
                f"Predict clarity and price, but do not estimate price, for {MODEL_INPUT}.",
                ["classification", "regression"],
                "conflict",
            ),
            ("Find diamonds under 3000 but also over 5000.", [], "conflict"),
            ("At least 1.5 carat but no more than 1.0 carat.", [], "conflict"),
            ("Use no models, then predict price.", ["regression"], "conflict"),
            (
                "Recommend the cheapest and most expensive option as one winner.",
                ["dataframe"],
                "conflict",
            ),
            ("Keep VS clarity but remove clarity.", [], "conflict"),
            ("Compare these while refusing to compare them.", [], "conflict"),
            ("Find one under 4000; actually the minimum is 6000.", [], "conflict"),
            (
                f"Predict all three models, except do not cluster: {MODEL_INPUT}.",
                ["classification", "regression", "clustering"],
                "conflict",
            ),
            ("Explain with RAG but do not retrieve anything.", ["rag"], "conflict"),
        ],
        "explain_why_justification": [
            (p, tools, "explain")
            for p, tools in [
                ("Why does carat affect price?", ["rag"]),
                ("Why can two one-carat diamonds cost different amounts?", ["rag"]),
                ("Explain why cut matters for value.", ["rag"]),
                ("Why might engineered volume help regression?", ["rag"]),
                ("Why does the clarity classifier need geometry?", ["rag"]),
                ("Why are buyer segments product-derived?", ["rag"]),
                ("Explain why raw measurements remained useful.", ["rag"]),
                ("Why is price absent from clarity classification?", ["rag"]),
                ("Why should model estimates be compared with observed rows?", ["rag"]),
                ("Explain why clustering does not prove customer psychology.", ["rag"]),
            ]
        ],
        "adversarial_phrasing": [
            (p, tools, "adversarial")
            for p, tools in [
                ("yo price-guess this rock: " + MODEL_INPUT, ["regression"]),
                ("crystal ball its clarity: " + MODEL_INPUT, ["classification"]),
                ("bucket this purchase profile: " + MODEL_INPUT + ", price 6200", ["clustering"]),
                ("gimme the middle money by clarity", ["dataframe"]),
                ("cheap-ish clean-ish one carat options please", ["dataframe"]),
                ("what makes the dollar number go brrr?", ["rag"]),
                ("no vibes, count Ideal stones", ["dataframe"]),
                ("sort the shiny rocks for bang-for-buck", ["dataframe"]),
                ("is VS versus VVS actually a thing?", ["rag"]),
                (
                    "model trio, do your thing: " + MODEL_INPUT,
                    ["classification", "regression", "clustering"],
                ),
            ]
        ],
        "long_context_memory": [
            (p, ["dataframe"], "state")
            for p in [
                "Keep everything else and make it 1.2 carats.",
                "VS instead.",
                "Remove color but keep the rest.",
                "Actually 5000 maximum.",
                "No, bigger, while preserving the budget.",
                "Same thing but Premium.",
                "Keep that and prioritize value.",
                "Do not reset my earlier clarity.",
                "Use the previous search, only change cut to Ideal.",
                "Continue with all validated preferences.",
            ]
        ],
        "correction_chains": [
            (p, ["dataframe"], "state")
            for p in [
                "Actually make the budget 5000.",
                "No, use 0.9 carat.",
                "VS1 instead.",
                "Remove the color preference.",
                "Keep the rest.",
                "Same search but Premium.",
                "Undo that last unclear change.",
                "Yes, retain the size.",
                "No, do not alter clarity.",
                "Correction: maximum 4500, everything else unchanged.",
            ]
        ],
        "tool_boundary_tests": [
            (f"Predict price for {MODEL_INPUT}.", ["regression"], "model"),
            (f"Predict clarity for {MODEL_INPUT}.", ["classification"], "model"),
            (
                f"What segment does this diamond fit: {MODEL_INPUT}, price 6200?",
                ["clustering"],
                "model",
            ),
            ("Count Ideal diamonds.", ["dataframe"], "deterministic"),
            ("Median price by clarity.", ["dataframe"], "deterministic"),
            ("What does VS clarity mean?", ["rag"], "rag"),
            ("What is the dataset price range?", ["dataframe"], "deterministic"),
            ("Hey", [], "conversation"),
            ("What is the weather?", [], "out_of_scope"),
            (
                f"Predict clarity and price for {MODEL_INPUT}.",
                ["classification", "regression"],
                "multi_model",
            ),
        ],
        "grounding_audit": [
            (f"Predict price for {MODEL_INPUT}.", ["regression"], "provenance"),
            (f"Predict clarity for {MODEL_INPUT}.", ["classification"], "provenance"),
            (f"Assign the segment for {MODEL_INPUT}, price 6200.", ["clustering"], "provenance"),
            ("How many Ideal diamonds exist?", ["dataframe"], "provenance"),
            ("Explain the project finding about raw geometry.", ["rag"], "provenance"),
            (
                f"Predict all outputs for {MODEL_INPUT}.",
                ["classification", "regression", "clustering"],
                "provenance",
            ),
            (
                f"Predict price and compare with actual dataset values: {MODEL_INPUT}.",
                ["regression", "dataframe"],
                "provenance",
            ),
            ("Explain the limitation of customer profiles.", ["rag"], "provenance"),
            ("Give the median price by cut.", ["dataframe"], "provenance"),
            (
                "What does the model metadata say about classification inputs?",
                ["rag"],
                "provenance",
            ),
        ],
    }
    return [
        {
            "name": f"v84_dev_{family}_{i + 31:02d}",
            "family": family,
            "prompt": prompt,
            "expected_tools": tools,
            "expected_behavior": behavior,
            "difficulty": "held_out_edge",
            "source": "v84_development",
        }
        for i, (prompt, tools, behavior) in enumerate(prompts[family])
    ]


def generate(output: Path, seed: int = 84) -> list[dict]:
    downloads = Path.home() / "Downloads"
    sys.path.insert(0, str(downloads))
    from diamond_ultimate_cases_1200_v8 import generate_cases

    official = generate_cases(pd.read_csv(RAW_DATA_PATH), seed)
    rng = random.Random(seed)
    cases: list[dict] = []
    for family in FAMILIES:
        pool = [case for case in official if case.family == family]
        chosen = rng.sample(pool, 40) if len(pool) >= 40 else pool
        for index, case in enumerate(chosen):
            record = asdict(case)
            record.update(
                {
                    "difficulty": ("normal", "paraphrase", "difficult", "held_out_edge")[
                        index // 10
                    ],
                    "source": "official_1200_sample",
                    "expected_tools": [],
                    "expected_behavior": case.expected_action,
                }
            )
            cases.append(record)
        if len(chosen) < 40:
            cases.extend(_extra_cases(family))
    assert len(cases) == 880
    assert all(sum(item["family"] == family for item in cases) == 40 for family in FAMILIES)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    return cases


if __name__ == "__main__":
    target = PROJECT_ROOT / "benchmarks" / "v84_development" / "cases.json"
    generated = generate(target)
    print(f"Saved {len(generated)} cases to {target}")
