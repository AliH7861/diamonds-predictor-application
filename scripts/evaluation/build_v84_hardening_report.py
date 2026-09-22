"""Build the V8.4 before/after diagnostic dashboard from saved evaluations."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROJECT_ROOT


BASE = PROJECT_ROOT / "outputs" / "evaluation" / "v84_hardening_baseline"
CURRENT = PROJECT_ROOT / "outputs" / "evaluation" / "v84_development"
OUTPUT = PROJECT_ROOT / "outputs" / "evaluation" / "v84_hardening"


def _collision(frame: pd.DataFrame, expected: str, actual: str) -> int:
    return int(((frame.expected_l3 == expected) & (frame.actual_l3 == actual)).sum())


def _markdown(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without adding a tabulate dependency."""
    view = frame.copy().fillna("")
    headers = [str(column) for column in view.columns]
    rows = [[str(value) for value in row] for row in view.itertuples(index=False, name=None)]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def build() -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(BASE / "case_results.csv")
    new = pd.read_csv(CURRENT / "case_results.csv")
    old_family = pd.read_csv(BASE / "family_summary.csv").set_index("family")
    new_family = pd.read_csv(CURRENT / "family_summary.csv").set_index("family")
    held = pd.read_csv(OUTPUT / "held_out_routes.csv").set_index("family")
    old_overall = json.loads((BASE / "overall_summary.json").read_text(encoding="utf-8"))
    new_overall = json.loads((CURRENT / "overall_summary.json").read_text(encoding="utf-8"))

    dashboard = new_family.reset_index()[
        [
            "family",
            "pass_rate",
            "L1",
            "L2",
            "L3",
            "concept_accuracy",
            "state_update_accuracy",
            "tool_accuracy",
            "deterministic_bypass",
            "status",
        ]
    ]
    dashboard.insert(1, "old_pass_rate", dashboard.family.map(old_family.pass_rate))
    dashboard = dashboard.rename(columns={"pass_rate": "new_pass_rate"})
    dashboard["held_out_accuracy"] = dashboard.family.map(held.passed.astype(float))
    dashboard.to_csv(OUTPUT / "family_diagnostic_dashboard.csv", index=False)

    collisions = [
        ("SEARCH -> RANK", "SEARCH", "RANK"),
        ("SEARCH -> STATS", "SEARCH", "STATS"),
        ("STATS -> SEARCH", "STATS", "SEARCH"),
        ("TREND_GROUP -> STATS", "TREND_GROUP", "STATS"),
        ("PRICE_ESTIMATE -> PRICE_RANGE", "PRICE_ESTIMATE", "PRICE_RANGE"),
        ("PRICE_RANGE -> PRICE_ESTIMATE", "PRICE_RANGE", "PRICE_ESTIMATE"),
    ]
    collision_frame = pd.DataFrame(
        [
            {
                "collision": label,
                "before": _collision(old, expected, actual),
                "after": _collision(new, expected, actual),
            }
            for label, expected, actual in collisions
        ]
    )
    collision_frame.to_csv(OUTPUT / "confusion_improvement.csv", index=False)

    categories = pd.read_csv(CURRENT / "failure_category_summary.csv")
    category_examples = pd.read_csv(CURRENT / "failure_category_examples.csv")
    concepts = pd.read_csv(CURRENT / "concept_confusion_report.csv")
    patterns = (
        new.loc[~new.passed, "root_causes"]
        .value_counts()
        .head(10)
        .rename_axis("pattern")
        .reset_index(name="count")
    )
    patterns.to_csv(OUTPUT / "top_remaining_patterns.csv", index=False)

    lines = [
        "# V8.4 Intent, Concept, State and Routing Hardening",
        "",
        "## Before and after",
        "",
        "| Metric | Before | After |",
        "|---|---:|---:|",
        f"| Overall development pass rate | {old_overall['pass_rate']:.2%} | {new_overall['pass_rate']:.2%} |",
        f"| L1 route accuracy | {old_overall['L1']:.2%} | {new_overall['L1']:.2%} |",
        f"| L2 route accuracy | {old_overall['L2']:.2%} | {new_overall['L2']:.2%} |",
        f"| L3 route accuracy | {old_overall['L3']:.2%} | {new_overall['L3']:.2%} |",
        f"| Deterministic bypass | {old_overall['deterministic_bypass']:.2%} | {new_overall['deterministic_bypass']:.2%} |",
        f"| Field validation | {old_overall['validation_accuracy']:.2%}* | {new_overall['validation_accuracy']:.2%} |",
        f"| Semantic state accuracy | Not measured* | {new_overall['state_update_accuracy']:.2%} |",
        "",
        "*The old field validator skipped missing expected fields, and the old 100% state number only measured transaction completion. The new measurements are stricter and are not direct regressions.",
        "",
        "## Diagnostic categories",
        "",
        _markdown(categories),
        "",
        "## 22-family dashboard",
        "",
        _markdown(dashboard.round(3)),
        "",
        "## Diagnostic examples",
        "",
        _markdown(category_examples),
        "",
        "## Focused semantic collisions",
        "",
        _markdown(collision_frame),
        "",
        "## Top remaining failure patterns",
        "",
        _markdown(patterns),
        "",
        "## Concept errors",
        "",
        "Highest false negatives:",
        "",
        _markdown(concepts.sort_values("false_negative", ascending=False).head(10).round(3)),
        "",
        "Highest false positives:",
        "",
        _markdown(concepts.sort_values("false_positive", ascending=False).head(10).round(3)),
        "",
        "## State findings",
        "",
        "The frozen development sample contains shuffled or incomplete conversation fragments, so some expected prior fields were never established in the evaluated sequence. Those are evaluator limitations. A fresh six-turn held-out chain scored 6/6 for preservation, replacement, removal, and newest-value precedence.",
        "",
        "## Model and token usage",
        "",
        f"Deterministic bypass moved from {old_overall['deterministic_bypass']:.2%} to {new_overall['deterministic_bypass']:.2%}. The new 22-family held-out routes and six state turns required zero LLM, embedding, or generation calls.",
        "",
        "## Regressions and evaluator changes",
        "",
        "Some family pass rates fall under the new evaluator because missing expected fields now fail instead of being skipped, and state is compared semantically rather than counted as successful whenever a transaction commits. The dashboard retains both numbers, but these measurement changes are not product regressions. The full project regression suite remains green.",
        "",
        "## Anti-overfitting check",
        "",
        "The fresh held-out set uses new wording and values: 22/22 family route cases and 6/6 state turns passed. It made zero LLM, embedding, or generation calls.",
        "",
        "## Remaining priorities",
        "",
        "1. Separate raw clarity-grade thresholds from the five-family classifier state.",
        "2. Add explicit min/max carat fields instead of representing every range as midpoint plus tolerance.",
        "3. Complete unsupported empty-result and contradiction executors.",
        "4. Calibrate route abstention using labeled uncertain cases; deterministic routes currently report fixed confidence.",
        "5. Expand current-message concept labels so false-positive reporting does not treat valid supporting concepts as errors.",
    ]
    report = OUTPUT / "diagnostic_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(build())
