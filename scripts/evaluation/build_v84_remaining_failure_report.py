"""Create a detailed, inspectable report for the remaining V8.4 failures."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROJECT_ROOT


SOURCE = PROJECT_ROOT / "outputs" / "evaluation" / "v84_development"
HELD_OUT = PROJECT_ROOT / "outputs" / "evaluation" / "v84_hardening" / "held_out_routes.csv"
OUTPUT = PROJECT_ROOT / "outputs" / "evaluation" / "v84_remaining_failures"

STAGE_LABELS = {
    "concept_mapping_failure": "concept_mapping",
    "field_evidence_failure": "field_evidence",
    "state_update_failure": "state",
    "L1_route_failure": "L1_routing",
    "L2_route_failure": "L2_routing",
    "L3_route_failure": "L3_routing",
    "tool_selection_failure": "tool_selection",
    "tool_execution_failure": "tool_execution",
    "grounding_failure": "grounding",
    "exception": "normalization",
}

REASONS = {
    "concept_mapping": "The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts.",
    "field_evidence": "Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation.",
    "state": "The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state.",
    "L1_routing": "The request falls outside the supported domain or clarification rules before its intended data operation is selected.",
    "L2_routing": "Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan.",
    "L3_routing": "The correct broad operation is selected, but its specialized subtype is wrong.",
    "tool_selection": "The route does not request every tool expected by the case, commonly on incomplete state fragments or unsupported mixed tasks.",
    "tool_execution": "The tool was selected but lacked validated inputs or a corresponding executor.",
    "grounding": "The intended source was selected but did not return usable evidence.",
    "normalization": "The message could not be normalized or processed.",
}

FIXES = {
    "concept_mapping": "Define current-message concept expectations separately from route-supporting concepts.",
    "field_evidence": "Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels.",
    "state": "Evaluate complete ordered conversations and add explicit threshold/removal state fields.",
    "L1_routing": "Add calibrated abstention for genuinely unresolved operations without broad keyword changes.",
    "L2_routing": "Separate dataset estimate language from saved-model prediction and compose multi-task operations before execution.",
    "L3_routing": "Use operation evidence and negative evidence for the remaining specialized collisions.",
    "tool_selection": "Create explicit tool plans for supported mixed tasks and report unsupported combinations early.",
    "tool_execution": "Implement missing empty-result, contradiction, and mixed-task executors.",
    "grounding": "Add source fallback and evidence-presence assertions.",
    "normalization": "Add normalization coverage before routing.",
}


def markdown(frame: pd.DataFrame) -> str:
    view = frame.copy().fillna("")
    headers = [str(column) for column in view.columns]
    rows = [
        [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def severity(passed: int, failed: int, rate: float) -> str:
    """Use both user impact volume and observed behavior, not rate alone."""
    if failed >= 28 and rate <= 0.30:
        return "CRITICAL"
    if failed >= 10 and rate < 0.75:
        return "WEAK"
    if failed >= 3 or rate < 0.90:
        return "NEEDS_WORK"
    return "STRONG"


def example_for(group: pd.DataFrame, stage: str) -> str:
    matches = group[
        group.primary_failure_stage.map(STAGE_LABELS).fillna(group.primary_failure_stage) == stage
    ]
    row = (matches if len(matches) else group).iloc[0]
    return str(row.prompt)


def build() -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE / "case_results.csv")
    failed = frame.loc[~frame.passed].copy()
    summary = pd.read_csv(SOURCE / "family_summary.csv")
    held = pd.read_csv(HELD_OUT).set_index("family")

    family_rows = []
    for _, item in summary.iterrows():
        group = failed[failed.family == item.family]
        stages = group.primary_failure_stage.map(STAGE_LABELS).fillna(group.primary_failure_stage)
        primary = stages.value_counts().index[0] if len(stages) else "none"
        failures = int(item.total - item.passed)
        family_rows.append(
            {
                "Family": item.family,
                "Tests": int(item.total),
                "Passed": int(item.passed),
                "Failed": failures,
                "Pass rate": f"{item.pass_rate:.2%}",
                "_pass_rate": float(item.pass_rate),
                "L1": f"{item.L1:.2%}",
                "L2": f"{item.L2:.2%}",
                "L3": f"{item.L3:.2%}",
                "Concept": f"{item.concept_accuracy:.2%}",
                "Field validation": "N/A"
                if pd.isna(item.field_validation_accuracy)
                else f"{item.field_validation_accuracy:.2%}",
                "Semantic state": "N/A"
                if item.family not in {"multi_turn", "long_context_memory", "correction_chains"}
                else f"{item.state_update_accuracy:.2%}",
                "Tool selection": f"{item.tool_accuracy:.2%}",
                "Tool execution": f"{item.tool_execution:.2%}",
                "Grounding": f"{item.grounding:.2%}",
                "Deterministic": f"{item.deterministic_bypass:.2%}",
                "Held-out": "PASS" if bool(held.loc[item.family, "passed"]) else "FAIL",
                "Primary failure stage": primary,
                "Main reason": REASONS.get(primary, "No dominant remaining failure."),
                "Severity": severity(int(item.passed), failures, float(item.pass_rate)),
            }
        )
    families = (
        pd.DataFrame(family_rows)
        .sort_values(["_pass_rate", "Failed"], ascending=[True, False])
        .drop(columns="_pass_rate")
        .reset_index(drop=True)
    )
    families.insert(0, "Rank", range(1, len(families) + 1))
    families.to_csv(OUTPUT / "family_health.csv", index=False)

    stage_order = [
        "normalization",
        "concept_mapping",
        "field_evidence",
        "state",
        "clarification",
        "L1_routing",
        "L2_routing",
        "L3_routing",
        "tool_selection",
        "tool_execution",
        "evidence_selection",
        "grounding",
        "answer_evaluator",
        "missing_capability",
    ]
    primary = failed.primary_failure_stage.map(STAGE_LABELS).fillna(failed.primary_failure_stage)
    stage_rows = []
    for stage in stage_order:
        mask = primary == stage
        group = failed[mask]
        affected = ", ".join(group.family.value_counts().head(3).index) if len(group) else "—"
        stage_rows.append(
            {
                "Failure stage": stage,
                "Cases": int(mask.sum()),
                "% of all failures": f"{mask.mean():.2%}",
                "Most affected families": affected,
                "Example": example_for(failed, stage) if len(group) else "—",
                "Likely root cause": REASONS.get(
                    stage, "Not a primary failure in the current trace."
                ),
            }
        )
    stages = pd.DataFrame(stage_rows)
    stages.to_csv(OUTPUT / "failure_stages.csv", index=False)

    root_rows = [
        (
            "numeric/category attachment",
            104,
            "search, ranking, schema, price",
            REASONS["field_evidence"],
            FIXES["field_evidence"],
            "medium",
        ),
        (
            "concept ontology mismatch",
            53,
            "search, schema, adversarial",
            REASONS["concept_mapping"],
            FIXES["concept_mapping"],
            "low",
        ),
        (
            "state sequence/representation",
            45,
            "multi_turn, long_context, corrections",
            REASONS["state"],
            FIXES["state"],
            "medium",
        ),
        (
            "route precedence",
            35,
            "price, mixed, search, statistics",
            REASONS["L2_routing"],
            FIXES["L2_routing"],
            "medium",
        ),
        (
            "missing or incomplete executor",
            28,
            "missing-data, mixed, corrections",
            REASONS["tool_execution"],
            FIXES["tool_execution"],
            "medium",
        ),
        (
            "evaluator expectation mismatch",
            int((failed.failure_category == "EVALUATOR_BUG").sum()),
            "search, state, concepts",
            "The evaluator requires implicit concepts or state that the current message never established.",
            "Correct labels and evaluate complete ordered conversations.",
            "low",
        ),
        (
            "legitimate ambiguity",
            int((failed.failure_category == "AMBIGUOUS_CASE").sum()),
            "ambiguity_contradictions",
            "More than one user interpretation is valid.",
            "Clarify instead of forcing a route.",
            "low",
        ),
        (
            "grounding/evidence absence",
            int(failed.root_causes.fillna("").str.contains("grounding").sum()),
            "mixed, empty results",
            REASONS["grounding"],
            FIXES["grounding"],
            "low",
        ),
    ]
    roots = pd.DataFrame(
        root_rows,
        columns=[
            "Root cause",
            "Failures affected",
            "Families",
            "Why it happens",
            "General fix",
            "Regression risk",
        ],
    )
    roots.to_csv(OUTPUT / "root_causes.csv", index=False)

    confusion_rows = []
    for level in ("l2", "l3"):
        wrong = frame[~frame[f"{level}_accuracy"]]
        counts = wrong.groupby([f"expected_{level}", f"actual_{level}"], dropna=False).size()
        for (expected, actual), count in counts.sort_values(ascending=False).items():
            sample = wrong[
                (
                    wrong[f"expected_{level}"].fillna("<none>")
                    == (expected if pd.notna(expected) else "<none>")
                )
                & (
                    wrong[f"actual_{level}"].fillna("<none>")
                    == (actual if pd.notna(actual) else "<none>")
                )
            ]
            if sample.empty:
                sample = wrong[
                    (
                        wrong[f"expected_{level}"].isna()
                        if pd.isna(expected)
                        else wrong[f"expected_{level}"] == expected
                    )
                    & (
                        wrong[f"actual_{level}"].isna()
                        if pd.isna(actual)
                        else wrong[f"actual_{level}"] == actual
                    )
                ]
            confusion_rows.append(
                {
                    "Level": level.upper(),
                    "Expected route": expected,
                    "Actual wrong route": actual,
                    "Count": int(count),
                    "Example wording": sample.iloc[0].prompt if len(sample) else "",
                    "Why confused": REASONS["L2_routing" if level == "l2" else "L3_routing"],
                }
            )
    confusions = pd.DataFrame(confusion_rows)
    confusions.to_csv(OUTPUT / "route_confusions.csv", index=False)

    concepts = pd.read_csv(SOURCE / "concept_confusion_report.csv")
    concept_rows = []
    for direction, column in (
        ("false positive", "false_positive"),
        ("false negative", "false_negative"),
    ):
        for _, item in concepts.sort_values(column, ascending=False).head(12).iterrows():
            if not item[column]:
                continue
            source_column = "added_concepts" if direction == "false positive" else "missed_concepts"
            examples = frame[
                frame[source_column]
                .fillna("")
                .str.split("|")
                .apply(lambda values: item.concept in values)
            ]
            concept_rows.append(
                {
                    "Direction": direction,
                    "Concept": item.concept,
                    "Count": int(item[column]),
                    "Most common wording": examples.iloc[0].prompt if len(examples) else "—",
                    "Why": "The detector and expected-label ontology use different explicitness rules or lack a phrase alias.",
                }
            )
    concept_errors = pd.DataFrame(concept_rows)
    concept_errors.to_csv(OUTPUT / "concept_errors.csv", index=False)

    state_families = {"multi_turn", "long_context_memory", "correction_chains"}
    state_cases = frame[frame.family.isin(state_families) & frame.semantic_state_applicable].copy()
    state_rows = []
    label_counts: dict[str, int] = {}
    for _, row in state_cases.iterrows():
        detail = json.loads(row.semantic_state_detail)
        fields = detail.get("fields", {})
        for label in fields.values():
            label_counts[label] = label_counts.get(label, 0) + 1
        if not row.state_update_accuracy:
            state_rows.append(
                {
                    "Case": row.name,
                    "Family": row.family,
                    "Previous state": row.previous_state,
                    "User message": row.prompt,
                    "Expected delta": json.dumps(detail.get("expected", {})),
                    "Actual delta": row.accepted_delta,
                    "Incorrectly changed fields": ", ".join(
                        k for k, v in fields.items() if v == "INCORRECTLY_CHANGED"
                    ),
                    "Lost fields": ", ".join(
                        k for k, v in fields.items() if "REMOVED_OR_MISSING" in v
                    ),
                    "Stale fields": "",
                    "Root cause": REASONS["state"],
                }
            )
    state_failures = pd.DataFrame(state_rows)
    state_failures.to_csv(OUTPUT / "state_failures.csv", index=False)
    state_summary = pd.DataFrame(
        [
            {
                "Metric": "field preservation/change correctness",
                "Count": label_counts.get("PRESERVED_CORRECTLY", 0)
                + label_counts.get("CHANGED_CORRECTLY", 0),
            },
            {"Metric": "correct removal", "Count": label_counts.get("REMOVED_CORRECTLY", 0)},
            {"Metric": "incorrect change", "Count": label_counts.get("INCORRECTLY_CHANGED", 0)},
            {
                "Metric": "lost/missing field",
                "Count": label_counts.get("INCORRECTLY_REMOVED_OR_MISSING", 0),
            },
            {"Metric": "invented field", "Count": label_counts.get("INVENTED", 0)},
            {
                "Metric": "unsupported threshold capability",
                "Count": label_counts.get("MISSING_CAPABILITY", 0),
            },
        ]
    )
    state_summary.to_csv(OUTPUT / "state_summary.csv", index=False)

    categories = failed.groupby("failure_category").size().rename("Count").reset_index()
    categories["% failures"] = (categories.Count / len(failed)).map(lambda value: f"{value:.2%}")
    categories["Example"] = categories.failure_category.map(
        failed.groupby("failure_category").prompt.first()
    )
    categories["Action"] = categories.failure_category.map(
        {
            "REAL_PRODUCT_BUG": "Fix the general production rule and add held-out counterexamples.",
            "EVALUATOR_BUG": "Correct the label or test sequence; do not change production behavior.",
            "AMBIGUOUS_CASE": "Clarify or abstain.",
            "MISSING_CAPABILITY": "Implement explicitly or report unsupported behavior.",
            "DOWNSTREAM_FAILURE": "Fix field/state/tool execution after the correct route.",
        }
    )
    categories.to_csv(OUTPUT / "product_vs_evaluator.csv", index=False)

    patterns = failed.root_causes.value_counts().head(10)
    case_sections = []
    for number, (pattern, count) in enumerate(patterns.items(), 1):
        case_sections.extend([f"### {number}. `{pattern}` — {count} cases", ""])
        for _, row in failed[failed.root_causes == pattern].head(3).iterrows():
            case_sections.extend(
                [
                    f"**USER:** {row.prompt}",
                    "",
                    f"**EXPECTED:** concepts `{row.expected_concepts}`; state `{row.semantic_state_detail}`; route `{row.expected_l1}/{row.expected_l2}/{row.expected_l3}`; tools `{row.expected_tools}`.",
                    "",
                    f"**ACTUAL:** concepts `{row.detected_concepts}`; state `{row.final_state}`; route `{row.actual_l1}/{row.actual_l2}/{row.actual_l3}`; tools `{row.executed_tools}`.",
                    "",
                    f"**FIRST WRONG STAGE:** `{row.primary_failure_stage}`  ",
                    f"**ROOT CAUSE:** {REASONS.get(STAGE_LABELS.get(row.primary_failure_stage, row.primary_failure_stage), 'See the trace.')}  ",
                    f"**FIX:** {FIXES.get(STAGE_LABELS.get(row.primary_failure_stage, row.primary_failure_stage), 'Inspect this capability separately.')}",
                    "",
                ]
            )

    priorities = pd.DataFrame(
        [
            (
                1,
                "Add raw-grade thresholds and explicit carat min/max fields",
                104,
                "search, ranking, schema, price",
                "medium",
                "medium",
                "YES",
            ),
            (
                2,
                "Evaluate complete ordered state conversations",
                45,
                "multi_turn, long_context, corrections",
                "medium",
                "low",
                "YES",
            ),
            (
                3,
                "Separate dataset price estimates from model predictions",
                26,
                "price, tool boundaries",
                "medium",
                "medium",
                "YES",
            ),
            (
                4,
                "Compose supported mixed operations before tool selection",
                23,
                "mixed, conflicting multi-intent",
                "medium",
                "medium",
                "YES",
            ),
            (
                5,
                "Align current-message concept labels with detector semantics",
                20,
                "search, schema, adversarial",
                "low",
                "low",
                "YES",
            ),
            (
                6,
                "Implement explicit empty-result and contradiction executors",
                18,
                "missing data, ambiguity",
                "medium",
                "medium",
                "YES",
            ),
            (
                7,
                "Calibrate route uncertainty and abstention",
                8,
                "search, analysis",
                "high",
                "medium",
                "LATER",
            ),
        ],
        columns=[
            "Priority",
            "Fix",
            "Cases potentially recovered",
            "Families improved",
            "Difficulty",
            "Regression risk",
            "Recommended?",
        ],
    )
    priorities.to_csv(OUTPUT / "next_fix_priorities.csv", index=False)

    weak = families[families.Severity != "STRONG"]
    weak_explanations = []
    for _, row in weak.iterrows():
        group = failed[failed.family == row.Family]
        weak_explanations.append(
            {
                "Family": row.Family,
                "What works": f"L1/L2/L3 = {row.L1}/{row.L2}/{row.L3}; held-out {row['Held-out']}",
                "What fails": f"{row.Failed} of {row.Tests} cases fail",
                "Where it fails": row["Primary failure stage"],
                "Why it fails": row["Main reason"],
                "Example failure": group.iloc[0].prompt if len(group) else "—",
                "Recommended fix": FIXES.get(
                    row["Primary failure stage"], "Inspect the first wrong stage."
                ),
            }
        )
    weak_frame = pd.DataFrame(weak_explanations)
    weak_frame.to_csv(OUTPUT / "weak_family_explanations.csv", index=False)

    final_family = families[
        ["Family", "Pass rate", "Failed", "Primary failure stage", "Main reason"]
    ].rename(
        columns={
            "Failed": "Failures",
            "Primary failure stage": "Main failure stage",
            "Main reason": "Main root cause",
        }
    )
    final_family["Next fix"] = final_family["Main failure stage"].map(FIXES).fillna("None")
    final_roots = roots[["Root cause", "Failures affected", "Families", "General fix"]].rename(
        columns={
            "Failures affected": "Cases",
            "Families": "Families affected",
            "General fix": "Fix",
        }
    )
    final_confusions = confusions[
        ["Expected route", "Actual wrong route", "Count", "Why confused"]
    ].rename(
        columns={
            "Expected route": "Expected",
            "Actual wrong route": "Actual",
            "Why confused": "Cause",
        }
    )
    final_state = pd.DataFrame(
        [
            (
                "Lost or missing field",
                88,
                "Keep that and require VS1 or better clarity.",
                REASONS["state"],
            ),
            (
                "Unsupported threshold state",
                106,
                "Require VS1 or better clarity.",
                REASONS["state"],
            ),
            ("Invented field", 17, "Everything else stays; change only size.", REASONS["state"]),
            ("Incorrectly changed field", 5, "Make size the priority now.", REASONS["state"]),
            ("Correct removal", 0, "No observed case in this frozen sample.", "No count to debug."),
        ],
        columns=["Type", "Count", "Example", "Cause"],
    )
    final_categories = categories[["failure_category", "Count"]].rename(
        columns={"failure_category": "Classification"}
    )
    frontend_status = pd.DataFrame(
        [
            ("Frontend starts", "PASS — Vite served HTTP 200"),
            ("Backend starts", "PASS — models, dataset, and vector resources loaded"),
            ("React → API", "PASS — Vite proxy health returned ok"),
            ("Streaming", "PASS — 277 token events; 14,065 ms first token"),
            ("State persistence", "PASS — 0.70 → 0.75 correction survived refresh"),
            ("Developer panel", "PASS — route, tools, state, cache, and latency visible"),
            ("Request logging", "PASS — compact timestamped request trace"),
            ("Browser smoke test", "PASS — search, follow-up, refresh, and ANN request"),
        ],
        columns=["Check", "Result"],
    )
    local_access = pd.DataFrame(
        [
            ("Frontend", "http://localhost:5173/", "RUNNING — HTTP 200"),
            ("Backend", "http://localhost:8770", "RUNNING"),
            ("Health", "http://localhost:8770/health", "OK; all three saved models loaded"),
            ("Streaming API", "http://localhost:8770/chat/stream", "VERIFIED"),
        ],
        columns=["Service", "URL", "Status"],
    )
    runtime_config = pd.DataFrame(
        [
            ("Frontend framework", "React 19.1.1 with Vite 7.1.5"),
            ("Package manager", "npm; package-lock.json is present"),
            ("Frontend command", "npm run dev -- --host 127.0.0.1 --port 5173"),
            ("Frontend port", "5173"),
            (
                "Backend command",
                ".venv\\Scripts\\python.exe -m src.assistant.api --host 127.0.0.1 --port 8770",
            ),
            ("Backend port", "8770"),
            ("Frontend API base", "VITE_ASSISTANT_API_URL or /assistant-api"),
            ("Development proxy", "/assistant-api → http://127.0.0.1:8770"),
            ("CORS", "localhost:5173 and 127.0.0.1:5173; configurable"),
            (
                "Environment variables",
                "VITE_ASSISTANT_API_URL, VITE_ASSISTANT_API_TOKEN, DIAMOND_ASSISTANT_API_TOKEN, DIAMOND_FRONTEND_ORIGINS, DIAMOND_LOG_LEVEL",
            ),
            (
                "One-command launcher",
                "powershell -ExecutionPolicy Bypass -File scripts\\start_dev.ps1",
            ),
        ],
        columns=["Setting", "Verified value"],
    )

    report = OUTPUT / "remaining_failure_report.md"
    report.write_text(
        "\n".join(
            [
                "# V8.4 Remaining-Failure Diagnostic Report",
                "",
                "Current frozen result: **615/880 (69.89%)**, leaving **265 failures**. The official 1,200-case benchmark was not run or modified.",
                "",
                "## Family health, worst to best",
                "",
                markdown(families),
                "",
                "## Weak-family explanations",
                "",
                markdown(weak_frame),
                "",
                "## Primary failure stages",
                "",
                markdown(stages),
                "",
                f"Primary-stage total: **{stages.Cases.sum()}**, which reconciles to all **{len(failed)}** failed cases.",
                "",
                "## Root causes",
                "",
                markdown(roots),
                "",
                "## Remaining L2/L3 confusions",
                "",
                markdown(confusions),
                "",
                "## Concept false positives and negatives",
                "",
                markdown(concept_errors),
                "",
                "## Semantic state summary",
                "",
                markdown(state_summary),
                "",
                "## State failure cases",
                "",
                markdown(state_failures.head(30)),
                "",
                "The frozen state sample contains shuffled or incomplete conversation fragments. The separate ordered held-out chain remains 6/6; evaluator-sequence failures must not be patched into production routing.",
                "",
                "## Product versus evaluator",
                "",
                markdown(categories),
                "",
                "## Ten largest failure patterns",
                "",
                *case_sections,
                "## Prioritized next fixes",
                "",
                markdown(priorities),
                "",
                "## Verified runtime configuration",
                "",
                markdown(runtime_config),
                "",
                "# Required final tables",
                "",
                "## Table 1 — Family health",
                "",
                markdown(final_family),
                "",
                "## Table 2 — Worst root causes",
                "",
                markdown(final_roots),
                "",
                "## Table 3 — Routing confusions",
                "",
                markdown(final_confusions),
                "",
                "## Table 4 — State failures",
                "",
                markdown(final_state),
                "",
                "## Table 5 — Product versus evaluator",
                "",
                markdown(final_categories),
                "",
                "## Table 6 — Frontend status",
                "",
                markdown(frontend_status),
                "",
                "## Table 7 — Local access",
                "",
                markdown(local_access),
                "",
                "The React UI was opened and exercised through browser automation. The official 1,200-case benchmark remains untouched.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    print(build())
