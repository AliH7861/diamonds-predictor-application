"""Ground comparisons and trade-offs in rows already shown to the user."""

from __future__ import annotations

import re

import pandas as pd

from .clarity import clarity_family


ORDINALS = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}


def displayed_rows(conversation: list[dict] | None) -> pd.DataFrame:
    """Return the most recent displayed rows and preserve their visible order."""
    for message in reversed(conversation or []):
        result = message.get("result") or {}
        rows = result.get("similar_matches")
        if not isinstance(rows, pd.DataFrame) or rows.empty:
            rows = result.get("matches")
        if isinstance(rows, pd.DataFrame) and not rows.empty:
            return rows.head(5).copy().reset_index(drop=True)
        if isinstance(rows, list) and rows:
            return pd.DataFrame(rows).head(5).reset_index(drop=True)
    return pd.DataFrame()


def selected_rows(rows: pd.DataFrame, question: str) -> pd.DataFrame:
    """Resolve ordinal references such as first and third."""
    positions = [
        index for word, index in ORDINALS.items() if re.search(rf"\b{word}\b", question, re.I)
    ]
    numbered = rows.copy()
    numbered["_display_position"] = range(len(numbered))
    if not positions:
        return numbered
    valid = [index for index in positions if index < len(rows)]
    return numbered.iloc[valid].copy().reset_index(drop=True)


def _quality_text(row: pd.Series) -> str:
    return (
        f"{float(row['carat']):.2f} ct, {row['cut']} cut, {row['color']} color, "
        f"{clarity_family(str(row['clarity']))} clarity"
    )


def compare_rows(rows: pd.DataFrame, question: str) -> tuple[str, pd.DataFrame, dict]:
    """Create a short factual comparison without an LLM inventing attributes."""
    chosen = selected_rows(rows, question)
    if chosen.empty:
        return (
            "I need a displayed set of diamonds before I can compare them.",
            chosen,
            {"selected_positions": []},
        )
    if len(chosen) == 1 and "expensive" in question.casefold():
        target = chosen.iloc[0]
        selected_positions = set(chosen["_display_position"].astype(int))
        baseline = rows.loc[~rows.index.isin(selected_positions)]
        baseline_price = (
            float(baseline["price"].median()) if not baseline.empty else float(target["price"])
        )
        reasons = []
        if float(target["carat"]) >= float(rows["carat"].median()):
            reasons.append("its carat weight is relatively high")
        if target["cut"] in {"Ideal", "Premium"}:
            reasons.append(f"it has a {target['cut']} cut")
        if clarity_family(str(target["clarity"])) in {"VS", "VVS", "IF"}:
            reasons.append(f"it has {clarity_family(str(target['clarity']))} clarity")
        reason = ", and ".join(reasons) if reasons else "the combination of its measured grades"
        answer = (
            f"That diamond is ${float(target['price']):,.0f}, compared with a ${baseline_price:,.0f} "
            f"median for the other displayed options. The dataset supports {reason} as the main "
            "visible differences; it does not establish a single causal reason for the price."
        )
    else:
        low = chosen.loc[chosen["price"].idxmin()]
        large = chosen.loc[chosen["carat"].idxmax()]
        answer = (
            f"The lowest-price option is ${float(low['price']):,.0f} with {_quality_text(low)}. "
            f"The largest is ${float(large['price']):,.0f} with {_quality_text(large)}. "
            "The practical trade-off is how much size you gain for the price while changing cut, "
            "color, or clarity."
        )
    evidence = {
        "selected_positions": chosen["_display_position"].astype(int).tolist(),
        "compared_rows": chosen.to_dict("records"),
    }
    return answer, chosen, evidence
