"""Deterministic Pandas execution for explicit statistics, trends, and price analysis."""

import re
import numpy as np
import pandas as pd


def _working_frame(frame: pd.DataFrame, text: str) -> pd.DataFrame:
    data = frame.copy()
    data["price_per_carat"] = data["price"] / data["carat"]
    # These are the canonical engineered features already used by this
    # project. They are derived on demand so analysis works with the raw CSV.
    data["volume_proxy"] = data["x"] * data["y"] * data["z"]
    data["face_area_proxy"] = data["x"] * data["y"]
    data["length_width_ratio"] = data["x"] / data["y"].replace(0, np.nan)
    data["depth_ratio_xyz"] = data["z"] / np.sqrt(data["x"] ** 2 + data["y"] ** 2).replace(
        0, np.nan
    )
    data["carat_squared"] = data["carat"] ** 2
    for column, choices in (
        ("cut", ("Very Good", "Premium", "Ideal", "Good", "Fair")),
        ("clarity", ("VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "IF", "I1")),
    ):
        for choice in choices:
            if re.search(rf"\b{re.escape(choice)}\b", text, re.I):
                data = data[data[column].str.casefold() == choice.casefold()]
                break
    color = re.search(r"\b(?:color\s+)?([D-J])(?:\s+color)?\b", text, re.I)
    if color:
        data = data[data["color"].str.upper() == color.group(1).upper()]
    family = re.search(r"\b(VVS|VS|SI)\b", text, re.I)
    if family:
        data = data[data["clarity"].str.upper().str.startswith(family.group(1).upper())]
    maximum_price = re.search(
        r"\b(?:under|below|less than|up to)\s*\$?\s*(\d[\d,]*(?:\.\d+)?)",
        text,
        re.I,
    )
    if maximum_price:
        data = data[data["price"] <= float(maximum_price.group(1).replace(",", ""))]
    carat_floor = re.search(
        r"\b(?:above|over|more than|at least)\s+(\d*\.\d+|\d+(?:\.\d+)?)\s*(?:ct|carats?)",
        text,
        re.I,
    )
    if carat_floor:
        data = data[data["carat"] >= float(carat_floor.group(1))]
    return data


def _metric(text: str) -> str:
    if re.search(r"\b(?:average|avg|mean)\b", text, re.I):
        return "mean"
    if re.search(r"\bmedian\b", text, re.I):
        return "median"
    if re.search(r"\b(?:minimum|min)\b", text, re.I):
        return "min"
    if re.search(r"\b(?:maximum|max)\b", text, re.I):
        return "max"
    if re.search(r"\b(?:standard deviation|std)\b", text, re.I):
        return "std"
    if re.search(r"\b(?:q25|25th percentile)\b", text, re.I):
        return "q25"
    if re.search(r"\b(?:q75|75th percentile)\b", text, re.I):
        return "q75"
    return "count"


def _target(text: str) -> str:
    engineered = {
        "volume_proxy": r"\bvolume(?: proxy)?\b",
        "face_area_proxy": r"\bface area(?: proxy)?\b",
        "length_width_ratio": r"\b(?:length width|x y|xy) ratio\b",
        "depth_ratio_xyz": r"\bdepth ratio xyz\b",
        "carat_squared": r"\bcarat squared\b",
    }
    for column, pattern in engineered.items():
        if re.search(pattern, text, re.I):
            return column
    if re.search(r"\bprices?\b", text, re.I):
        return "price"
    if re.search(r"\b(?:carats?|carots?|carrots?)\b", text, re.I):
        return "carat"
    if re.search(r"price\s+per\s+(?:carat|ct)", text, re.I):
        return "price_per_carat"
    for column in ("price", "carat", "depth", "table", "x", "y", "z"):
        if re.search(rf"\b{column}\b", text, re.I):
            return column
    return "price"


def _aggregate(series: pd.Series, metric: str) -> float:
    """Apply a supported aggregate, including explicit quartiles."""
    if metric == "q25":
        return float(series.quantile(0.25))
    if metric == "q75":
        return float(series.quantile(0.75))
    return float(getattr(series, metric)())


def execute_analysis(frame: pd.DataFrame, question: str, action: str) -> tuple[str, dict]:
    """Execute a frozen analysis action without embeddings or an LLM."""
    data = _working_frame(frame, question)
    if data.empty:
        return "No rows match those constraints.", {"row_count": 0, "action": action}
    target, metric = _target(question), _metric(question)
    if action == "DATASET_OVERVIEW":
        numeric = {
            column: {
                "minimum": float(data[column].min()),
                "median": float(data[column].median()),
                "maximum": float(data[column].max()),
            }
            for column in ("price", "carat", "depth", "table", "x", "y", "z")
        }
        categories = {
            column: data[column].value_counts().to_dict() for column in ("cut", "color", "clarity")
        }
        answer = (
            f"The dataset contains {len(data):,} diamonds. Prices range from "
            f"${numeric['price']['minimum']:,.0f} to ${numeric['price']['maximum']:,.0f}, "
            f"with a ${numeric['price']['median']:,.0f} median. Carat ranges from "
            f"{numeric['carat']['minimum']:.2f} to {numeric['carat']['maximum']:.2f}, "
            f"with a {numeric['carat']['median']:.2f} median. Its 10 modeling fields cover price, "
            "carat, cut, color, clarity, depth, table, and x/y/z millimeter dimensions."
        )
        return answer, {
            "action": action,
            "row_count": len(data),
            "numeric_ranges": numeric,
            "category_counts": categories,
        }
    if action in {"RANGE", "PRICE_RANGE"}:
        low = float(data[target].min())
        median = float(data[target].median())
        high = float(data[target].max())
        unit = "$" if target == "price" else ""
        suffix = " carats" if target == "carat" else ""
        precision = ",.0f" if target == "price" else ",.2f"
        return (
            f"Across {len(data):,} dataset rows, {target} ranges from "
            f"{unit}{format(low, precision)} to {unit}{format(high, precision)}{suffix}; "
            f"the median is {unit}{format(median, precision)}{suffix}.",
            {
                "action": action,
                "target": target,
                "minimum": low,
                "median": median,
                "maximum": high,
                "row_count": len(data),
            },
        )
    if action == "EXTREME":
        wants_low = bool(re.search(r"\b(?:smallest|lowest)\b", question, re.I))
        index = data[target].idxmin() if wants_low else data[target].idxmax()
        row = data.loc[index]
        label = "smallest" if wants_low else "largest"
        answer = (
            f"The {label} {target} in the dataset is {float(row[target]):,.2f}. "
            f"That row is a {float(row['carat']):.2f} carat {row['cut']} diamond priced at "
            f"${float(row['price']):,.0f}."
        )
        return answer, {
            "metric": label,
            "target": target,
            "value": float(row[target]),
            "row": row[["price", "carat", "cut", "color", "clarity"]].to_dict(),
        }
    if action == "DISTRIBUTION":
        if target == "carat":
            counts = data["carat"].round(2).value_counts().head(5)
            values = {f"{value:.2f}": int(count) for value, count in counts.items()}
            summary = ", ".join(f"{value} ct ({count:,})" for value, count in values.items())
            return f"The most common recorded carat weights are {summary}.", {
                "target": "carat",
                "metric": "most_common",
                "values": values,
            }
        counts = data[target].value_counts().head(10)
        values = {str(value): int(count) for value, count in counts.items()}
        return f"The most common {target} values are " + ", ".join(
            f"{value} ({count:,})" for value, count in values.items()
        ) + ".", {"target": target, "metric": "most_common", "values": values}
    if action == "PRICE_ESTIMATE":
        value = data["price"].median()
        return f"The typical matching dataset price is ${value:,.0f} (median).", {
            "median": float(value),
            "row_count": len(data),
        }
    if action == "TREND_RELATION":
        columns = [
            name
            for name in ("carat", "price", "depth", "table", "x", "y", "z")
            if re.search(rf"\b{name}\b", question, re.I)
        ]
        x, y = (columns + ["carat", "price"])[:2]
        value = data[[x, y]].corr().iloc[0, 1]
        return f"The dataset correlation between {x} and {y} is {value:.3f}.", {
            "x": x,
            "y": y,
            "correlation": float(value),
            "row_count": len(data),
        }
    if action == "TREND_GROUP":
        group_match = re.search(
            r"\bby\s+(cut|color|clarity)\b|\bgroup.{0,20}\b(cut|color|clarity)\b", question, re.I
        )
        group = (
            next(item for item in group_match.groups() if item).casefold()
            if group_match
            else "clarity"
        )
        if metric in {"q25", "q75"}:
            quantile = 0.25 if metric == "q25" else 0.75
            values = data.groupby(group)[target].quantile(quantile).sort_values().to_dict()
        else:
            values = data.groupby(group)[target].agg(metric).sort_values().to_dict()
        summary = "; ".join(f"{key}: {value:,.2f}" for key, value in values.items())
        return f"{metric.title()} {target.replace('_', ' ')} by {group}: {summary}.", {
            "group_by": group,
            "target": target,
            "metric": metric,
            "values": values,
        }
    if metric == "count":
        return f"There are {len(data):,} matching diamonds.", {
            "metric": "count",
            "value": len(data),
        }
    value = _aggregate(data[target], metric)
    return (
        f"The {metric} {target.replace('_', ' ')} is {value:,.2f} across {len(data):,} matching rows.",
        {"metric": metric, "target": target, "value": float(value), "row_count": len(data)},
    )
