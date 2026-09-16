"""
Create human-readable statistical profiles and cautious buyer interpretations for K-Means segments.

This module summarizes each cluster using size, price, carat, common quality
characteristics, numeric feature statistics, and categorical distributions.

It also assigns a simple buyer-orientation label based on observed cluster
patterns such as price level, size, cut, and clarity rather than treating the
cluster label itself as a known customer identity.
"""

import pandas as pd
from .feature_engineering import CATEGORICAL_FEATURES, PROFILE_NUMERIC_FEATURES

def _mode(series: pd.Series) -> str:
    """Return the most common non-missing value in a series."""

    values = series.mode(dropna=True)

    return "Unknown" if values.empty else str(values.iloc[0])

def _quality_family(clarity: str) -> str:
    """Convert a detailed clarity grade into its broader clarity family."""

    for family in ("IF", "VVS", "VS", "SI", "I"):
        if clarity.upper().startswith(family):
            return family

    return clarity

def interpret_cluster(group: pd.DataFrame, overall: pd.DataFrame) -> tuple[str, str]:
    """Describe the observed cluster and assign a cautious buyer-orientation label."""

    # Compare the cluster's median price and carat with overall dataset thirds.
    price = group["price"].median()
    carat = group["carat"].median()

    price_quantiles = overall["price"].quantile([0.33, 0.67])
    carat_quantiles = overall["carat"].quantile([0.33, 0.67])

    price_tier = "affordable" if price <= price_quantiles.iloc[0] else "premium-priced" if price >= price_quantiles.iloc[1] else "mid-range"
    size_tier = "smaller" if carat <= carat_quantiles.iloc[0] else "larger" if carat >= carat_quantiles.iloc[1] else "medium-size"

    # Use the most common cut and broader clarity family to describe quality.
    cut = _mode(group["cut"])
    clarity = _quality_family(_mode(group["clarity"]))

    profile = f"{price_tier}, {size_tier} diamonds; mainly {cut} cut with {clarity} clarity"

    # Translate observed cluster characteristics into a cautious buyer-style label.
    if (
        size_tier == "larger"
        and clarity in {"SI", "I"}
        and cut != "Ideal"
    ):
        buyer = "Size-focused buyer"
    elif price_tier == "premium-priced" and size_tier == "larger":
        buyer = "Luxury-oriented buyer"
    elif clarity in {"VS", "VVS", "IF"} and cut in {"Ideal", "Premium"}:
        buyer = "Quality-focused buyer"
    elif price_tier == "affordable":
        buyer = "Value-focused buyer"
    else:
        buyer = "Balanced trade-off buyer"

    return profile, buyer

def build_cluster_profiles(frame: pd.DataFrame, labels, k: int) -> dict[str, pd.DataFrame]:
    """Create summary, numeric-statistic, categorical-distribution, and labeled cluster tables."""

    labeled = frame.copy()
    if "BUY_ClarityFamily" not in labeled and "clarity" in labeled:
        labeled["BUY_ClarityFamily"] = labeled["clarity"].map(_quality_family)
    labeled["Cluster"] = labels

    summaries = []
    numeric_rows = []
    categorical_rows = []

    # Build separate statistical profiles for every discovered cluster.
    for cluster, group in labeled.groupby("Cluster", sort=True):
        profile, buyer = interpret_cluster(group, labeled)

        # High-level cluster summary used for interpretation and reporting.
        summaries.append({
            "K": k, "Cluster": int(cluster), "Size": len(group),
            "Percentage": len(group) / len(labeled) * 100,
            "Median_Price": group["price"].median(),
            "Price_Min": group["price"].min(), "Price_Max": group["price"].max(),
            "Median_Carat": group["carat"].median(),
            "Main_Cut": _mode(group["cut"]), "Main_Color": _mode(group["color"]),
            "Main_Clarity": _quality_family(_mode(group["clarity"])),
            "Purchase_Profile": profile, "Buyer_Interpretation": buyer
        })

        # Store detailed numeric statistics for every clustering feature.
        for feature in PROFILE_NUMERIC_FEATURES:
            values = group[feature]

            numeric_rows.append({
                "K": k, "Cluster": int(cluster), "Feature": feature,
                "Mean": values.mean(), "Median": values.median(),
                "Min": values.min(), "Max": values.max(),
                "Range": values.max() - values.min(), "Std": values.std()
            })

        # Store category counts and proportions for cut, color, and clarity.
        for feature in CATEGORICAL_FEATURES:
            counts = group[feature].value_counts(dropna=False)

            for category, count in counts.items():
                categorical_rows.append({
                    "K": k, "Cluster": int(cluster), "Feature": feature,
                    "Category": str(category), "Count": int(count),
                    "Percentage": count / len(group) * 100,
                    "Is_Most_Common": category == counts.index[0]
                })

    return {
        "summary": pd.DataFrame(summaries),
        "numeric": pd.DataFrame(numeric_rows),
        "categorical": pd.DataFrame(categorical_rows),
        "labeled": labeled
    }
