"""Interpret discovered clusters as product-derived buyer-preference archetypes."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

from .feature_engineering import CUSTOMER_PILLARS
from .models import SegmentProfile

FRIENDLY_PILLAR_NAMES = {
    "Pillar_VisualPresence": "visible size / visual impact",
    "Pillar_Quality": "certificate quality",
    "Pillar_Value": "value for money",
    "Pillar_PriceLevel": "higher market tier",
    "Pillar_QualityBalance": "balanced quality",
    "Pillar_Proportion": "proportion consistency",
    "Pillar_Rarity": "rarity / uniqueness",
    "Pillar_Milestone": "carat milestone positioning",
}


def _market_tier(price_percentile: float) -> str:
    if price_percentile < 33.0:
        return "lower-price / budget-oriented"
    if price_percentile < 67.0:
        return "mid-market"
    return "higher-price / premium-market"


def _archetype_name(profile_delta: pd.Series) -> str:
    """Assign a descriptive label from the actual profile pattern.

    The label is downstream interpretation. It is not used to create clusters.
    """
    visual = profile_delta["Pillar_VisualPresence"]
    quality = profile_delta["Pillar_Quality"]
    value = profile_delta["Pillar_Value"]
    price = profile_delta["Pillar_PriceLevel"]
    balance = profile_delta["Pillar_QualityBalance"]
    proportion = profile_delta["Pillar_Proportion"]
    rarity = profile_delta["Pillar_Rarity"]
    milestone = profile_delta["Pillar_Milestone"]

    if visual > 0.35 and price > 0.30:
        return "Premium Visual-Impact Seekers"
    if quality > 0.35 and price < -0.20:
        return "Quality-Conscious Budget Buyers"
    if milestone > 0.35 and value > 0.15:
        return "Milestone-Conscious Value Buyers"
    if rarity > 0.40 and balance > 0.15:
        return "Distinctive Balanced Pragmatists"
    if rarity > 0.40:
        return "Rarity / Distinctiveness Seekers"
    if value > 0.40 and price < 0.0:
        return "Budget Value Seekers"
    if quality > 0.40:
        return "Quality-First Buyers"
    if visual > 0.40:
        return "Visual-Impact Seekers"
    if balance > 0.30 and proportion > 0.20:
        return "Balanced Mid-Market Pragmatists"
    return "Balanced Mid-Market Pragmatists"


def _mode(series: pd.Series) -> object:
    values = series.mode()
    return values.iloc[0] if len(values) else None


def build_profiles(
    df: pd.DataFrame,
    scaled_features: np.ndarray,
    labels: np.ndarray,
) -> Tuple[
    List[SegmentProfile],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build segment profiles and pairwise profile-distance information."""
    profiled = df.copy()
    profiled["cluster_id"] = labels

    medians = profiled.groupby("cluster_id")[list(CUSTOMER_PILLARS)].median()

    overall_median = profiled[list(CUSTOMER_PILLARS)].median()
    overall_iqr = (
        profiled[list(CUSTOMER_PILLARS)].quantile(0.75)
        - profiled[list(CUSTOMER_PILLARS)].quantile(0.25)
    ).replace(0.0, np.nan)

    deltas = (medians - overall_median) / overall_iqr

    cluster_ids = list(medians.index)
    centers = np.vstack(
        [scaled_features[labels == cluster_id].mean(axis=0) for cluster_id in cluster_ids]
    )

    matrix = pd.DataFrame(
        euclidean_distances(centers, centers),
        index=cluster_ids,
        columns=cluster_ids,
    )

    pair_rows = []
    for i, cluster_a in enumerate(cluster_ids):
        for j in range(i + 1, len(cluster_ids)):
            cluster_b = cluster_ids[j]
            pair_rows.append(
                {
                    "cluster_a": int(cluster_a),
                    "cluster_b": int(cluster_b),
                    "distance": float(matrix.loc[cluster_a, cluster_b]),
                }
            )
    distance_pairs = pd.DataFrame(pair_rows).sort_values(
        "distance",
        ascending=False,
    )

    profiles: List[SegmentProfile] = []

    for cluster_id in cluster_ids:
        cluster_rows = profiled.loc[profiled["cluster_id"] == cluster_id]
        # Constant pillars have a zero IQR and therefore no power to define a
        # cluster. Treat their relative delta as neutral instead of removing
        # the field required by the interpretation layer.
        delta = deltas.loc[cluster_id].reindex(CUSTOMER_PILLARS).fillna(0.0)

        priorities = [
            FRIENDLY_PILLAR_NAMES[name] for name in delta.sort_values(ascending=False).head(3).index
        ]
        tradeoffs = [
            FRIENDLY_PILLAR_NAMES[name] for name in delta.sort_values(ascending=True).head(2).index
        ]

        distances = matrix.loc[cluster_id].drop(cluster_id)
        nearest_id = int(distances.idxmin()) if not distances.empty else None
        furthest_id = int(distances.idxmax()) if not distances.empty else None

        name = _archetype_name(delta)
        share = len(cluster_rows) / len(profiled) * 100.0

        typical_diamond = {
            "carat": float(cluster_rows["carat"].median()),
            "price": float(cluster_rows["price"].median()),
            "cut": _mode(cluster_rows["cut"]),
            "color": _mode(cluster_rows["color"]),
            "clarity": _mode(cluster_rows["clarity_5"]),
            "carat_band": _mode(cluster_rows["CaratBand"]),
        }

        pillar_scores = {
            key.replace("Pillar_", "").lower(): float(value)
            for key, value in medians.loc[cluster_id].items()
        }

        summary = (
            f"May appeal to buyers who prioritize {priorities[0]} "
            f"and {priorities[1]}, while accepting relatively less "
            f"emphasis on {tradeoffs[0]}."
        )

        profiles.append(
            SegmentProfile(
                cluster_id=int(cluster_id),
                name=name,
                segment_share_pct=float(share),
                market_tier=_market_tier(
                    float(
                        medians.loc[
                            cluster_id,
                            "Pillar_PriceLevel",
                        ]
                    )
                ),
                top_priorities=priorities,
                tradeoffs=tradeoffs,
                pillar_scores=pillar_scores,
                typical_diamond=typical_diamond,
                summary=summary,
                nearest_profile_id=nearest_id,
                most_distant_profile_id=furthest_id,
            )
        )

    # Resolve duplicate human-readable names without changing cluster semantics.
    name_counts: Dict[str, int] = {}
    for profile in profiles:
        name_counts[profile.name] = name_counts.get(profile.name, 0) + 1

    resolved_profiles: List[SegmentProfile] = []
    for profile in profiles:
        if name_counts[profile.name] > 1:
            profile = SegmentProfile(
                **{
                    **profile.__dict__,
                    "name": f"{profile.name} — Cluster {profile.cluster_id}",
                }
            )
        resolved_profiles.append(profile)

    return resolved_profiles, medians, distance_pairs


def attach_profile_names(
    df: pd.DataFrame,
    labels: np.ndarray,
    profiles: List[SegmentProfile],
) -> pd.DataFrame:
    """Attach cluster ID and profile name to each segmented diamond."""
    mapping = {profile.cluster_id: profile.name for profile in profiles}

    out = df.copy()
    out["cluster_id"] = labels
    out["customer_profile_name"] = out["cluster_id"].map(mapping)
    return out
