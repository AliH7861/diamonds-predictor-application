"""Segmentation-specific feature engineering.

The host application owns data loading. This module receives an existing
diamonds DataFrame and builds only the features needed for segmentation.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import SegmentationConfig

EPS = 1e-9

REQUIRED_COLUMNS = (
    "carat",
    "cut",
    "color",
    "clarity",
    "depth",
    "table",
    "price",
    "x",
    "y",
    "z",
)

CUSTOMER_PILLARS = (
    "Pillar_VisualPresence",
    "Pillar_Quality",
    "Pillar_Value",
    "Pillar_PriceLevel",
    "Pillar_QualityBalance",
    "Pillar_Proportion",
    "Pillar_Rarity",
    "Pillar_Milestone",
)

PILLAR_QUESTIONS: Dict[str, str] = {
    "Pillar_VisualPresence": "How much visible size / visual impact does the diamond provide?",
    "Pillar_Quality": "How strongly does the diamond prioritize cut, color, and clarity?",
    "Pillar_Value": "How much size / quality does the buyer receive for the price?",
    "Pillar_PriceLevel": "What budget / market tier does the diamond occupy?",
    "Pillar_QualityBalance": "Are cut, color, and clarity reasonably balanced?",
    "Pillar_Proportion": "Are the physical proportions relatively typical / balanced?",
    "Pillar_Rarity": "How unusual is the diamond configuration?",
    "Pillar_Milestone": "How close is the diamond to a meaningful carat threshold?",
}


def validate_input(df: pd.DataFrame) -> None:
    """Validate that the host DataFrame contains the required base columns."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Segmentation requires the following missing columns: " + ", ".join(missing)
        )


def _percentile_rank(series: pd.Series) -> pd.Series:
    """Return percentile ranks on a 0-100 scale."""
    return series.rank(pct=True, method="average") * 100.0


def _coerce_clean_base(df: pd.DataFrame) -> pd.DataFrame:
    """Return a segmentation-safe copy without mutating the host DataFrame."""
    out = df.copy()

    numeric_columns = ("carat", "depth", "table", "price", "x", "y", "z")
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    for column in ("cut", "color", "clarity"):
        out[column] = out[column].astype(str).str.strip()

    out = out.dropna(subset=list(REQUIRED_COLUMNS)).copy()

    valid = (
        (out["carat"] > 0)
        & (out["price"] > 0)
        & (out["x"] > 0)
        & (out["y"] > 0)
        & (out["z"] > 0)
        & (out["depth"] > 0)
        & (out["table"] > 0)
    )
    return out.loc[valid].reset_index(drop=True)


def build_segmentation_features(
    df: pd.DataFrame,
    config: SegmentationConfig,
) -> pd.DataFrame:
    """Build customer-oriented segmentation features from a host DataFrame.

    The function intentionally converts raw diamond measurements into
    interpretable preference dimensions instead of clustering directly on
    x/y/z and other raw geometry measurements.
    """
    validate_input(df)
    out = _coerce_clean_base(df)

    clarity_5_map = {
        "I1": "I",
        "SI2": "SI",
        "SI1": "SI",
        "VS2": "VS",
        "VS1": "VS",
        "VVS2": "VVS",
        "VVS1": "VVS",
        "IF": "IF",
    }
    cut_score_map = {
        "Fair": 1,
        "Good": 2,
        "Very Good": 3,
        "Premium": 4,
        "Ideal": 5,
    }
    color_score_map = {
        "J": 1,
        "I": 2,
        "H": 3,
        "G": 4,
        "F": 5,
        "E": 6,
        "D": 7,
    }
    clarity_score_map = {
        "I": 1,
        "SI": 2,
        "VS": 3,
        "VVS": 4,
        "IF": 5,
    }

    out["clarity_5"] = out["clarity"].map(clarity_5_map)
    out["CutScore"] = out["cut"].map(cut_score_map)
    out["ColorScore"] = out["color"].map(color_score_map)
    out["ClarityScore"] = out["clarity_5"].map(clarity_score_map)

    if out[["CutScore", "ColorScore", "ClarityScore"]].isna().any().any():
        raise ValueError("Unexpected cut/color/clarity category found in segmentation input.")

    # Geometry ingredients.
    out["AvgDiameter"] = (out["x"] + out["y"]) / 2.0
    out["FaceArea"] = (np.pi / 4.0) * out["x"] * out["y"]
    out["AspectRatio"] = np.maximum(out["x"], out["y"]) / np.maximum(
        np.minimum(out["x"], out["y"]), EPS
    )
    out["XYAsymmetryPct"] = (
        np.abs(out["x"] - out["y"]) / np.maximum(out["AvgDiameter"], EPS) * 100.0
    )
    out["HeightToDiameter"] = out["z"] / np.maximum(out["AvgDiameter"], EPS)

    # Carat bands and milestone distance.
    bins = [-np.inf, 0.50, 0.70, 0.90, 1.00, 1.25, 1.50, 2.00, np.inf]
    labels = [
        "<0.50",
        "0.50-0.69",
        "0.70-0.89",
        "0.90-0.99",
        "1.00-1.24",
        "1.25-1.49",
        "1.50-1.99",
        "2.00+",
    ]
    out["CaratBand"] = pd.cut(
        out["carat"],
        bins=bins,
        labels=labels,
        right=False,
    )

    magic_sizes = np.array([0.50, 0.75, 1.00, 1.50, 2.00])
    carats = out["carat"].to_numpy()
    nearest_idx = np.argmin(
        np.abs(carats[:, None] - magic_sizes[None, :]),
        axis=1,
    )
    out["AbsDistanceToMagicSize"] = np.abs(out["carat"] - magic_sizes[nearest_idx])

    # Visible size relative to similar carat.
    out["CaratPeerBin"] = pd.qcut(
        out["carat"],
        q=25,
        duplicates="drop",
    )
    expected_face = out.groupby("CaratPeerBin", observed=True)["FaceArea"].transform("median")
    expected_diameter = out.groupby("CaratPeerBin", observed=True)["AvgDiameter"].transform(
        "median"
    )
    out["FaceAreaResidualPct"] = (
        (out["FaceArea"] - expected_face) / np.maximum(expected_face, EPS) * 100.0
    )
    out["DiameterResidualPct"] = (
        (out["AvgDiameter"] - expected_diameter) / np.maximum(expected_diameter, EPS) * 100.0
    )

    # Quality and balance.
    quality_array = np.column_stack(
        [
            (out["CutScore"] - 1) / 4.0,
            (out["ColorScore"] - 1) / 6.0,
            (out["ClarityScore"] - 1) / 4.0,
        ]
    )
    out["OverallQualityScore"] = quality_array.mean(axis=1) * 100.0

    quality_std = quality_array.std(axis=1)
    quality_ref = max(float(np.quantile(quality_std, 0.99)), EPS)
    out["QualityBalanceScore"] = 100.0 * (1.0 - np.clip(quality_std / quality_ref, 0.0, 1.0))

    # Dataset-relative proportion typicality.
    depth_iqr = max(
        out["depth"].quantile(0.75) - out["depth"].quantile(0.25),
        EPS,
    )
    table_iqr = max(
        out["table"].quantile(0.75) - out["table"].quantile(0.25),
        EPS,
    )
    depth_dev = np.abs(out["depth"] - out["depth"].median()) / depth_iqr
    table_dev = np.abs(out["table"] - out["table"].median()) / table_iqr
    proportion_dev = (depth_dev + table_dev) / 2.0
    proportion_ref = max(
        float(proportion_dev.quantile(0.99)),
        EPS,
    )
    out["ProportionTypicalityScore"] = 100.0 * (
        1.0
        - np.clip(
            proportion_dev / proportion_ref,
            0.0,
            1.0,
        )
    )

    # Hierarchical peer value.
    def peer_stats(columns: Sequence[str]) -> Tuple[pd.Series, pd.Series]:
        grouped = out.groupby(list(columns), observed=True)["price"]
        return grouped.transform("median"), grouped.transform("size")

    peer1_price, peer1_n = peer_stats(("CaratBand", "cut", "color", "clarity_5"))
    peer2_price, peer2_n = peer_stats(("CaratBand", "cut", "clarity_5"))
    peer3_price, peer3_n = peer_stats(("CaratBand", "clarity_5"))
    peer4_price, peer4_n = peer_stats(("CaratBand",))

    min_group = config.min_peer_group_size
    global_median = out["price"].median()

    out["ExpectedPeerPrice"] = np.where(
        peer1_n >= min_group,
        peer1_price,
        np.where(
            peer2_n >= min_group,
            peer2_price,
            np.where(
                peer3_n >= min_group,
                peer3_price,
                np.where(
                    peer4_n >= min_group,
                    peer4_price,
                    global_median,
                ),
            ),
        ),
    )
    out["PeerValuePct"] = (
        (out["ExpectedPeerPrice"] - out["price"])
        / np.maximum(out["ExpectedPeerPrice"], EPS)
        * 100.0
    )

    # Visible size and quality received for the current price level.
    out["PricePeerBin"] = pd.qcut(
        out["price"],
        q=25,
        duplicates="drop",
    )
    face_by_price = out.groupby("PricePeerBin", observed=True)["FaceArea"].transform("median")
    quality_by_price = out.groupby("PricePeerBin", observed=True)["OverallQualityScore"].transform(
        "median"
    )
    out["VisibleSizeForPrice"] = (
        (out["FaceArea"] - face_by_price) / np.maximum(face_by_price, EPS) * 100.0
    )
    out["QualityForPrice"] = out["OverallQualityScore"] - quality_by_price

    # Rarity.
    configuration_count = out.groupby(
        ["CaratBand", "cut", "color", "clarity_5"],
        observed=True,
    )["carat"].transform("size")
    out["ConfigurationRarityRaw"] = -np.log(np.maximum(configuration_count / len(out), EPS))

    # Customer-oriented percentile positions.
    out["VisibleSizePosition"] = _percentile_rank(out["FaceArea"])
    out["VisibleSizeEfficiency"] = _percentile_rank(out["FaceAreaResidualPct"])
    out["PricePosition"] = _percentile_rank(out["price"])
    out["PeerValuePosition"] = _percentile_rank(out["PeerValuePct"])
    out["VisibleSizeValuePosition"] = _percentile_rank(out["VisibleSizeForPrice"])
    out["QualityValuePosition"] = _percentile_rank(out["QualityForPrice"])
    out["CutPosition"] = _percentile_rank(out["CutScore"])
    out["ColorPosition"] = _percentile_rank(out["ColorScore"])
    out["ClarityPosition"] = _percentile_rank(out["ClarityScore"])
    out["MagicSizeCloseness"] = 100.0 - _percentile_rank(out["AbsDistanceToMagicSize"])
    out["ConfigurationRarityPct"] = _percentile_rank(out["ConfigurationRarityRaw"])

    return out


def split_geometry_anomalies(
    df: pd.DataFrame,
    config: SegmentationConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separate likely measurement anomalies from segmentation candidates."""
    out = df.copy()

    geometry_features = (
        "AspectRatio",
        "XYAsymmetryPct",
        "HeightToDiameter",
        "FaceAreaResidualPct",
        "DiameterResidualPct",
    )

    out["GeometryExtremeCount"] = 0

    for feature in geometry_features:
        low = out[feature].quantile(config.geometry_low_quantile)
        high = out[feature].quantile(config.geometry_high_quantile)
        out["GeometryExtremeCount"] += ((out[feature] < low) | (out[feature] > high)).astype(int)

    aspect_cap = out["AspectRatio"].quantile(config.aspect_ratio_hard_quantile)

    out["GeometryAnomaly"] = (out["GeometryExtremeCount"] >= config.min_extreme_geometry_flags) | (
        out["AspectRatio"] > aspect_cap
    )

    anomalies = out.loc[out["GeometryAnomaly"]].copy()
    segmentation = out.loc[~out["GeometryAnomaly"]].copy().reset_index(drop=True)
    return segmentation, anomalies


def build_customer_pillars(df: pd.DataFrame) -> pd.DataFrame:
    """Build the eight interpretable buyer-preference dimensions."""
    out = df.copy()

    out["Pillar_VisualPresence"] = out[["VisibleSizePosition", "VisibleSizeEfficiency"]].mean(
        axis=1
    )

    out["Pillar_Quality"] = out[["CutPosition", "ColorPosition", "ClarityPosition"]].mean(axis=1)

    out["Pillar_Value"] = out[
        [
            "PeerValuePosition",
            "VisibleSizeValuePosition",
            "QualityValuePosition",
        ]
    ].mean(axis=1)

    out["Pillar_PriceLevel"] = out["PricePosition"]
    out["Pillar_QualityBalance"] = out["QualityBalanceScore"]
    out["Pillar_Proportion"] = out["ProportionTypicalityScore"]
    out["Pillar_Rarity"] = out["ConfigurationRarityPct"]
    out["Pillar_Milestone"] = out["MagicSizeCloseness"]

    return out
