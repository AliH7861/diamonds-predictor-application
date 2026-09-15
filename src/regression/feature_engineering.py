"""
Feature engineering utilities for the diamond price-regression experiments.

This module creates physically interpretable geometry features, ordered quality
scores, market-size features, peer-relative features, and rarity features used
by the maintained regression experiments.

Some features depend only on the current diamond, while others use context
learned from training rows only so validation and test data do not leak
information into feature generation.

Price is never used as an input feature.
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


# Ordered quality mappings used to convert categorical grades into numeric scores.
CUT_SCORE = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
COLOR_SCORE = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
CLARITY_SCORE = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}

# Carat bands and common market threshold sizes.
CARAT_BINS = [0.0, 0.5, 0.7, 0.9, 1.0, 1.25, 1.5, 2.0, np.inf]
CARAT_LABELS = ["<0.50", "0.50-0.69", "0.70-0.89", "0.90-0.99", "1.00-1.24", "1.25-1.49", "1.50-1.99", "2.00+"]
MAGIC_SIZES = np.array([0.5, 0.75, 1.0, 1.5, 2.0], dtype=float)

# Small value used to prevent division by zero.
EPSILON = 1e-8


def add_basic_geometry(data):
    """Create physically interpretable geometry features from raw diamond dimensions."""

    result = data.copy()

    # Basic visible size and volume measurements.
    result["H_AvgDiameter"] = (result["x"] + result["y"]) / 2
    result["H_FaceArea"] = result["x"] * result["y"]
    result["H_FaceDiagonal"] = np.sqrt(result["x"] ** 2 + result["y"] ** 2)
    result["H_Volume"] = result["x"] * result["y"] * result["z"]

    # Shape and symmetry relationships.
    result["H_AspectRatio"] = np.maximum(result["x"], result["y"]) / np.maximum(np.minimum(result["x"], result["y"]), EPSILON)
    result["H_XYAsymmetryPct"] = np.abs(result["x"] - result["y"]) / np.maximum(result["H_AvgDiameter"], EPSILON) * 100

    # Depth and proportion relationships.
    result["H_HeightToDiameter"] = result["z"] / np.maximum(result["H_AvgDiameter"], EPSILON)
    result["H_DepthToFace"] = result["z"] / np.maximum(np.sqrt(result["H_FaceArea"]), EPSILON)
    result["H_TableDepthRatio"] = result["table"] / np.maximum(result["depth"], EPSILON)

    # Weight relative to visible size and estimated volume.
    result["H_FaceAreaPerCarat"] = result["H_FaceArea"] / np.maximum(result["carat"], EPSILON)
    result["H_CaratPerVolume"] = result["carat"] / np.maximum(result["H_Volume"], EPSILON)
    result["H_VolumePerCarat"] = result["H_Volume"] / np.maximum(result["carat"], EPSILON)

    return result


def fit_human_context(train_data):
    """Learn size expectations, peer medians, and rarity information from training rows."""

    # Add the physical features needed for training-derived context.
    train = add_basic_geometry(train_data)

    # Group diamonds into size ranges used for peer comparisons.
    train["H_CaratBand"] = pd.cut(
        train["carat"], bins=CARAT_BINS, labels=CARAT_LABELS,
        right=False, include_lowest=True
    ).astype(str)

    # Learn expected face area and volume as smooth functions of carat.
    face_model = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), Ridge(alpha=1.0))
    face_model.fit(train[["carat"]], train["H_FaceArea"])

    volume_model = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), Ridge(alpha=1.0))
    volume_model.fit(train[["carat"]], train["H_Volume"])

    # Learn typical depth/table values for similar carat-band and cut combinations.
    peer_profiles = train.groupby(["H_CaratBand", "cut"], observed=True).agg(
        ExpectedDepth=("depth", "median"),
        ExpectedTable=("table", "median")
    )

    # Count similar diamonds so rarity can be estimated later.
    rarity_counts = train.groupby(["H_CaratBand", "cut", "color", "clarity"], observed=True).size().to_dict()

    return {
        "face_model": face_model, "volume_model": volume_model,
        "peer_profiles": peer_profiles, "rarity_counts": rarity_counts,
        "total_rows": len(train), "global_depth": float(train["depth"].median()),
        "global_table": float(train["table"].median())
    }

def transform_human_features(data, context):
    """Apply the saved training context and generate the human-style regression features."""

    result = add_basic_geometry(data)

    # Convert ordered quality grades into numeric quality scores.
    result["H_CutScore"] = result["cut"].map(CUT_SCORE).astype(float)
    result["H_ColorScore"] = result["color"].map(COLOR_SCORE).astype(float)
    result["H_ClarityScore"] = result["clarity"].map(CLARITY_SCORE).astype(float)

    # Combine cut, color, and clarity into one normalized quality score.
    result["H_QualityBundle"] = (
        result["H_CutScore"] / 4
        + result["H_ColorScore"] / 6
        + result["H_ClarityScore"] / 7
    ) / 3

    # Assign each diamond to its market-size carat band.
    result["H_CaratBand"] = pd.cut(
        result["carat"], bins=CARAT_BINS, labels=CARAT_LABELS,
        right=False, include_lowest=True
    ).astype(str)

    # Measure distance from common carat thresholds such as 0.5, 1.0, and 2.0 ct.
    carat_values = result["carat"].to_numpy()
    distances = np.abs(carat_values[:, None] - MAGIC_SIZES[None, :])
    nearest_index = np.argmin(distances, axis=1)
    nearest_magic = MAGIC_SIZES[nearest_index]
    signed_distance = carat_values - nearest_magic

    result["H_DistanceToMagicSize"] = signed_distance
    result["H_AbsDistanceToMagicSize"] = np.abs(signed_distance)
    result["H_JustAboveMagicSize"] = ((signed_distance >= 0) & (signed_distance <= 0.05)).astype(int)
    result["H_JustBelowMagicSize"] = ((signed_distance < 0) & (signed_distance >= -0.05)).astype(int)

    # Create threshold indicators for each common market size.
    for magic_size in MAGIC_SIZES:
        name = str(magic_size).replace(".", "_")
        result[f"H_AtLeast_{name}ct"] = (result["carat"] >= magic_size).astype(int)

    # Compare observed face area and volume with training-derived expectations.
    expected_face = context["face_model"].predict(result[["carat"]])
    result["H_ExpectedFaceArea"] = expected_face
    result["H_FaceAreaResidualPct"] = (result["H_FaceArea"] - expected_face) / np.maximum(expected_face, EPSILON) * 100

    expected_volume = context["volume_model"].predict(result[["carat"]])
    result["H_ExpectedVolume"] = expected_volume
    result["H_VolumeResidualPct"] = (result["H_Volume"] - expected_volume) / np.maximum(expected_volume, EPSILON) * 100

    # Compare depth and table with diamonds of similar size and cut.
    peer_index = pd.MultiIndex.from_arrays([result["H_CaratBand"], result["cut"]])
    expected_depth = context["peer_profiles"]["ExpectedDepth"].reindex(peer_index).to_numpy()
    expected_table = context["peer_profiles"]["ExpectedTable"].reindex(peer_index).to_numpy()

    expected_depth = np.where(pd.isna(expected_depth), context["global_depth"], expected_depth)
    expected_table = np.where(pd.isna(expected_table), context["global_table"], expected_table)

    result["H_DepthDeviation"] = result["depth"].to_numpy() - expected_depth
    result["H_TableDeviation"] = result["table"].to_numpy() - expected_table

    # Capture interactions between diamond size and quality grades.
    log_size = np.log1p(result["carat"])
    result["H_ClarityAtSize"] = result["H_ClarityScore"] * log_size
    result["H_ColorAtSize"] = result["H_ColorScore"] * log_size
    result["H_CutAtSize"] = result["H_CutScore"] * log_size

    # Rare training peer groups receive larger rarity scores.
    rarity_values = []
    rarity_counts = context["rarity_counts"]
    total_rows = context["total_rows"]

    for row in result[["H_CaratBand", "cut", "color", "clarity"]].itertuples(index=False, name=None):
        count = rarity_counts.get(row, 0)
        frequency = (count + 1) / (total_rows + 1)
        rarity_values.append(-np.log(frequency))

    result["H_PeerRarity"] = rarity_values

    return result

# Numeric features used by the human-style experiment.
HUMAN_NUMERIC = [
    "carat", "H_CutScore", "H_ColorScore", "H_ClarityScore", "H_QualityBundle",
    "H_AvgDiameter", "H_FaceArea", "H_FaceDiagonal", "H_Volume",
    "H_AspectRatio", "H_XYAsymmetryPct", "H_HeightToDiameter", "H_DepthToFace",
    "H_TableDepthRatio", "H_FaceAreaPerCarat", "H_CaratPerVolume",
    "H_VolumePerCarat", "H_FaceAreaResidualPct", "H_VolumeResidualPct",
    "H_DepthDeviation", "H_TableDeviation", "H_DistanceToMagicSize",
    "H_AbsDistanceToMagicSize", "H_JustAboveMagicSize", "H_JustBelowMagicSize",
    "H_AtLeast_0_5ct", "H_AtLeast_0_75ct", "H_AtLeast_1_0ct",
    "H_AtLeast_1_5ct", "H_AtLeast_2_0ct", "H_ClarityAtSize",
    "H_ColorAtSize", "H_CutAtSize", "H_PeerRarity"
]

# Categorical features retained by the human-style experiment.
HUMAN_CATEGORICAL = ["cut", "color", "clarity", "H_CaratBand"]

# Physical baseline features recovered from the original regression workflow.
BASELINE_NUMERIC = [
    "carat", "depth", "table", "x", "y", "z", "REG_FaceArea", "REG_Volume",
    "REG_AspectRatio", "REG_XYAsymmetry", "REG_DepthToFace",
    "REG_WeightPerFace", "REG_WeightPerVolume", "REG_TableDepthRatio"
]

BASELINE_CATEGORICAL = ["cut", "color", "clarity"]
RAW_GEOMETRY = ["depth", "table", "x", "y", "z"]

# Exact feature combinations used by the maintained regression experiments.
FEATURE_SETS = {
    "CURRENT_BASELINE": {
        "numeric": BASELINE_NUMERIC,
        "categorical": BASELINE_CATEGORICAL,
        "source": "raw"
    },
    "HUMAN_ONLY": {
        "numeric": HUMAN_NUMERIC,
        "categorical": HUMAN_CATEGORICAL,
        "source": "human"
    },
    "HUMAN_PLUS_RAW": {
        "numeric": HUMAN_NUMERIC + RAW_GEOMETRY,
        "categorical": HUMAN_CATEGORICAL,
        "source": "human"
    }
}

def engineer_base_features(df):
    """Create the physical engineered features used by the baseline regression experiment."""

    df = df.copy()

    # Physical size and shape relationships; price is never used here.
    df["REG_FaceArea"] = df["x"] * df["y"]
    df["REG_Volume"] = df["x"] * df["y"] * df["z"]
    df["REG_AspectRatio"] = np.maximum(df["x"], df["y"]) / np.maximum(np.minimum(df["x"], df["y"]), EPSILON)
    df["REG_XYAsymmetry"] = np.abs(df["x"] - df["y"]) / np.maximum((df["x"] + df["y"]) / 2, EPSILON)

    # Depth, weight, and proportion relationships.
    df["REG_DepthToFace"] = df["z"] / np.maximum(np.sqrt(df["REG_FaceArea"]), EPSILON)
    df["REG_WeightPerFace"] = df["carat"] / np.maximum(df["REG_FaceArea"], EPSILON)
    df["REG_WeightPerVolume"] = df["carat"] / np.maximum(df["REG_Volume"], EPSILON)
    df["REG_TableDepthRatio"] = df["table"] / np.maximum(df["depth"], EPSILON)

    return df