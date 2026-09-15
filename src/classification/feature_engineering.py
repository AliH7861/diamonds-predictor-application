"""
Feature definitions and engineered inputs for the diamond classification experiments.

This file defines the categorical and numeric features used by each experiment.

The experiments gradually add more information:

Experiment 1:
Physical diamond measurements and engineered geometry features.

Experiment 2:
Experiment 1 features plus the raw diamond price.

Experiment 3:
Experiment 2 features plus price-relative features such as price per carat,
price per volume, and price per visible face area.

The engineered physical features help describe diamond size, shape,
proportions, density-like relationships, and dimensional symmetry.
"""

import numpy as np

# FEATURE GROUPS

# Categorical diamond characteristics used by every experiment.
CATEGORICAL_FEATURES = ["cut", "color"]


# Raw physical measurements plus engineered geometry relationships.

# Raw measurements:
# carat = diamond weight
# depth = total depth percentage
# table = width of the top facet
# x, y, z = physical diamond dimensions

# REG_ features are engineered from those measurements.
PHYSICAL_FEATURES = [
    "carat", "depth", "table", "x", "y", "z",
    "REG_FaceArea", "REG_Volume", "REG_AspectRatio", "REG_XYAsymmetry",
    "REG_DepthToFace", "REG_WeightPerFace", "REG_WeightPerVolume",
    "REG_TableDepthRatio"
]

# Raw market price is introduced separately so we can test whether
# price improves clarity prediction beyond physical measurements alone.
RAW_PRICE_FEATURES = ["price"]


# Price-relative features describe how expensive a diamond is compared
# with its weight and physical dimensions.
PRICE_RELATIVE_FEATURES = [
    "PricePerCarat", "PricePerVolume",
    "PricePerFaceArea", "PricePerFaceDiagonal"
]

# EXPERIMENT FEATURE SETS

# Experiment 1 asks:
# How well can clarity be predicted using physical information only?
EXPERIMENT_1_NUMERIC = PHYSICAL_FEATURES

# Experiment 2 asks:
# Does adding raw price improve clarity prediction?
EXPERIMENT_2_NUMERIC = PHYSICAL_FEATURES + RAW_PRICE_FEATURES

# Experiment 3 asks:
# Do price-to-size relationships provide additional clarity information?
EXPERIMENT_3_NUMERIC = PHYSICAL_FEATURES + RAW_PRICE_FEATURES + PRICE_RELATIVE_FEATURES


# FEATURE ENGINEERING
def engineer_features(df):
    """Create physical and price-relative features from the raw diamond data."""

    # Work on a copy so the original DataFrame is never changed.
    df = df.copy()

    # PHYSICAL / GEOMETRY FEATURES

    # Approximate visible face area using diamond width × length.
    # Larger values represent a larger top-facing physical footprint.
    df["REG_FaceArea"] = df["x"] * df["y"]

    # Approximate physical volume using all three dimensions.
    # This is not the exact geometric volume of a diamond, but it gives
    # the model a useful size relationship between x, y, and z.
    df["REG_Volume"] = df["x"] * df["y"] * df["z"]

    # Compare x and y dimensions to describe the diamond's shape.
    # Values close to 1 mean x and y are very similar.
    df["REG_AspectRatio"] = df["x"] / df["y"].replace(0, np.nan)

    # Measure absolute difference between x and y.
    # Larger values indicate greater dimensional asymmetry.
    df["REG_XYAsymmetry"] = abs(df["x"] - df["y"])

    # Calculate the diagonal across the x-y face.
    # This gives another overall measurement of the diamond's visible size.
    face_diagonal = np.sqrt(df["x"]**2 + df["y"]**2)

    # Compare diamond depth with its visible face size.
    # This helps represent whether a diamond is relatively deep or shallow.
    df["REG_DepthToFace"] = df["z"] / face_diagonal.replace(0, np.nan)

    # Compare carat weight with visible face area.
    # Two diamonds with similar carat weights may distribute that weight differently.
    df["REG_WeightPerFace"] = df["carat"] / df["REG_FaceArea"].replace(0, np.nan)

    # Compare carat weight with estimated physical volume.
    # This captures another relationship between weight and physical dimensions.
    df["REG_WeightPerVolume"] = df["carat"] / df["REG_Volume"].replace(0, np.nan)

    # Compare table percentage with depth percentage.
    # This gives the model another representation of diamond proportions.
    df["REG_TableDepthRatio"] = df["table"] / df["depth"].replace(0, np.nan)


    # PRICE-RELATIVE FEATURES
    # Price per carat measures how much value is associated with each unit of weight.
    df["PricePerCarat"] = df["price"] / df["carat"].replace(0, np.nan)

    # Price relative to estimated physical volume.
    df["PricePerVolume"] = df["price"] / df["REG_Volume"].replace(0, np.nan)

    # Price relative to the diamond's visible face area.
    df["PricePerFaceArea"] = df["price"] / df["REG_FaceArea"].replace(0, np.nan)

    # Price relative to the diagonal size of the diamond's visible face.
    df["PricePerFaceDiagonal"] = df["price"] / face_diagonal.replace(0, np.nan)

    # Division by zero can create positive or negative infinity.
    # Convert these values to NaN so the preprocessing pipeline can handle them safely.
    return df.replace([np.inf, -np.inf], np.nan)

