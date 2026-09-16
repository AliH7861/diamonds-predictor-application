"""
Feature definitions and purchase-profile engineering for diamond clustering experiments.

This module defines the shared numeric and categorical inputs used by every
K-Means experiment and creates additional purchase-oriented features describing
diamond size, shape, volume, and price relative to carat weight.

These features are designed to represent different buyer profiles without using
any target variable, since clustering is an unsupervised learning task.
"""

import numpy as np
import pandas as pd

# Compact model inputs. Log transforms reduce the effect of the long price and
# size tails, while face area and volume replace repeatedly counting x/y/z.
NUMERIC_FEATURES = [
    "BUY_LogPrice", "BUY_LogCarat", "depth", "table",
    "BUY_LogFaceArea", "BUY_LogVolume", "BUY_AspectRatio",
    "BUY_LogPricePerCarat",
]

# Human-readable values retained in the exported statistical profile tables.
PROFILE_NUMERIC_FEATURES = [
    "price", "carat", "depth", "table", "x", "y", "z",
    "BUY_FaceArea", "BUY_Volume", "BUY_AspectRatio", "BUY_PricePerCarat",
]

# Categorical quality characteristics used when building customer-style segments.
CATEGORICAL_FEATURES = ["cut", "color", "BUY_ClarityFamily"]

# Complete feature set used by the clustering pipeline.
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def engineer_purchase_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create purchase-profile features from the raw diamond columns."""
    result = frame.copy()

    # Approximate the visible face size and overall physical volume.
    result["BUY_FaceArea"] = result["x"] * result["y"]
    result["BUY_Volume"] = result["x"] * result["y"] * result["z"]

    # Describe shape and the amount paid relative to diamond weight.
    result["BUY_AspectRatio"] = result["x"] / result["y"].replace(0, np.nan)
    result["BUY_PricePerCarat"] = result["price"] / result["carat"].replace(0, np.nan)

    # Use the same five clarity families as the classification model and chat UI.
    clarity = result["clarity"].astype(str).str.upper()
    result["BUY_ClarityFamily"] = np.select(
        [
            clarity.eq("IF"), clarity.str.startswith("VVS"),
            clarity.str.startswith("VS"), clarity.str.startswith("SI"),
            clarity.str.startswith("I"),
        ],
        ["IF", "VVS", "VS", "SI", "I"],
        default="Unknown",
    )

    # K-Means is distance based. Log transforms stop a small number of very
    # expensive or large diamonds from dominating the centroids.
    result["BUY_LogPrice"] = np.log1p(result["price"])
    result["BUY_LogCarat"] = np.log1p(result["carat"])
    result["BUY_LogFaceArea"] = np.log1p(result["BUY_FaceArea"])
    result["BUY_LogVolume"] = np.log1p(result["BUY_Volume"])
    result["BUY_LogPricePerCarat"] = np.log1p(result["BUY_PricePerCarat"])

    # Replace infinite values from division by zero with missing values.
    return result.replace([np.inf, -np.inf], np.nan)
