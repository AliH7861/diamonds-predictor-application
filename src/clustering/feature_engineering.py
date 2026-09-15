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

# Numeric features describing price, physical size, shape, and value relationships.
NUMERIC_FEATURES = [
    "price", "carat", "depth", "table", "x", "y", "z",
    "BUY_FaceArea", "BUY_Volume", "BUY_AspectRatio", "BUY_PricePerCarat"]

# Categorical quality characteristics used when building customer-style segments.
CATEGORICAL_FEATURES = ["cut", "color", "clarity"]

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

    # Replace infinite values from division by zero with missing values.
    return result.replace([np.inf, -np.inf], np.nan)