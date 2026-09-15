"""
Load the saved buyer-segmentation artifact and assign new diamond purchases to clusters.

This module restores the selected K-Means model, fitted preprocessor, and saved
segment profiles, then applies the same purchase feature engineering and
preprocessing steps to new raw diamond purchases before assigning the nearest cluster.

The returned result includes both the numeric cluster ID and its saved buyer interpretation.
"""
from pathlib import Path
import joblib
import pandas as pd
from .feature_engineering import MODEL_FEATURES, engineer_purchase_features

def load_segmentation_model(path: str | Path) -> dict:
    """Load the saved K-Means model, preprocessor, and segment profiles."""

    return joblib.load(path)

def assign_purchase_segment(artifact: dict, purchase: dict | pd.DataFrame) -> pd.DataFrame:
    """Assign one or more raw diamond purchases to their nearest saved cluster."""

    # Accept either one purchase dictionary or an existing DataFrame.
    frame = pd.DataFrame([purchase]) if isinstance(purchase, dict) else purchase.copy()

    # Recreate the same purchase-profile features used during clustering.
    featured = engineer_purchase_features(frame)

    # Apply the saved preprocessor and match the numeric type used by the K-Means centroids.
    matrix = artifact["preprocessor"].transform(featured[MODEL_FEATURES]).astype(artifact["model"].cluster_centers_.dtype, copy=False)

    # Assign each purchase to its nearest cluster.
    labels = artifact["model"].predict(matrix)

    result = frame.copy()
    result["Cluster"] = labels

    # Map cluster IDs back to the saved human-readable buyer interpretations.
    profile_lookup = artifact["segment_profiles"].set_index("Cluster")["Buyer_Interpretation"]
    result["Buyer_Interpretation"] = result["Cluster"].map(profile_lookup)

    return result
