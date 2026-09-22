"""Load a saved segmentation artifact and assign diamonds to buyer profiles."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

from .config import SegmentationConfig
from .feature_engineering import (
    CUSTOMER_PILLARS,
    REQUIRED_COLUMNS,
    build_customer_pillars,
    build_segmentation_features,
)


def load_segmentation_model(path: str | Path) -> dict:
    """Load the selected cluster model, scaler, reference data, and profiles."""
    return joblib.load(path)


def _predict_labels(model, matrix: np.ndarray) -> np.ndarray:
    """Use the selected model or saved hierarchical centroids for assignment."""
    if hasattr(model, "predict"):
        return np.asarray(model.predict(matrix), dtype=int)
    centers = np.asarray(model["centers"])
    return euclidean_distances(matrix, centers).argmin(axis=1)


def assign_purchase_segment(artifact: dict, purchase: dict | pd.DataFrame) -> pd.DataFrame:
    """Assign raw diamond rows with the training data as percentile reference."""
    incoming = pd.DataFrame([purchase]) if isinstance(purchase, dict) else purchase.copy()
    missing = set(REQUIRED_COLUMNS) - set(incoming.columns)
    if missing:
        raise ValueError(f"Purchase is missing columns: {sorted(missing)}")

    reference = artifact.get("reference_frame")
    if reference is None:
        raise ValueError("Segmentation artifact does not contain its feature reference data.")
    combined = pd.concat(
        [reference[list(REQUIRED_COLUMNS)], incoming[list(REQUIRED_COLUMNS)]],
        ignore_index=True,
    )
    engineered = build_segmentation_features(combined, SegmentationConfig())
    featured = build_customer_pillars(engineered).tail(len(incoming))
    matrix = artifact["scaler"].transform(featured[list(CUSTOMER_PILLARS)])
    labels = _predict_labels(artifact["model"], matrix)
    names = {profile.cluster_id: profile.name for profile in artifact["profiles"]}

    result = incoming.reset_index(drop=True)
    result["cluster_id"] = labels
    result["customer_profile_name"] = result["cluster_id"].map(names)
    return result
