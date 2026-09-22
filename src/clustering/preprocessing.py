"""Compatibility helpers for preparing an existing or file-backed DataFrame."""

from pathlib import Path

import pandas as pd

from src.datasets import load_diamond_frame

from .config import SegmentationConfig
from .feature_engineering import (
    CUSTOMER_PILLARS,
    build_customer_pillars,
    build_segmentation_features,
    split_geometry_anomalies,
)


def prepare_segmentation_frame(
    frame: pd.DataFrame, config: SegmentationConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return segmentable rows and excluded geometry anomalies."""
    settings = config or SegmentationConfig()
    engineered = build_segmentation_features(frame, settings)
    segmentable, anomalies = split_geometry_anomalies(engineered, settings)
    return build_customer_pillars(segmentable), anomalies


def prepare_clustering_data(
    data_path: str | Path, sample_rows: int | None = None
) -> dict[str, pd.DataFrame]:
    """Retain the historical file entry point while reusing shared cleaning."""
    frame = load_diamond_frame(data_path)
    if sample_rows and len(frame) > sample_rows:
        frame = frame.sample(sample_rows, random_state=42).reset_index(drop=True)
    profiles, anomalies = prepare_segmentation_frame(frame)
    return {
        "profiles": profiles,
        "pillars": profiles[list(CUSTOMER_PILLARS)],
        "anomalies": anomalies,
    }
