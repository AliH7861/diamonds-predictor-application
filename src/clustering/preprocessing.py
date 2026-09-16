"""
Load, clean, engineer, and preprocess diamond purchase profiles for K-Means clustering.

This module prepares anonymous purchase profiles, removes invalid rows, creates
shared purchase-oriented features, scales numeric inputs, one-hot encodes
categorical inputs, and produces the final matrix used by every K-Means candidate.

The same fitted preprocessing pipeline is reused across K = 3, 5, 7, and 10
so the cluster comparisons remain consistent.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from .feature_engineering import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, engineer_purchase_features

# Raw columns required before purchase-profile features can be created.
REQUIRED_COLUMNS = {"price", "carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"}

def load_purchase_profiles(data_path: str | Path) -> pd.DataFrame:
    """Load unique, physically valid diamond rows and engineer purchase-profile features."""

    frame = pd.read_csv(data_path)

    # Remove the exported CSV index column when present.
    if "Unnamed: 0" in frame.columns:
        frame = frame.drop(columns="Unnamed: 0")

    # Make sure all required raw features are available.
    missing = REQUIRED_COLUMNS - set(frame.columns)

    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    # Remove duplicate purchases before clustering.
    frame = frame.drop_duplicates().copy()

    # Keep only diamonds with positive price, weight, and physical dimensions.
    valid = (frame["price"] > 0) & (frame["carat"] > 0) & (frame["x"] > 0) & (frame["y"] > 0) & (frame["z"] > 0)
    frame = frame.loc[valid].reset_index(drop=True)

    if len(frame) < 10:
        raise ValueError("At least ten valid purchase profiles are required.")

    return engineer_purchase_features(frame)

def create_preprocessor() -> ColumnTransformer:
    """Create the numeric-scaling and categorical-encoding pipeline used by K-Means."""

    # Robust scaling limits the influence of extreme market prices and sizes.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler(quantile_range=(10, 90)))
    ])

    # Categorical features are imputed and converted into one-hot encoded columns.
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        # Seventeen one-hot columns would otherwise outweigh the eight numeric
        # value and geometry measures solely because there are more of them.
        transformer_weights={"numeric": 1.0, "categorical": 0.5},
    )

def prepare_clustering_data(data_path: str | Path, sample_rows: int | None = None) -> dict:
    """Prepare the shared purchase-profile matrix used by every K-Means candidate."""

    frame = load_purchase_profiles(data_path)

    # Optional sampling keeps smoke tests fast while remaining reproducible.
    if sample_rows and len(frame) > sample_rows:
        frame = frame.sample(sample_rows, random_state=42).reset_index(drop=True)

    preprocessor = create_preprocessor()

    # Fit one shared preprocessing pipeline and create the clustering matrix.
    matrix = preprocessor.fit_transform(frame[MODEL_FEATURES]).astype(np.float32)

    if not np.isfinite(matrix).all():
        raise ValueError("Clustering matrix contains nonfinite values.")

    return {"profiles": frame, "matrix": matrix, "preprocessor": preprocessor}
