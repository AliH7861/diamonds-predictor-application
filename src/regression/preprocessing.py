"""
Prepare the shared datasets and preprocessing pipelines used by the diamond price-regression experiments.

This module loads and cleans the raw diamond data, creates baseline engineered
features, applies one shared 70/15/15 train-validation-test split, learns
training-only human context, and prepares the exact feature matrices used by
each maintained regression experiment.

All preprocessors and human-context features are fitted using training rows only
so validation and test data remain isolated from model preparation.
"""

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from .config import RANDOM_STATE
from .cleaning import load_and_clean_data
from .feature_engineering import FEATURE_SETS, engineer_base_features, fit_human_context, transform_human_features

def prepare_regression_data(data_path=None, random_state=RANDOM_STATE):
    """Run the full regression data-preparation workflow."""

    # Load the default dataset unless another path is provided.
    if data_path is None:
        raw, cleaned, removed = load_and_clean_data()
    else:
        raw, cleaned, removed = load_and_clean_data(data_path)

    # Create baseline engineered features before applying the shared split.
    featured = engineer_base_features(cleaned)

    # Prepare every experiment using the same train, validation, and test rows.
    raw_splits = split_data(featured, random_state)
    prepared = prepare_experiments(raw_splits)

    # Keep the original cleaning stages available for diagnostics and export.
    prepared.update({"original": raw, "cleaned": cleaned, "featured": featured, "removed": removed})

    return prepared

def split_data(df, random_state=RANDOM_STATE):
    """Split regression data into shared 70/15/15 train, validation, and test sets."""

    # First split keeps 70% for training and 30% for validation/test.
    indices = np.arange(len(df))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=random_state)

    # Divide the remaining 30% evenly into validation and test sets.
    valid_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=random_state)

    # Preserve the original row indices inside each split.
    split_indices = {"train": train_idx, "valid": valid_idx, "test": test_idx}

    return {name: df.iloc[index].copy() for name, index in split_indices.items()}

def prepare_experiments(raw_splits):
    """Fit training-only context and prepare the feature matrices for every regression experiment."""

    # Learn human-style reference information from training rows only.
    context = fit_human_context(raw_splits["train"])

    # Apply the saved training context consistently to every split.
    human = {split: transform_human_features(df, context) for split, df in raw_splits.items()}

    experiments = {}

    # Build the exact feature frame and encoded matrix required by each experiment.
    for name, feature_config in FEATURE_SETS.items():
        source = raw_splits if feature_config["source"] == "raw" else human
        features = feature_config["numeric"] + feature_config["categorical"]

        # Numeric features pass through unchanged while categories are one-hot encoded.
        preprocessor = ColumnTransformer([
            ("numeric", "passthrough", feature_config["numeric"]),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), feature_config["categorical"])
        ])

        # Keep original-unit feature frames for interpretation and export.
        frames = {split: df[features].copy() for split, df in source.items()}

        # Fit preprocessing only on the training split.
        matrices = {"train": preprocessor.fit_transform(frames["train"]).astype(np.float32)}

        # Validation and test data use the already fitted training preprocessor.
        for split in ["valid", "test"]:
            matrices[split] = preprocessor.transform(frames[split]).astype(np.float32)

        # Catch invalid engineered or encoded values before training.
        if not all(np.isfinite(matrix).all() for matrix in matrices.values()):
            raise ValueError(f"Nonfinite inputs in {name}.")

        experiments[name] = {"frames": frames, "matrices": matrices, "preprocessor": preprocessor}

    # Price stays separate from the input features and remains the regression target.
    targets = {split: df["price"].to_numpy() for split, df in raw_splits.items()}

    return {"raw": raw_splits, "human": human, "context": context, "experiments": experiments, "targets": targets}