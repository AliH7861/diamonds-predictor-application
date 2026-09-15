"""
Save, reload, and use the selected diamond price-regression model.

This module stores the evaluated regression model together with its preprocessing
pipeline, learned human-feature context, validation metrics, metadata, and provenance.

It also reloads the saved model later and supports prediction directly from raw
diamond inputs while preventing price or engineered features from being supplied manually.

Both ANN and tree-based regression models are supported.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from .config import PROJECT_ROOT
from .cleaning import RAW_FEATURES
from .feature_engineering import (
    engineer_base_features, transform_human_features, FEATURE_SETS,
    CUT_SCORE, COLOR_SCORE, CLARITY_SCORE
)
from .training import predict_run

# Store the final regression model inside the project's models directory.
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "regression"

def save_best_model(run, output_dir=DEFAULT_MODEL_DIR, provenance=None):
    """Save the selected regression model, pipeline, metrics, and metadata."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ANN weights are stored separately while the rest of the run is saved with Joblib.
    saved_run = dict(run)

    if run["algorithm"] == "ANN":
        run["model"].save(output_dir / "model.keras")
        saved_run["model"] = None

    joblib.dump(saved_run, output_dir / "pipeline.joblib")

    # Convert NumPy scalar metrics into standard Python values for JSON.
    validation = {key: value.item() if isinstance(value, np.generic) else value for key, value in run["metrics"].items()}

    # Store the main information needed to identify and reproduce the saved model.
    metadata = {
        "algorithm": run["algorithm"], "feature_set": run["feature_set"],
        "target": "price", "target_transform": "log1p / expm1",
        "required_inputs": RAW_FEATURES,
        "model_file": "model.keras" if run["algorithm"] == "ANN" else "pipeline.joblib",
        "smoke_test_only": run.get("smoke", False),
        "validation": validation, "provenance": provenance or {}
    }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return output_dir

def load_best_model(model_dir=DEFAULT_MODEL_DIR):
    """Load the saved regression model and its preprocessing bundle."""

    model_dir = Path(model_dir)
    pipeline_path = model_dir / "pipeline.joblib"

    # Make sure a saved regression bundle exists.
    if not pipeline_path.is_file():
        raise FileNotFoundError("Run RegressionModel.ipynb through model saving first.")

    run = joblib.load(pipeline_path)

    # ANN architecture is stored separately from the serialized run bundle.
    if run["algorithm"] == "ANN":
        from tensorflow.keras.models import load_model
        run["model"] = load_model(model_dir / "model.keras")

    return run

def predict_prices(run, records):
    """Validate raw diamond inputs and return predicted prices."""

    # Accept either one diamond dictionary or a list of diamonds.
    if isinstance(records, dict):
        records = [records]

    # Require 1 to 1000 dictionary records.
    valid_records = isinstance(records, list) and 1 <= len(records) <= 1000 and all(isinstance(row, dict) for row in records)

    if not valid_records:
        raise ValueError("Supply one diamond object or a list of 1 to 1000 objects.")

    # Only the original raw features are allowed; price and engineered features must not be supplied.
    for row in records:
        if set(row) != set(RAW_FEATURES):
            raise ValueError(f"Required inputs: {RAW_FEATURES}. Do not supply price or engineered features.")

    df = pd.DataFrame(records)

    # Validate physical measurements as positive finite numbers.
    numeric_features = ["carat", "depth", "table", "x", "y", "z"]

    for feature in numeric_features:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

        if df[feature].isna().any() or not np.isfinite(df[feature]).all() or (df[feature] <= 0).any():
            raise ValueError(f"{feature} must be positive and finite.")

    # Restrict categorical inputs to grades known by the training pipeline.
    category_options = {"cut": CUT_SCORE, "color": COLOR_SCORE, "clarity": CLARITY_SCORE}

    for feature, choices in category_options.items():
        if not df[feature].isin(choices).all():
            raise ValueError(f"Invalid {feature}; expected one of {list(choices)}.")

    # Recreate the baseline engineered features from the raw inputs.
    features = engineer_base_features(df)
    feature_config = FEATURE_SETS[run["feature_set"]]

    # Human-style experiments also require the training-derived context.
    if feature_config["source"] == "human":
        features = transform_human_features(features, run["context"])

    # Select the exact feature set used by the saved model.
    model_features = feature_config["numeric"] + feature_config["categorical"]
    values = predict_run(run, features[model_features])

    # Reject invalid outputs before returning predictions.
    if not np.isfinite(values).all():
        raise ValueError("These inputs produced nonfinite predictions.")

    return [{"predicted_price": float(value)} for value in values]