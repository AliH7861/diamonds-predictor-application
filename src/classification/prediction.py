from pathlib import Path
from datetime import datetime, timezone
import json
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from .cleaning import CLARITY_CLASSES
from .feature_engineering import engineer_features

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "classification"

# Save the Selected Model Together With Its Fitted Preprocessor
def save_best_model(model, model_type, preprocessor, experiment, model_name, validation_metrics, output_dir=DEFAULT_MODEL_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = "best_model.keras" if model_type == "ANN" else "best_model.joblib"
    if model_type == "ANN":
        model.save(output_dir / model_file)
    else:
        joblib.dump(model, output_dir / model_file)
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")

    versions = {}
    for package in ["numpy", "pandas", "scikit-learn", "tensorflow", "xgboost", "joblib"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = {
        "model_name": model_name, "model_type": model_type, "model_file": model_file,
        "experiment": int(experiment), "clarity_classes": CLARITY_CLASSES,
        "features": list(preprocessor.feature_names_in_),
        "required_inputs": ["carat", "depth", "table", "x", "y", "z", "cut", "color"] + (["price"] if experiment > 1 else []),
        "validation_metrics": {key: float(value) for key, value in validation_metrics.items() if isinstance(value, (int, float, np.number))},
        "saved_at": datetime.now(timezone.utc).isoformat(), "package_versions": versions
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


# Load a Model Bundle Previously Saved by This Notebook
def load_best_model(model_dir=DEFAULT_MODEL_DIR):
    model_dir = Path(model_dir)
    if not (model_dir / "metadata.json").is_file():
        raise FileNotFoundError("No saved classification model. Run the notebook through 'Save and Use the Best Model' first.")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["model_type"] == "ANN":
        from tensorflow.keras.models import load_model
        model = load_model(model_dir / metadata["model_file"])
    else:
        model = joblib.load(model_dir / metadata["model_file"])
    return {"model": model, "preprocessor": joblib.load(model_dir / "preprocessor.joblib"), "metadata": metadata}


# Predict From Raw Diamond Details Without Requiring the Unknown Clarity
def predict_diamonds(bundle, records):
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list) or not records or len(records) > 1000 or not all(isinstance(row, dict) for row in records):
        raise ValueError("Provide one diamond object or a list of 1 to 1000 diamond objects.")

    required = bundle["metadata"]["required_inputs"]
    allowed = set(required) | {"price"}
    for number, row in enumerate(records):
        missing = set(required) - set(row)
        extra = set(row) - allowed
        if missing or extra:
            raise ValueError(f"Diamond {number}: missing inputs {sorted(missing)}; unexpected inputs {sorted(extra)}.")

    df = pd.DataFrame(records)
    for feature in ["carat", "depth", "table", "x", "y", "z"] + (["price"] if "price" in df else []):
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
        if df[feature].isna().any() or not np.isfinite(df[feature]).all() or (df[feature] <= 0).any():
            raise ValueError(f"{feature} must contain positive, finite numbers.")
    for feature, categories in [("cut", {"Fair", "Good", "Very Good", "Premium", "Ideal"}), ("color", set("DEFGHIJ"))]:
        if not df[feature].isin(categories).all():
            raise ValueError(f"{feature} must be one of {sorted(categories)}.")

    # Experiment 1 Does Not Use Price, but the Shared Feature Function Computes It
    if "price" not in df:
        df["price"] = np.nan
    features = engineer_features(df)
    X = bundle["preprocessor"].transform(features[bundle["metadata"]["features"]]).astype(np.float32)
    model = bundle["model"]
    if bundle["metadata"]["model_type"] == "ANN":
        probabilities = np.asarray(model.predict(X, verbose=0))
        classes = np.arange(len(CLARITY_CLASSES))
    else:
        probabilities = np.asarray(model.predict_proba(X))
        classes = np.asarray(model.classes_, dtype=int)
    predictions = classes[probabilities.argmax(axis=1)]

    return [{"clarity_target": int(target), "clarity_family": CLARITY_CLASSES[target],
             "probabilities": {CLARITY_CLASSES[int(label)]: float(probability) for label, probability in zip(classes, row)}}
            for target, row in zip(predictions, probabilities)]
