"""Train comparable ordinal clarity models on one physical-only representation."""

import time
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

from config import RANDOM_STATE
from .evaluation import evaluate_model
from .model import build_ordinal_ann
from .ordinal import make_ordinal_targets, predict_ordinal_ann, predict_ordinal_models

RF_PARAMETERS = {"bootstrap": False, "n_estimators": 800, "criterion": "entropy", "max_depth": 24, "min_samples_split": 5, "min_samples_leaf": 1, "max_features": "log2"}
XGB_PARAMETERS = {"learning_rate": 0.041606498696893375, "max_depth": 10, "min_child_weight": 1.4115316963725624, "subsample": 0.8456814329116074, "colsample_bytree": 0.6727723142480915, "gamma": 0.6413066044459287, "reg_alpha": 0.008871468610038968, "reg_lambda": 0.2684941807745774}


def train_ann_smoke(X_train, y_train, X_valid, y_valid, epochs: int = 1):
    """Briefly train and decode the maintained ordinal ANN."""
    model = build_ordinal_ann(X_train.shape[1])
    model.fit(X_train, make_ordinal_targets(y_train), validation_data=(X_valid, make_ordinal_targets(y_valid)), epochs=epochs, batch_size=32, verbose=0)
    predictions, _ = predict_ordinal_ann(model, X_valid)
    return model, predictions, evaluate_model(y_valid, predictions, "Y23 Physical Only", "ANN")


def _train_tree_boundaries(X_train, y_train, X_valid, y_valid, algorithm, smoke):
    """Fit one balanced binary classifier for each ordered class boundary."""
    models = []
    for threshold in range(4):
        binary_train = (np.asarray(y_train) > threshold).astype(int)
        binary_valid = (np.asarray(y_valid) > threshold).astype(int)
        weights = compute_sample_weight(class_weight="balanced", y=binary_train)
        if algorithm == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            parameters = dict(RF_PARAMETERS)
            if smoke:
                parameters.update(n_estimators=8, max_depth=4)
            model = RandomForestClassifier(**parameters, random_state=RANDOM_STATE + threshold, n_jobs=-1)
            model.fit(X_train, binary_train, sample_weight=weights)
        else:
            from xgboost import XGBClassifier
            parameters = dict(XGB_PARAMETERS)
            parameters.update(objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_estimators=8 if smoke else 3000, early_stopping_rounds=3 if smoke else 120, random_state=RANDOM_STATE + threshold, n_jobs=-1)
            if smoke:
                parameters.update(max_depth=3, learning_rate=0.1)
            model = XGBClassifier(**parameters)
            model.fit(X_train, binary_train, sample_weight=weights, eval_set=[(X_valid, binary_valid)], verbose=False)
        models.append(model)
    return models


def train_candidate(prepared, experiment: int, algorithm: str, smoke: bool = False) -> dict:
    """Train one algorithm using the exact same split and selected features."""
    if experiment != 1:
        raise ValueError("Classification now has one shared physical-only experiment.")
    data = prepared["experiments"][1]
    X_train, X_valid = data["X_train"], data["X_valid"]
    y_train, y_valid = prepared["y_train"], prepared["y_valid"]
    started = time.perf_counter()
    if algorithm == "ANN":
        import tensorflow as tf
        model = build_ordinal_ann(X_train.shape[1])
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1 if smoke else 20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1 if smoke else 8, min_lr=1e-6),
        ]
        history = model.fit(X_train, make_ordinal_targets(y_train), validation_data=(X_valid, make_ordinal_targets(y_valid)), epochs=1 if smoke else 160, batch_size=128, callbacks=callbacks, verbose=0 if smoke else 2)
        predictions, probabilities = predict_ordinal_ann(model, X_valid)
        model_type, trained_epochs = "OrdinalANN", len(history.history["loss"])
    elif algorithm in {"Random Forest", "XGBoost"}:
        model = _train_tree_boundaries(X_train, y_train, X_valid, y_valid, algorithm, smoke)
        predictions, probabilities = predict_ordinal_models(model, X_valid)
        model_type, trained_epochs = "OrdinalTree", None
    else:
        raise ValueError(f"Unknown classification algorithm: {algorithm}")
    metrics = evaluate_model(y_valid, predictions, "Y23 Physical Only", algorithm)
    metrics["Runtime_Min"] = (time.perf_counter() - started) / 60
    return {"model": model, "model_type": model_type, "algorithm": algorithm, "experiment": 1, "preprocessor": data["preprocessor"], "metrics": metrics, "validation_probabilities": probabilities, "trained_epochs": trained_epochs}


def evaluate_on_test(run: dict, prepared: dict) -> dict:
    """Evaluate a validation-selected model once on the untouched test split."""
    matrix = prepared["experiments"][1]["X_test"]
    if run["model_type"] == "OrdinalANN":
        predictions, _ = predict_ordinal_ann(run["model"], matrix)
    elif run["model_type"] == "OrdinalTree":
        predictions, _ = predict_ordinal_models(run["model"], matrix)
    elif run["model_type"] == "ANN":
        predictions = np.argmax(run["model"].predict(matrix, verbose=0), axis=1)
    else:
        predictions = run["model"].predict(matrix)
    return evaluate_model(prepared["y_test"], predictions, "Y23 Physical Only", run["algorithm"])
