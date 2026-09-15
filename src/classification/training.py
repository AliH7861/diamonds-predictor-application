"""
Train and evaluate the maintained diamond clarity-classification algorithms.

This module trains ANN, Random Forest, and XGBoost models on one prepared
experiment, evaluates them on the validation split, records runtime, and
supports one final evaluation on the untouched test set.

Smoke mode uses smaller models and shorter training so integration and
end-to-end tests can verify the workflow quickly.
"""

import time
import numpy as np
from config import RANDOM_STATE
from .evaluation import evaluate_model
from .model import build_ann


def train_ann_smoke(X_train, y_train, X_valid, y_valid, epochs: int = 1):
    """Briefly train the production ANN architecture for pipeline validation."""
    model = build_ann(X_train.shape[1])
    model.fit(
        X_train, y_train, validation_data=(X_valid, y_valid),
        epochs=epochs, batch_size=32, verbose=0,
    )
    predictions = np.argmax(model.predict(X_valid, verbose=0), axis=1)
    from .evaluation import classification_metrics

    return model, predictions, classification_metrics(y_valid, predictions)

def train_candidate(prepared, experiment: int, algorithm: str, smoke: bool = False) -> dict:
    """Train one classifier and evaluate it on the validation split."""

    from sklearn.utils.class_weight import compute_sample_weight

    data = prepared["experiments"][experiment]
    X_train, X_valid = data["X_train"], data["X_valid"]
    y_train, y_valid = prepared["y_train"], prepared["y_valid"]

    start = time.perf_counter()

    # Train the baseline ANN with early stopping and learning-rate reduction.
    if algorithm == "ANN":
        import tensorflow as tf

        model = build_ann(X_train.shape[1])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1 if smoke else 20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1 if smoke else 8, min_lr=1e-6)
        ]

        model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            epochs=1 if smoke else 160,
            batch_size=128,
            callbacks=callbacks,
            verbose=0 if smoke else 2
        )

        predictions = np.argmax(model.predict(X_valid, verbose=0), axis=1)
        model_type = "ANN"

    # Random Forest uses smaller settings in smoke mode for faster testing.
    elif algorithm == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=8 if smoke else 300,
            max_depth=4 if smoke else None,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=2 if smoke else -1
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_valid)
        model_type = "Tree"

    # XGBoost uses balanced sample weights to reduce the effect of class imbalance.
    elif algorithm == "XGBoost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            objective="multi:softprob", num_class=5,
            n_estimators=8 if smoke else 1800,
            learning_rate=0.05,
            max_depth=3 if smoke else 6,
            min_child_weight=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="mlogloss",
            early_stopping_rounds=3 if smoke else 100,
            random_state=RANDOM_STATE,
            n_jobs=2 if smoke else -1
        )

        weights = compute_sample_weight(class_weight="balanced", y=y_train)

        model.fit(
            X_train,
            y_train,
            sample_weight=weights,
            eval_set=[(X_valid, y_valid)],
            verbose=False if smoke else 100,
        )

        predictions = model.predict(X_valid)
        model_type = "Tree"

    else:
        raise ValueError(f"Unknown classification algorithm: {algorithm}")

    # Evaluate the validation predictions using the shared classification metrics.
    metrics = evaluate_model(y_valid, predictions, f"Experiment {experiment}", algorithm)
    metrics["Runtime_Min"] = (time.perf_counter() - start) / 60

    # Keep the trained model and its experiment information together for later use.
    return {
        "model": model, "model_type": model_type, "algorithm": algorithm,
        "experiment": experiment, "preprocessor": data["preprocessor"], "metrics": metrics
    }

def evaluate_on_test(run: dict, prepared: dict) -> dict:
    """Evaluate a validation-selected model once on the held-out test set."""

    matrix = prepared["experiments"][run["experiment"]]["X_test"]

    # ANN outputs probabilities while tree models return class labels directly.
    if run["model_type"] == "ANN":
        predictions = np.argmax(run["model"].predict(matrix, verbose=0), axis=1)
    else:
        predictions = run["model"].predict(matrix)

    return evaluate_model(prepared["y_test"], predictions, f"Experiment {run['experiment']}", run["algorithm"])
