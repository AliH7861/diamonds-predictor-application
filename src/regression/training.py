"""
Train, compare, select, and evaluate the maintained diamond price-regression models.

All algorithms predict log1p(price) during training and convert predictions back
to dollar values with expm1() during evaluation and inference.

The module supports ANN, XGBoost, Random Forest, Extra Trees, CatBoost, and
LightGBM across the maintained feature sets. Model selection uses validation MAE
only, while the held-out test set is evaluated once after the winner is selected.

Smoke mode uses smaller models and shorter training for workflow validation.
"""

import time
import numpy as np

from .config import RANDOM_STATE, XGB_PARAMS, RF_PARAMS
from .evaluation import full_regression_metrics
from .feature_engineering import FEATURE_SETS


def train_ann_smoke(X_train, y_train, X_valid, y_valid, epochs: int = 1):
    """Briefly train the price ANN for pipeline validation."""
    from .model import build_ann

    model = build_ann(X_train.shape[1])
    model.fit(
        X_train, np.log1p(y_train),
        validation_data=(X_valid, np.log1p(y_valid)),
        epochs=epochs, batch_size=32, verbose=0,
    )
    predictions = np.maximum(
        np.expm1(model.predict(X_valid, verbose=0).reshape(-1)), 0
    )
    return model, predictions, full_regression_metrics(y_valid, predictions)

def predict_run(run, frame):
    """Generate dollar-price predictions from one trained regression run."""

    # CatBoost uses original DataFrames so categorical columns remain intact.
    X = frame if run["algorithm"] == "CatBoost" else run["preprocessor"].transform(frame).astype(np.float32)

    # ANN prediction output needs flattening; tree models already return a 1D array.
    if run["algorithm"] == "ANN":
        values = run["model"].predict(X, verbose=0).reshape(-1)
    else:
        values = run["model"].predict(X)

    # Models predict log1p(price), so convert back to dollars and prevent negatives.
    return np.maximum(np.expm1(values), 0)

def train_model(prepared, feature_set, algorithm, smoke=False):
    """Train one regression candidate and evaluate it on the validation split."""

    data = prepared["experiments"][feature_set]

    # All algorithms use the same train/validation rows and log-transformed price target.
    X_train, X_valid = data["matrices"]["train"], data["matrices"]["valid"]
    y_train, y_valid = (np.log1p(prepared["targets"][split]) for split in ["train", "valid"])

    run_preprocessor = data["preprocessor"]
    start = time.perf_counter()

    # ANN uses scaled numeric inputs while categorical features remain one-hot encoded.
    if algorithm == "ANN":
        import tensorflow as tf
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        from .model import build_ann

        feature_config = FEATURE_SETS[feature_set]

        run_preprocessor = ColumnTransformer([
            ("numeric", StandardScaler(), feature_config["numeric"]),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), feature_config["categorical"])
        ])

        X_train = run_preprocessor.fit_transform(data["frames"]["train"]).astype(np.float32)
        X_valid = run_preprocessor.transform(data["frames"]["valid"]).astype(np.float32)

        model = build_ann(X_train.shape[1])

        # Early stopping keeps the best validation-loss weights.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=1 if smoke else 20, restore_best_weights=True
        )

        model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            epochs=1 if smoke else 200,
            batch_size=128,
            callbacks=[early_stopping],
            verbose=0 if smoke else 2
        )

    # XGBoost uses the recovered tuned R10 parameters.
    elif algorithm == "XGBoost":
        from xgboost import XGBRegressor

        params = dict(XGB_PARAMS)

        if smoke:
            params.update(n_estimators=8, max_depth=3, early_stopping_rounds=3, n_jobs=2)

        model = XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False if smoke else 100,
        )

    # Random Forest uses recovered R10 parameters; Extra Trees uses its own configuration.
    elif algorithm in ["RandomForest", "ExtraTrees"]:
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

        if algorithm == "RandomForest":
            params = dict(RF_PARAMS)
            model_class = RandomForestRegressor
        else:
            params = {
                "n_estimators": 700, "max_depth": None,
                "min_samples_split": 2, "min_samples_leaf": 1,
                "max_features": 0.8, "bootstrap": False,
                "random_state": RANDOM_STATE, "n_jobs": -1
            }
            model_class = ExtraTreesRegressor

        if smoke:
            params.update(n_estimators=8, max_depth=4, n_jobs=2)

        model = model_class(**params)
        model.fit(X_train, y_train)

    # CatBoost works directly with original categorical columns.
    elif algorithm == "CatBoost":
        from catboost import CatBoostRegressor

        feature_config = FEATURE_SETS[feature_set]
        X_train, X_valid = data["frames"]["train"], data["frames"]["valid"]

        model = CatBoostRegressor(
            iterations=8 if smoke else 5000, learning_rate=0.03,
            depth=3 if smoke else 8, loss_function="RMSE", eval_metric="RMSE",
            l2_leaf_reg=5.0, random_strength=0.5, bootstrap_type="Bayesian",
            bagging_temperature=0.5, random_seed=RANDOM_STATE,
            allow_writing_files=False, verbose=False
        )

        model.fit(
            X_train, y_train,
            cat_features=feature_config["categorical"],
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=3 if smoke else 150,
            verbose=False
        )

    # LightGBM uses validation RMSE for early stopping.
    elif algorithm == "LightGBM":
        from lightgbm import LGBMRegressor, early_stopping

        model = LGBMRegressor(
            objective="regression", n_estimators=8 if smoke else 6000,
            learning_rate=0.02, num_leaves=48, max_depth=-1,
            min_child_samples=20, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.05, reg_lambda=0.5, random_state=RANDOM_STATE,
            n_jobs=2 if smoke else -1, verbosity=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmse",
            callbacks=[early_stopping(3 if smoke else 150, verbose=False)]
        )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Keep the trained model and everything needed for later prediction together.
    run = {
        "model": model, "algorithm": algorithm, "feature_set": feature_set,
        "preprocessor": run_preprocessor, "context": prepared["context"], "smoke": smoke
    }

    # Evaluate in original dollar units on validation data.
    predictions = predict_run(run, data["frames"]["valid"])

    run["metrics"] = {
        "Algorithm": algorithm, "Feature_Set": feature_set,
        "Inputs": X_train.shape[1], "Runtime_Min": (time.perf_counter() - start) / 60,
        **full_regression_metrics(prepared["targets"]["valid"], predictions)
    }

    return run

def train_candidates(prepared, feature_sets, algorithms, smoke=False):
    """Train every requested feature-set and algorithm combination."""

    runs = []

    for feature_set in feature_sets:
        for algorithm in algorithms:
            print(f"Training {algorithm} with {feature_set}...")

            run = train_model(prepared, feature_set, algorithm, smoke=smoke)

            print(f"Validation MAE: ${run['metrics']['MAE']:,.2f}")
            runs.append(run)

    return runs

def select_best_run(runs):
    """Select the trained candidate with the lowest validation MAE."""

    if not runs:
        raise ValueError("At least one trained run is required.")

    return min(runs, key=lambda run: run["metrics"]["MAE"])


def evaluate_on_test(run, prepared):
    """Evaluate the validation-selected winner once on the held-out test split."""

    data = prepared["experiments"][run["feature_set"]]
    predictions = predict_run(run, data["frames"]["test"])

    metrics = {
        "Algorithm": run["algorithm"], "Feature_Set": run["feature_set"],
        "Inputs": data["matrices"]["test"].shape[1],
        **full_regression_metrics(prepared["targets"]["test"], predictions)
    }

    return predictions, metrics
