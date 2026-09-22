"""Prepare the shared physical-only Y23 clarity representation."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import RANDOM_STATE, RAW_DATA_PATH

from .cleaning import CLARITY_CLASSES, CLARITY_MAP, create_target, load_and_clean_data

# These names remain available here for notebook and test compatibility.
__all__ = ["CLARITY_CLASSES", "CLARITY_MAP", "create_target", "load_and_clean_data"]
from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES_ALL, engineer_features

DEFAULT_DATA_PATH = RAW_DATA_PATH
CORRELATION_THRESHOLD = 0.75


def select_numeric_features(
    training_frame: pd.DataFrame, target: np.ndarray, threshold: float = CORRELATION_THRESHOLD
) -> tuple[list[str], pd.DataFrame]:
    """Keep high-information numeric features while pruning correlated duplicates."""
    medians = training_frame[NUMERIC_FEATURES_ALL].median()
    numeric = training_frame[NUMERIC_FEATURES_ALL].fillna(medians)
    information = mutual_info_classif(numeric, target, random_state=RANDOM_STATE)
    ranking = pd.DataFrame(
        {"Feature": NUMERIC_FEATURES_ALL, "Mutual_Information": information}
    ).sort_values("Mutual_Information", ascending=False)
    correlations = numeric.corr(method="pearson")
    selected: list[str] = []
    for candidate in ranking["Feature"]:
        if all(
            pd.isna(correlations.loc[candidate, existing])
            or abs(correlations.loc[candidate, existing]) < threshold
            for existing in selected
        ):
            selected.append(candidate)
    return selected, ranking.reset_index(drop=True)


def create_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    """Scale selected physical features and encode cut and color."""
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def prepare_classification_data(
    data_path=DEFAULT_DATA_PATH, random_state: int = RANDOM_STATE
) -> dict:
    """Create one leakage-free 70/15/15 split shared by every classifier."""
    cleaned = create_target(load_and_clean_data(data_path))
    diamonds = engineer_features(cleaned)
    y = diamonds["Clarity_Target"].astype(int).to_numpy()
    indices = np.arange(len(diamonds))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=random_state, stratify=y
    )
    valid_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=random_state, stratify=y[temp_idx]
    )

    selected, ranking = select_numeric_features(diamonds.iloc[train_idx], y[train_idx])
    features = selected + CATEGORICAL_FEATURES
    preprocessor = create_preprocessor(selected)
    matrix_train = preprocessor.fit_transform(diamonds.iloc[train_idx][features]).astype(np.float32)
    matrix_valid = preprocessor.transform(diamonds.iloc[valid_idx][features]).astype(np.float32)
    matrix_test = preprocessor.transform(diamonds.iloc[test_idx][features]).astype(np.float32)
    if any("price" in str(name).casefold() for name in preprocessor.get_feature_names_out()):
        raise AssertionError("Price must never enter clarity preprocessing.")

    experiment = {
        "name": "Y23 Physical Only",
        "preprocessor": preprocessor,
        "numeric_features": selected,
        "X_train": matrix_train,
        "X_valid": matrix_valid,
        "X_test": matrix_test,
    }
    return {
        "cleaned": cleaned,
        "diamonds": diamonds,
        "experiments": {1: experiment},
        "experiment": experiment,
        "mutual_information": ranking,
        "y_train": y[train_idx],
        "y_valid": y[valid_idx],
        "y_test": y[test_idx],
        "train_indices": train_idx,
        "valid_indices": valid_idx,
        "test_indices": test_idx,
    }
