"""Load and prepare shared experiment inputs without training any models."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import RANDOM_STATE, RAW_DATA_PATH
from .cleaning import CLARITY_CLASSES as CLARITY_CLASSES
from .cleaning import CLARITY_MAP as CLARITY_MAP
from .cleaning import create_target, load_and_clean_data
from .feature_engineering import (
    engineer_features, CATEGORICAL_FEATURES,
    EXPERIMENT_1_NUMERIC, EXPERIMENT_2_NUMERIC, EXPERIMENT_3_NUMERIC,
)

DEFAULT_DATA_PATH = RAW_DATA_PATH

# Scale numeric features and one-hot encode categorical features.
def create_preprocessor(numeric_features):
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),

        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), CATEGORICAL_FEATURES)
    ])

def prepare_classification_data(data_path=DEFAULT_DATA_PATH, random_state=RANDOM_STATE):
    """Prepare all three experiments using one stratified 70/15/15 split.

    Preserves the notebook's full-dataset geometry outlier filtering.
    Imputers, scalers, and encoders are fitted on training rows only.
    All model types share the same scaled arrays, as in the original code.
    """
    df = load_and_clean_data(data_path)
    df = create_target(df)
    cleaned_df = df.copy()
    df = engineer_features(df)
    y = df["Clarity_Target"].astype(int).to_numpy()
    indices = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=random_state, stratify=y
    )
    valid_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=random_state, stratify=y[temp_idx]
    )
    experiments = {}
    for number, numeric_features in enumerate(
        [EXPERIMENT_1_NUMERIC, EXPERIMENT_2_NUMERIC, EXPERIMENT_3_NUMERIC], start=1
    ):
        X = df[numeric_features + CATEGORICAL_FEATURES]
        preprocessor = create_preprocessor(numeric_features)
        experiments[number] = {
            "preprocessor": preprocessor,
            "X_train": preprocessor.fit_transform(X.iloc[train_idx]).astype(np.float32),
            "X_valid": preprocessor.transform(X.iloc[valid_idx]).astype(np.float32),
            "X_test": preprocessor.transform(X.iloc[test_idx]).astype(np.float32)
        }

    return {
        "cleaned": cleaned_df, "diamonds": df, "experiments": experiments,
        "y_train": y[train_idx], "y_valid": y[valid_idx], "y_test": y[test_idx],
        "train_indices": train_idx, "valid_indices": valid_idx, "test_indices": test_idx
    }
