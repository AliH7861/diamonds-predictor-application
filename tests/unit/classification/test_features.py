import numpy as np
from src.classification.feature_engineering import engineer_features
from tests.helpers import make_diamonds


def test_classification_features_are_finite():
    frame = engineer_features(make_diamonds(20).drop(columns="Unnamed: 0"))
    engineered = [
        column for column in frame if column.startswith("REG_") or column.startswith("PricePer")
    ]
    assert len(engineered) == 12
    assert np.isfinite(frame[engineered]).all().all()
