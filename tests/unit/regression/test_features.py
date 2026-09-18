import numpy as np
from src.regression.feature_engineering import (
    engineer_base_features,
    fit_human_context,
    transform_human_features,
)
from tests.helpers import make_diamonds


def test_human_features_use_saved_training_context():
    frame = engineer_base_features(make_diamonds(40).drop(columns="Unnamed: 0"))
    context = fit_human_context(frame.iloc[:30])
    transformed = transform_human_features(frame.iloc[30:], context)
    human = [
        column for column in transformed if column.startswith("H_") and column != "H_CaratBand"
    ]
    assert len(human) >= 35
    assert np.isfinite(transformed[human]).all().all()
