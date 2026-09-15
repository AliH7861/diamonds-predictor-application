import numpy as np
from src.regression.preprocessing import prepare_regression_data
from src.regression.training import train_ann_smoke
from tests.helpers import make_diamonds, require_tensorflow

tf = require_tensorflow(module_level=True)


def test_regression_pipeline_trains_and_persists(tmp_path):
    path = tmp_path / "diamonds.csv"
    make_diamonds().to_csv(path, index=False)
    prepared = prepare_regression_data(path)
    data = prepared["experiments"]["CURRENT_BASELINE"]["matrices"]
    targets = prepared["targets"]
    model, predictions, metrics = train_ann_smoke(
        data["train"], targets["train"], data["valid"], targets["valid"]
    )
    model_path = tmp_path / "price_ann.keras"
    before_reload = model.predict(data["valid"][:2], verbose=0)
    model.save(model_path)
    reloaded = tf.keras.models.load_model(model_path)
    after_reload = reloaded.predict(data["valid"][:2], verbose=0)
    assert predictions.shape == targets["valid"].shape
    np.testing.assert_allclose(after_reload, before_reload, rtol=1e-5, atol=1e-6)
    assert {"MAE", "MSE", "RMSE", "R2", "MAPE"}.issubset(metrics)
