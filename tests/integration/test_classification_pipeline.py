import numpy as np

from src.classification.preprocessing import prepare_classification_data
from src.classification.training import train_ann_smoke
from tests.helpers import make_diamonds, require_tensorflow

tf = require_tensorflow(module_level=True)


def test_classification_pipeline_trains_and_persists(tmp_path):
    path = tmp_path / "diamonds.csv"
    make_diamonds().to_csv(path, index=False)
    prepared = prepare_classification_data(path)
    data = prepared["experiments"][1]
    model, predictions, metrics = train_ann_smoke(
        data["X_train"], prepared["y_train"], data["X_valid"], prepared["y_valid"]
    )
    model_path = tmp_path / "clarity_ann.keras"
    before_reload = model.predict(data["X_valid"][:2], verbose=0)
    model.save(model_path)
    reloaded = tf.keras.models.load_model(model_path)
    after_reload = reloaded.predict(data["X_valid"][:2], verbose=0)
    assert predictions.shape == prepared["y_valid"].shape
    np.testing.assert_allclose(after_reload, before_reload, rtol=1e-5, atol=1e-6)
    assert after_reload.shape == (2, 5)
    assert "Macro_F1" in metrics
