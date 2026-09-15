from src.regression.model import build_ann
from tests.helpers import require_tensorflow


def test_regression_ann_output_shape():
    require_tensorflow()
    model = build_ann(12)
    assert model.input_shape == (None, 12)
    assert model.output_shape == (None, 1)
