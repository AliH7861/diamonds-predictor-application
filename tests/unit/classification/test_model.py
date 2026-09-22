from src.classification.model import build_ann, build_ordinal_ann
from tests.helpers import require_tensorflow


def test_classification_ann_output_shape():
    require_tensorflow()
    model = build_ann(12)
    assert model.input_shape == (None, 12)
    assert model.output_shape == (None, 5)


def test_ordinal_classification_ann_output_shape():
    require_tensorflow()
    model = build_ordinal_ann(12)
    assert model.input_shape == (None, 12)
    assert model.output_shape == (None, 4)
