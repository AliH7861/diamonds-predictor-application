from src.classification.preprocessing import CLARITY_CLASSES, create_target
from tests.helpers import make_diamonds


def test_target_contains_five_ordered_families():
    frame = create_target(make_diamonds(40))
    assert CLARITY_CLASSES == ["I", "SI", "VS", "VVS", "IF"]
    assert set(frame["Clarity_Target"]) == set(range(5))
