from scripts.phase1_end_to_end import run_validation
from tests.helpers import require_tensorflow


def test_phase1_end_to_end():
    require_tensorflow()
    assert run_validation() is True
