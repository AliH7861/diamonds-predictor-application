from src.regression.preprocessing import prepare_regression_data
from tests.helpers import make_diamonds


def test_price_is_target_and_clarity_is_input(tmp_path):
    path = tmp_path / "diamonds.csv"
    make_diamonds().to_csv(path, index=False)
    prepared = prepare_regression_data(path)
    for experiment in prepared["experiments"].values():
        assert "price" not in experiment["preprocessor"].feature_names_in_
        assert "clarity" in experiment["preprocessor"].feature_names_in_
