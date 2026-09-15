import numpy as np

from src.clustering.profiling import build_cluster_profiles
from tests.helpers import make_diamonds


def test_cluster_profiles_include_statistics_distributions_and_interpretations():
    frame = make_diamonds(rows=40)
    frame["BUY_FaceArea"] = frame["x"] * frame["y"]
    frame["BUY_Volume"] = frame["x"] * frame["y"] * frame["z"]
    frame["BUY_AspectRatio"] = frame["x"] / frame["y"]
    frame["BUY_PricePerCarat"] = frame["price"] / frame["carat"]
    profiles = build_cluster_profiles(frame, np.arange(len(frame)) % 3, 3)
    assert len(profiles["summary"]) == 3
    assert {"Mean", "Median", "Min", "Max", "Range", "Std"}.issubset(profiles["numeric"])
    assert {"Count", "Percentage", "Is_Most_Common"}.issubset(profiles["categorical"])
    assert profiles["summary"]["Buyer_Interpretation"].str.endswith("buyer").all()
