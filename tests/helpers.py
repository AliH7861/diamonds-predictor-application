"""Small deterministic data used only by automated smoke tests."""

import os

import numpy as np
import pandas as pd


def require_tensorflow(*, module_level: bool = False):
    """Import TensorFlow or skip when the host blocks its native runtime."""
    import pytest

    try:
        import tensorflow as tf
    except (ImportError, OSError) as error:
        is_windows_policy_block = os.name == "nt" and "Application Control policy" in str(error)
        if not is_windows_policy_block:
            raise
        pytest.skip(
            f"TensorFlow native runtime is unavailable: {error}",
            allow_module_level=module_level,
        )
    return tf


def make_diamonds(rows: int = 150, seed: int = 42) -> pd.DataFrame:
    """Return representative raw diamonds with every supported category."""
    rng = np.random.default_rng(seed)
    clarity = np.resize(["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], rows)
    cut = np.resize(["Fair", "Good", "Very Good", "Premium", "Ideal"], rows)
    color = np.resize(list("DEFGHIJ"), rows)
    carat = rng.uniform(0.25, 2.5, rows)
    diameter = 3.5 + 3.0 * np.sqrt(carat) + rng.normal(0, 0.05, rows)
    x = diameter * rng.normal(1.0, 0.01, rows)
    y = diameter * rng.normal(1.0, 0.01, rows)
    z = diameter * rng.normal(0.62, 0.01, rows)
    quality = np.array([{"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}[v] for v in cut])
    grade = np.array([{"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4,
                       "VVS2": 5, "VVS1": 6, "IF": 7}[v] for v in clarity])
    price = 400 + 2800 * carat**1.45 + 110 * quality + 90 * grade + rng.normal(0, 40, rows)
    return pd.DataFrame({
        "Unnamed: 0": np.arange(rows), "carat": carat, "cut": cut, "color": color,
        "clarity": clarity, "depth": rng.normal(61.5, 1.0, rows),
        "table": rng.normal(57.0, 1.2, rows), "price": np.maximum(price, 100),
        "x": x, "y": y, "z": z,
    })
