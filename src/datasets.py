"""Shared, non-model-specific loading for the diamonds dataset."""

from pathlib import Path

import pandas as pd

DIAMOND_COLUMNS = {
    "carat",
    "cut",
    "color",
    "clarity",
    "depth",
    "table",
    "price",
    "x",
    "y",
    "z",
}


def load_diamond_frame(path: str | Path) -> pd.DataFrame:
    """Load the project dataset without applying model-specific row rules."""
    frame = pd.read_csv(path)
    index_columns = [
        column for column in frame.columns if str(column).casefold().startswith("unnamed:")
    ]
    frame = frame.drop(columns=index_columns, errors="ignore")
    missing = DIAMOND_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Diamond dataset is missing columns: {sorted(missing)}")
    return frame
