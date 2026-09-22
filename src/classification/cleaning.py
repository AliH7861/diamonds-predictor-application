"""
Data cleaning and target creation for diamond clarity classification.

This module loads the raw diamond dataset, validates required columns, removes
invalid physical measurements, and converts the original clarity grades into
five broader clarity families. Price is not required by this workflow.

Final clarity families:
I → SI → VS → VVS → IF
"""

import pandas as pd


# Final clarity families used by the classification models.
CLARITY_CLASSES = ["I", "SI", "VS", "VVS", "IF"]

# Map the original clarity grades into five integer family classes.
CLARITY_MAP = {
    "I1": 0, "SI2": 1, "SI1": 1, "VS2": 2,
    "VS1": 2, "VVS2": 3, "VVS1": 3, "IF": 4
}


def load_and_clean_data(data_path):
    """Load the diamond dataset and apply the shared cleaning rules."""

    # Load the raw CSV.
    df = pd.read_csv(data_path)

    # Make sure every required modeling column exists.
    required = {"carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Remove the extra CSV index column when present.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns="Unnamed: 0")

    # Match the Y23 notebook: remove only missing or physically impossible
    # measurements. Unusual valid geometry remains available to the model.
    numeric = ["carat", "depth", "table", "x", "y", "z"]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=list(required)).copy()
    df = df.loc[(df[numeric] > 0).all(axis=1)].reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df


def create_target(df):
    """Convert original clarity grades into the five-family classification target."""

    # Work on a copy so the original DataFrame is unchanged.
    df = df.copy()

    if "clarity" not in df.columns:
        raise ValueError("Missing 'clarity' column.")

    # Check that every clarity grade has a defined family mapping.
    unknown_clarity = set(df["clarity"].dropna().unique()) - set(CLARITY_MAP)

    if unknown_clarity:
        raise ValueError(f"Unknown clarity grades: {sorted(unknown_clarity)}")

    # Map the original grades into integer class IDs.
    df["Clarity_Target"] = df["clarity"].map(CLARITY_MAP)

    if df["Clarity_Target"].isna().any():
        raise ValueError("Some clarity grades could not be mapped.")

    # Store both the integer target and its readable family name.
    df["Clarity_Target"] = df["Clarity_Target"].astype(int)
    df["Clarity_Family"] = df["Clarity_Target"].map(dict(enumerate(CLARITY_CLASSES)))

    return df
