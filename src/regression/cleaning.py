import numpy as np
import pandas as pd
from .config import DATA_PATH


# Define the Raw Features for Regression Cleaning
RAW_FEATURES = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]

# Reproduce the Historical Regression Cleaning Rules
def load_and_clean_data(data_path=DATA_PATH):

    # Drop the Rows That Were not Needed
    raw = pd.read_csv(data_path).drop(columns="Unnamed: 0", errors="ignore")

    # Keep the Features That Were Needed
    required = RAW_FEATURES + ["price"]
    
    # Checking Missing Columns
    missing = set(required) - set(raw.columns)
    
    # Raise Error If There Are Missing Columns
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")


    # Remove Any Dimensions That Are Zero or Negative, and Create a Volume Proxy
    df = raw.loc[(raw[["x", "y", "z"]] > 0).all(axis=1)].copy()
    df["Volume_Proxy"] = df["x"] * df["y"] * df["z"]

    # Create Carat Bands for Volume-Based Outlier Detection
    bands = pd.cut(df["carat"], [0, .4, .6, .8, 1, 1.5, 2, 3, np.inf], right=False)

    # Group the Data by Carat Bands and Remove Outliers Based on the 10th and 90th Percentiles
    groups = df.groupby(bands, observed=True)["Volume_Proxy"]
    lower = groups.transform(lambda values: values.quantile(.10) * .70)
    upper = groups.transform(lambda values: values.quantile(.90) * 1.30)

    # Identify the Removed Observations Based on the Volume Proxy Outlier Rule
    removed = df.loc[~df["Volume_Proxy"].between(lower, upper)].copy()

    # Remove the Outliers and Recalculate the Carat-to-Volume Ratio
    df = df.loc[df["Volume_Proxy"].between(lower, upper)].copy()
    df["Carat_Per_Volume"] = df["carat"] / df["Volume_Proxy"]

    # Raise an Error If the Regression Data is Empty or Contains Missing Values
    if df.empty or df[required].isna().any().any():
        raise ValueError("Regression data must be nonempty and contain complete raw inputs.")

    # Define Numeric Columns and Check for Finite and Positive Values
    numeric = ["carat", "depth", "table", "x", "y", "z", "price"]

    # Raise an Error If Any Numeric Regression Inputs or Prices Are Not Positive and Finite
    if not np.isfinite(df[numeric]).all().all() or (df[numeric] <= 0).any().any():
        raise ValueError("Numeric regression inputs and prices must be positive and finite.")

    # Preserve duplicates and pre-split cleaning to reproduce the original regression population.
    return raw, df, removed
