"""
Regression evaluation utilities for the diamond price-prediction models.

This module calculates overall regression performance, percentage-error severity,
prediction bias, percentile error thresholds, and accuracy-within-range metrics.

It also breaks performance down by diamond price bands so model quality can be
compared across inexpensive, mid-range, and high-value diamonds.
"""
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def full_regression_metrics(actual, predicted):
    """Calculate the full regression and error-severity metric set."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    # Prevent impossible negative price predictions.
    predicted = np.maximum(predicted, 0)

    # Core prediction errors used by several later metrics.
    residuals = actual - predicted
    absolute_error = np.abs(residuals)
    percentage_error = absolute_error / np.maximum(actual, 1) * 100

    # Standard regression metrics.
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    # Typical absolute and percentage prediction error.
    median_absolute_error = np.median(absolute_error)
    mape = np.mean(percentage_error)
    median_ape = np.median(percentage_error)

    # Alternative percentage metrics for scale-independent comparison.
    smape = np.mean(2 * absolute_error / np.maximum(np.abs(actual) + np.abs(predicted), 1e-8)) * 100
    wape = absolute_error.sum() / np.maximum(np.abs(actual).sum(), 1e-8) * 100
    nrmse_mean = rmse / np.mean(actual) * 100

    # Positive bias means the model tends to underpredict; negative means overpredict.
    bias = np.mean(residuals)
    median_bias = np.median(residuals)

    # Percentiles show how large percentage errors become for the harder predictions.
    p75_ape = np.percentile(percentage_error, 75)
    p90_ape = np.percentile(percentage_error, 90)
    p95_ape = np.percentile(percentage_error, 95)
    p99_ape = np.percentile(percentage_error, 99)

    # Percentage of predictions that stay within useful error thresholds.
    within_5 = np.mean(percentage_error <= 5) * 100
    within_10 = np.mean(percentage_error <= 10) * 100
    within_15 = np.mean(percentage_error <= 15) * 100
    within_20 = np.mean(percentage_error <= 20) * 100

    # Percentage of predictions with larger errors.
    over_20 = np.mean(percentage_error > 20) * 100
    over_30 = np.mean(percentage_error > 30) * 100
    over_50 = np.mean(percentage_error > 50) * 100

    return {
        "MAE": mae, "Median_AE": median_absolute_error, "MSE": mse, "RMSE": rmse, "R2": r2,
        "MAPE": mape, "Median_APE": median_ape, "sMAPE": smape, "WAPE": wape,
        "Bias": bias, "Median_Bias": median_bias, "NRMSE_Mean": nrmse_mean,
        "P75_APE": p75_ape, "P90_APE": p90_ape, "P95_APE": p95_ape, "P99_APE": p99_ape,
        "Within_5": within_5, "Within_10": within_10, "Within_15": within_15, "Within_20": within_20,
        "Over_20": over_20, "Over_30": over_30, "Over_50": over_50
    }

def price_band_report(actual, predicted):
    """Evaluate regression performance separately across diamond price ranges."""

    # Build a row-level error table before grouping diamonds by actual price.
    report_df = pd.DataFrame({"Actual": actual, "Predicted": predicted})
    report_df["Absolute_Error"] = np.abs(report_df["Actual"] - report_df["Predicted"])
    report_df["APE"] = report_df["Absolute_Error"] / np.maximum(report_df["Actual"], 1) * 100

    # Price bands make it easier to see whether performance changes for expensive diamonds.
    bins = [0, 1000, 3000, 5000, 10000, np.inf]
    labels = ["<$1K", "$1K-$3K", "$3K-$5K", "$5K-$10K", "$10K+"]

    report_df["Price_Band"] = pd.cut(report_df["Actual"], bins=bins, labels=labels, include_lowest=True)

    rows = []

    # Calculate the most useful error metrics inside each price band.
    for band, group in report_df.groupby("Price_Band", observed=True):
        rows.append({
            "Price_Band": str(band), "Count": len(group),
            "MAE": group["Absolute_Error"].mean(), "Median_AE": group["Absolute_Error"].median(),
            "MAPE": group["APE"].mean(), "Median_APE": group["APE"].median(),
            "P90_APE": np.percentile(group["APE"], 90),
            "Within_10": (group["APE"] <= 10).mean() * 100,
            "Within_20": (group["APE"] <= 20).mean() * 100,
            "Over_30": (group["APE"] > 30).mean() * 100
        })

    return pd.DataFrame(rows)