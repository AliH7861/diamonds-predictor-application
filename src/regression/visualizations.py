"""
Visualization utilities for the diamond price-regression workflow.

This module creates and saves exploratory data-analysis plots, validation model
comparisons, permutation feature-importance charts, feature-to-price relationship
plots, and final test-error diagnostics.

Every figure is saved to disk and also displayed in the notebook so the same
visuals can be reused in reports, documentation, and project analysis.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from .training import predict_run

def finish_figure(fig, output_dir, name):
    """Save a Matplotlib figure and display it in the notebook."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=140, bbox_inches="tight")

    plt.show()
    plt.close(fig)

def plot_eda(df, output_dir):
    """Plot key regression feature distributions and numeric correlations."""

    # Show distributions for price and the main physical diamond measurements.
    features = ["price", "carat", "depth", "table", "x", "z"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for ax, feature in zip(axes.flat, features):
        ax.hist(df[feature], bins=40)
        ax.set(title=f"{feature} distribution", xlabel=feature, ylabel="Count")

    finish_figure(fig, output_dir, "distributions")

    # Compare linear relationships across all numeric features.
    numeric = df.select_dtypes("number").corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(numeric, vmin=-1, vmax=1)

    ax.set_xticks(range(len(numeric)), numeric.columns, rotation=90)
    ax.set_yticks(range(len(numeric)), numeric.columns)
    ax.set_title("Numeric features and price: Pearson correlation")

    fig.colorbar(image, ax=ax)

    finish_figure(fig, output_dir, "correlations")

def plot_comparison(results, output_dir, name):
    """Compare regression candidates using validation MAE."""

    # Higher bars represent worse validation performance, so lower is better.
    df = pd.DataFrame(results).sort_values("MAE", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    labels = df["Feature_Set"] + " / " + df["Algorithm"]

    ax.barh(labels, df["MAE"])
    ax.set(xlabel="Validation MAE ($), lower is better", title="Regression model comparison")

    finish_figure(fig, output_dir, name)

def plot_top_features(run, frame, y, output_dir, name, max_samples=1500, repeats=3):
    """Rank original features by permutation importance and plot their relationship with price."""

    rng = np.random.default_rng(42)

    # Use a random validation sample so repeated permutation predictions stay manageable.
    selected = rng.choice(len(frame), min(max_samples, len(frame)), replace=False)
    X = frame.iloc[selected].reset_index(drop=True)
    y = np.asarray(y)[selected]

    # Baseline MAE is compared against MAE after each feature is shuffled.
    baseline = mean_absolute_error(y, predict_run(run, X))
    rows = []

    # Shuffling one feature breaks its relationship with price while leaving others unchanged.
    for feature in X.columns:
        increases = []

        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[feature] = rng.permutation(X[feature].to_numpy())

            shuffled_mae = mean_absolute_error(y, predict_run(run, shuffled))
            increases.append(shuffled_mae - baseline)

        rows.append({
            "Feature": feature,
            "MAE increase": np.mean(increases),
            "Repeat std": np.std(increases)
        })

    # Larger MAE increases mean the model relied more heavily on that feature.
    ranking = pd.DataFrame(rows).sort_values("MAE increase", ascending=False)
    top = ranking.head(8)
    ordered = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(ordered["Feature"], ordered["MAE increase"], xerr=ordered["Repeat std"], capsize=3)
    ax.set(xlabel="Increase in validation MAE ($) after shuffling", title=f"{name}: top 8 features")

    finish_figure(fig, output_dir, f"{name}_importance")

    # Show how the strongest original features relate to actual diamond prices.
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))

    for ax, feature in zip(axes.flat, top["Feature"]):

        # Numeric features use scatter plots.
        if pd.api.types.is_numeric_dtype(X[feature]):
            ax.scatter(X[feature], y, s=8, alpha=0.25)

        # Categorical features use price distributions for each category.
        else:
            categories = sorted(X[feature].dropna().unique())
            groups = [y[X[feature].to_numpy() == category] for category in categories]

            ax.boxplot(groups, showfliers=False)
            ax.set_xticks(range(1, len(categories) + 1), categories, rotation=45)

        ax.set(xlabel=feature, ylabel="Actual price ($)", title=feature)

    # Hide unused subplot spaces if fewer than eight features are available.
    for ax in list(axes.flat)[len(top):]:
        ax.set_visible(False)

    fig.suptitle("Top feature relationships with price: validation sample", fontsize=15)

    finish_figure(fig, output_dir, f"{name}_relationships")

    # Save the full ranking so it can also be inspected as a table.
    ranking.to_csv(Path(output_dir) / f"{name}_importance.csv", index=False)

    return ranking

def plot_errors(actual, predicted, band_report, output_dir):
    """Plot final test predictions, residuals, percentage errors, and price-band MAE."""

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    # Residuals show dollar error while APE shows error relative to actual price.
    residuals = actual - predicted
    ape = np.abs(residuals) / np.maximum(actual, 1) * 100

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Actual vs predicted prices; points close to the diagonal are more accurate.
    axes[0, 0].scatter(actual, predicted, s=5, alpha=0.25)
    axes[0, 0].plot([actual.min(), actual.max()], [actual.min(), actual.max()])
    axes[0, 0].set(xlabel="Actual price ($)", ylabel="Predicted price ($)", title="Actual vs predicted")

    # Residuals should ideally stay centered around zero across prediction values.
    axes[0, 1].scatter(predicted, residuals, s=5, alpha=0.25)
    axes[0, 1].axhline(0)
    axes[0, 1].set(xlabel="Predicted price ($)", ylabel="Actual minus predicted ($)", title="Residuals")

    # Distribution of percentage-error severity.
    axes[1, 0].hist(ape, bins=60)
    axes[1, 0].set(xlabel="Absolute percentage error (%)", ylabel="Count", title="Percentage error distribution")

    # Compare dollar error across actual diamond price ranges.
    axes[1, 1].bar(band_report["Price_Band"], band_report["MAE"])
    axes[1, 1].set(xlabel="Actual price band", ylabel="MAE ($)", title="MAE by price band")

    finish_figure(fig, output_dir, "final_test_errors")