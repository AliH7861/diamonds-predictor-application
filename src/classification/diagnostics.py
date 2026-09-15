"""
Diagnostics and feature analysis for the diamond clarity-classification models.

This file helps explain what the trained classification models learned.

It:
- Shows the processed features used by each experiment.
- Displays descriptive statistics and feature distributions.
- Checks correlations between features and the encoded clarity target.
- Measures feature importance using permutation importance.
- Shows how the most important features differ across clarity families.

Permutation importance works by shuffling one feature at a time.
If Macro F1 drops after that feature is shuffled, the model was relying on it.
A larger drop means that feature was more important to the model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .cleaning import CLARITY_CLASSES
from .preprocessing import RANDOM_STATE

# FEATURE DISTRIBUTION
def plot_all_feature_distributions(processed_df, experiment_name):
    """Plot the distribution of every processed model feature."""

    columns = processed_df.columns

    # Use four plots per row so all processed features can be inspected together.
    n_cols = 4
    n_rows = int(np.ceil(len(columns) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = np.array(axes).flatten()

    for i, column in enumerate(columns):

        # ColumnTransformer adds prefixes such as numeric__ and categorical__.
        # Remove them only from the plot label so the original DataFrame stays unchanged.
        clean_name = column.replace("numeric__", "").replace("categorical__", "")

        # Histograms help reveal skew, scaling differences, sparse encoded columns,
        # and unusual distributions before interpreting the trained model.
        axes[i].hist(processed_df[column], bins=30)
        axes[i].set_title(clean_name)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")

    # Hide unused subplot spaces in the final row.
    for i in range(len(columns), len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"{experiment_name} — Feature Distributions", fontsize=18)
    plt.tight_layout()
    plt.show()


# CORRELATION MATRIX
def plot_correlation_matrix(processed_df, y_train, features, experiment_name):
    """Plot correlations between processed features and encoded clarity."""

    # Work on a copy so this analysis never changes the training data.
    correlation_df = processed_df[list(features)].copy()
    correlation_df["Clarity_Target"] = y_train

    # Pearson correlation is mainly exploratory here.
    # Clarity is encoded as ordered classes, so correlation should not be treated
    # as proof that a feature directly determines clarity.
    correlation_matrix = correlation_df.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(correlation_matrix, vmin=-1, vmax=1)

    # Clean preprocessing prefixes from the displayed feature names.
    labels = [column.replace("numeric__", "").replace("categorical__", "") for column in correlation_matrix.columns]

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Write the correlation value inside every matrix cell.
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = correlation_matrix.iloc[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title(f"{experiment_name} — Feature Correlation Matrix")
    fig.colorbar(image, ax=ax, label="Pearson Correlation")

    plt.tight_layout()
    plt.show()
# EXPERIMENT DIAGNOSTICS

def show_diagnostics(prepared):
    """Show processed data, statistics, distributions, and correlations."""

    from IPython.display import display

    # Every experiment may contain a different feature set or preprocessing setup.
    for number, experiment in prepared["experiments"].items():

        # Rebuild a readable DataFrame from the processed training matrix.
        feature_names = experiment["preprocessor"].get_feature_names_out()
        frame = pd.DataFrame(experiment["X_train"], columns=feature_names)

        name = f"Experiment {number}"

        print(name)

        # Show the processed rows and standard descriptive statistics.
        display(frame)
        display(frame.describe().T)

        # Inspect individual distributions and relationships between features.
        plot_all_feature_distributions(frame, name)
        plot_correlation_matrix(frame, prepared["y_train"], frame.columns, name)


# MODEL PREDICTION
def predict_clarity(model, model_type, preprocessor, X):
    """Preprocess raw features and return predicted clarity-family labels."""

    # Always reuse the preprocessing pipeline fitted during training.
    X_processed = preprocessor.transform(X).astype(np.float32)

    # ANN returns class probabilities, so take the class with the largest probability.
    if model_type == "ANN":
        return np.argmax(model.predict(X_processed, verbose=0), axis=1)

    # Random Forest and XGBoost already return class IDs.
    return model.predict(X_processed)

# PERMUTATION FEATURE IMPORTANCE
def rank_features(model, model_type, preprocessor, X, y, repeats=5, random_state=RANDOM_STATE):
    """
    Rank original features by how much validation Macro F1 drops after shuffling.

    Large drop  = model depends heavily on the feature.
    Near zero   = feature has little effect on predictions.
    Negative    = shuffling happened to slightly improve the model score.
    """

    if repeats < 1:
        raise ValueError("repeats must be at least 1.")

    rng = np.random.default_rng(random_state)

    # Calculate normal model performance before changing any feature.
    predictions = predict_clarity(model, model_type, preprocessor, X)
    baseline = f1_score(y, predictions, labels=range(len(CLARITY_CLASSES)), average="macro", zero_division=0)

    rows = []

    # Shuffle each ORIGINAL feature separately.
    # This keeps importance understandable as "carat", "cut", "price", etc.
    # instead of ranking individual one-hot encoded columns.
    for feature in X.columns:

        drops = []

        # Repeat the shuffle because a single random permutation may be noisy.
        for _ in range(repeats):

            shuffled = X.copy()

            # Shuffling destroys this feature's relationship with clarity while
            # keeping every other feature exactly the same.
            shuffled[feature] = rng.permutation(X[feature].to_numpy())

            predictions = predict_clarity(model, model_type, preprocessor, shuffled)
            score = f1_score(y, predictions, labels=range(len(CLARITY_CLASSES)), average="macro", zero_division=0)

            # Convert the score decrease into percentage points.
            drops.append((baseline - score) * 100)

        # Store the average importance and how much it changed across repeats.
        row = {"Feature": feature, "Macro F1 drop (pp)": np.mean(drops), "Repeat std (pp)": np.std(drops)}
        rows.append(row)

    # The feature causing the largest F1 decrease appears first.
    importance = pd.DataFrame(rows)
    importance = importance.sort_values("Macro F1 drop (pp)", ascending=False).reset_index(drop=True)

    return importance

# TOP FEATURE ANALYSIS
def plot_top_features(prepared, trained_models, results, experiment_number, max_samples=2000, repeats=5):
    """Rank and visualize the strongest features from one classification experiment."""

    experiment_name = f"Experiment {experiment_number}"
    scores = pd.DataFrame(results)

    # Find the best-performing model within the selected experiment.
    # Macro F1 is used because each clarity family should contribute equally.
    experiment_scores = scores[scores["Experiment"] == experiment_name]
    best = experiment_scores.sort_values("Macro_F1", ascending=False).iloc[0]

    run = f"{experiment_name} - {best['Model']}"
    info = trained_models[run]
    preprocessor = prepared["experiments"][experiment_number]["preprocessor"]

    # Use the ORIGINAL validation columns instead of processed columns.
    # This means the final plots show understandable units and categories.
    validation_rows = prepared["diamonds"].iloc[prepared["valid_indices"]]
    feature_names = list(preprocessor.feature_names_in_)
    X = validation_rows[feature_names].copy()
    y = prepared["y_valid"]

    # Permutation importance requires many repeated predictions.
    # Limit the validation rows for speed while preserving class proportions.
    if max_samples is not None and len(X) > max_samples:

        indices = np.arange(len(X))

        selected, _ = train_test_split(
            indices, train_size=max_samples, stratify=y, random_state=RANDOM_STATE
        )

        X = X.iloc[selected].copy()
        y = y[selected]

    X = X.reset_index(drop=True)

    # Rank every original feature and keep the eight strongest for visualization.
    importance = rank_features(info["model"], info["type"], preprocessor, X, y, repeats=repeats)
    top = importance.head(8)

    print(f"{run}: top features ranked on {len(X):,} validation rows, {repeats} shuffles per feature.")

    # PLOT 1 — FEATURE IMPORTANCE
    # Reverse the ranking so the strongest feature appears at the top of the chart.
    ordered = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Bar length = average Macro F1 loss after shuffling the feature.
    # Error bar = variation across repeated shuffles.
    ax.barh(ordered["Feature"], ordered["Macro F1 drop (pp)"], xerr=ordered["Repeat std (pp)"], capsize=3)

    # Features near zero had little effect on the model.
    ax.axvline(0, linewidth=1)

    ax.set_xlabel("Decrease in validation Macro F1 (percentage points)")
    ax.set_title(f"{run}: Top 8 Features for Predicting Clarity")

    plt.tight_layout()
    plt.show()

    # PLOT 2 — IMPORTANT FEATURES VS ACTUAL CLARITY

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for ax, feature in zip(axes.flat, top["Feature"]):

        # Numeric features such as carat or price are shown as boxplots so we
        # can compare their distributions across the actual clarity families.
        if pd.api.types.is_numeric_dtype(X[feature]):

            groups = [X.loc[y == family, feature].dropna().to_numpy() for family in range(len(CLARITY_CLASSES))]

            # Hide extreme points so the central distributions remain readable.
            ax.boxplot(groups, showfliers=False)

            ax.set_xticks(range(1, len(CLARITY_CLASSES) + 1), CLARITY_CLASSES)
            ax.set_ylabel(f"{feature} (original units)")

        # Categorical features such as cut or color are shown as proportions.
        else:

            feature_values = X[feature].fillna("Missing")
            proportions = pd.crosstab(pd.Series(y, name="Clarity"), feature_values, normalize="index")

            # Ensure all clarity families remain visible even if a sampled class
            # happens to contain no observations for a particular category.
            proportions = proportions.reindex(range(len(CLARITY_CLASSES)), fill_value=0)
            proportions.index = CLARITY_CLASSES

            proportions.plot.bar(stacked=True, ax=ax, rot=0)

            ax.set_ylabel("Category proportion")
            ax.legend(title=feature, fontsize=7)

        ax.set_xlabel("Actual clarity family")
        ax.set_title(feature)

    # Hide unused subplot spaces if fewer than eight features are available.
    for ax in list(axes.flat)[len(top):]:
        ax.set_visible(False)

    title = f"{run}: Top Features vs Actual Diamond Clarity\nValidation sample; numeric outlier points hidden for readability"
    fig.suptitle(title, fontsize=15)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()

    # Return the entire ranking so it can also be displayed, saved, or analyzed.
    return importance

