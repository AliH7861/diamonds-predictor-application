"""
Evaluation metrics for the diamond clarity-classification models.

This file gives every classification model the same evaluation system so
ANN, Random Forest, and XGBoost can be compared fairly.

The main metric is Macro F1 because the five clarity families are imbalanced.
Additional metrics measure overall accuracy, performance across classes,
and how far incorrect predictions are from the correct clarity family.

Clarity families are ordered:
I → SI → VS → VVS → IF

Because of this order, predicting a neighboring family is treated as a
smaller mistake than predicting a family several levels away.
"""

import numpy as np
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,f1_score,
    precision_score, recall_score)

# CLASSIFICATION METRICS
def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate performance metrics for the five clarity families."""

    # Convert both inputs to NumPy arrays so subtraction and comparisons work consistently.
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Clarity families are ordered from I → SI → VS → VVS → IF.
    # The absolute difference tells us how many families away a prediction was.
    # Example: actual SI (1), predicted VS (2) → family error = 1.
    family_error = np.abs(y_true - y_pred)

    # Overall percentage of predictions that matched the exact clarity family.
    accuracy = accuracy_score(y_true, y_pred) * 100

    # Macro metrics calculate each clarity family separately and then average them.
    # This prevents the larger SI/VS classes from dominating the evaluation.
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100

    # Weighted F1 still considers every class but gives larger classes more influence.
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100

    # Balanced accuracy calculates recall for each family and averages the results.
    # This gives every clarity family equal importance despite class imbalance.
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred) * 100

    # Average number of clarity-family levels the predictions were away from the truth.
    mean_family_error = family_error.mean()

    # Percentage of predictions that were either exactly correct or only one family away.
    within_1_family = (family_error <= 1).mean() * 100

    # Percentage of predictions that missed the correct clarity by more than one family.
    severe_error = (family_error > 1).mean() * 100

    # Return one consistent metric dictionary for notebooks, tests, and model comparison.
    return {
        "Accuracy": accuracy,
        "Macro_Precision": macro_precision,
        "Macro_Recall": macro_recall,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1,
        "Balanced_Accuracy": balanced_accuracy,
        "Mean_Family_Error": mean_family_error,
        "Within_1_Family": within_1_family,
        "Severe_Error": severe_error
    }

# MODEL EVALUATION RESULT
def evaluate_model(y_true, y_pred, experiment_name: str, model_name: str) -> dict:
    """Combine model identifiers with its classification metrics."""

    # Calculate the shared metric set first.
    metrics = classification_metrics(y_true, y_pred)

    # Add experiment and model names so results from every run can be placed
    # directly into the same comparison DataFrame.
    return {"Experiment": experiment_name, "Model": model_name, **metrics}

