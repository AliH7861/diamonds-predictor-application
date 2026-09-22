"""Shared helpers for five-family ordinal clarity classifiers."""

import numpy as np

N_CLASSES = 5
N_THRESHOLDS = N_CLASSES - 1


def make_ordinal_targets(target) -> np.ndarray:
    """Convert class IDs into four cumulative greater-than targets."""
    values = np.asarray(target, dtype=int)
    return (values[:, None] > np.arange(N_THRESHOLDS)).astype(np.float32)


def cumulative_to_class_probabilities(cumulative) -> np.ndarray:
    """Convert monotonic cumulative probabilities into five class probabilities."""
    values = np.minimum.accumulate(np.asarray(cumulative, dtype=float), axis=1)
    probabilities = np.column_stack(
        [
            1.0 - values[:, 0],
            values[:, 0] - values[:, 1],
            values[:, 1] - values[:, 2],
            values[:, 2] - values[:, 3],
            values[:, 3],
        ]
    )
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def predict_ordinal_models(models, matrix) -> tuple[np.ndarray, np.ndarray]:
    """Predict with four binary boundary models."""
    cumulative = np.column_stack(
        [model.predict_proba(matrix)[:, 1] for model in models]
    )
    probabilities = cumulative_to_class_probabilities(cumulative)
    return probabilities.argmax(axis=1), probabilities


def predict_ordinal_ann(model, matrix) -> tuple[np.ndarray, np.ndarray]:
    """Decode the ANN's four cumulative sigmoid outputs."""
    cumulative = np.asarray(model.predict(matrix, verbose=0))
    probabilities = cumulative_to_class_probabilities(cumulative)
    return probabilities.argmax(axis=1), probabilities
