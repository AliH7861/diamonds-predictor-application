"""
Evaluate and compare candidate K-Means customer-segmentation solutions.

This module measures cluster separation, compactness, centroid distance, and
cluster-size balance for each candidate value of K.

It also creates a combined selection score that favors well-separated,
distinct, reasonably balanced, and simpler clustering solutions.
"""

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

def evaluate_stability(matrix, model, sample_size: int = 5000) -> float:
    """Measure agreement with repeat fits using adjusted Rand index."""
    rng = np.random.default_rng(42)
    size = min(sample_size, len(matrix))
    indices = np.sort(rng.choice(len(matrix), size=size, replace=False))
    sample = matrix[indices]
    reference = model.predict(sample)
    scores = []
    for seed in (7, 21, 84):
        repeated = KMeans(
            n_clusters=model.n_clusters,
            init="k-means++",
            n_init=5,
            max_iter=300,
            random_state=seed,
        ).fit_predict(sample)
        scores.append(adjusted_rand_score(reference, repeated))
    return float(np.mean(scores))


def evaluate_kmeans(model, matrix, labels, sample_size: int = 5000) -> dict:
    """Calculate separation, compactness, centroid distance, and cluster balance metrics."""

    # Measure how many observations belong to each cluster.
    counts = pd.Series(labels).value_counts().sort_index()
    percentages = counts / len(labels) * 100

    # Compare every cluster centroid with every other centroid.
    centers = model.cluster_centers_
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)

    # Ignore each centroid's zero distance from itself.
    distances[distances == 0] = np.nan

    # Silhouette measures how well observations fit their own cluster versus neighboring clusters.
    metric_size = min(sample_size, len(labels))
    rng = np.random.default_rng(42)
    metric_indices = np.sort(rng.choice(len(matrix), size=metric_size, replace=False))
    metric_matrix = matrix[metric_indices]
    metric_labels = np.asarray(labels)[metric_indices]
    silhouette = silhouette_score(metric_matrix, metric_labels)
    calinski = calinski_harabasz_score(metric_matrix, metric_labels)
    davies = davies_bouldin_score(metric_matrix, metric_labels)
    stability = evaluate_stability(matrix, model, sample_size=metric_size)

    return {
        "K": int(model.n_clusters), "Silhouette": float(silhouette),
        "Calinski_Harabasz": float(calinski),
        "Davies_Bouldin": float(davies),
        "Stability_ARI": stability,
        "Inertia": float(model.inertia_),
        "Min_Cluster_Pct": float(percentages.min()), "Max_Cluster_Pct": float(percentages.max()),
        "Cluster_Size_CV": float(counts.std(ddof=0) / counts.mean()),
        "Min_Centroid_Distance": float(np.nanmin(distances))
    }

def add_selection_score(metrics: pd.DataFrame) -> pd.DataFrame:
    """Combine clustering quality metrics into one weighted selection score."""

    result = metrics.copy()

    def scale(column: str, higher_is_better: bool = True) -> pd.Series:
        """Normalize one metric to a 0-1 scale for fair comparison."""

        values = result[column].astype(float)
        spread = values.max() - values.min()

        # If every candidate has the same value, give them equal normalized scores.
        normalized = pd.Series(1.0, index=result.index) if spread == 0 else (values - values.min()) / spread

        return normalized if higher_is_better else 1 - normalized

    # Combine complementary quality measures. Silhouette remains the strongest
    # signal, but stability and two independent separation metrics prevent a
    # fragile result from winning on one number alone.
    result["Selection_Score"] = (
        0.30 * scale("Silhouette")
        + 0.10 * scale("Calinski_Harabasz")
        + 0.10 * scale("Davies_Bouldin", higher_is_better=False)
        + 0.15 * scale("Stability_ARI")
        + 0.10 * scale("Min_Cluster_Pct")
        + 0.05 * scale("Cluster_Size_CV", higher_is_better=False)
        + 0.10 * scale("Interpretability_Ratio")
        + 0.10 * scale("K", higher_is_better=False)
    )

    result["Separation_Check"] = result["Silhouette"] >= 0.10
    result["Stability_Check"] = result["Stability_ARI"] >= 0.75
    result["Cluster_Size_Check"] = result["Min_Cluster_Pct"] >= 2.0
    result["Interpretability_Check"] = result["Interpretability_Ratio"] >= 0.40
    result["Quality_Checks_Passed"] = result[
        [
            "Separation_Check", "Stability_Check", "Cluster_Size_Check",
            "Interpretability_Check",
        ]
    ].all(axis=1)

    return result.sort_values("K").reset_index(drop=True)
