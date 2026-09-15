"""
Evaluate and compare candidate K-Means customer-segmentation solutions.

This module measures cluster separation, compactness, centroid distance, and
cluster-size balance for each candidate value of K.

It also creates a combined selection score that favors well-separated,
distinct, reasonably balanced, and simpler clustering solutions.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score

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
    silhouette = silhouette_score(matrix, labels, sample_size=min(sample_size, len(labels)), random_state=42)

    return {
        "K": int(model.n_clusters), "Silhouette": float(silhouette),
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

    # Favor separation most heavily, then centroid distinctness and cluster balance.
    result["Selection_Score"] = (
        0.45 * scale("Silhouette")
        + 0.25 * scale("Min_Centroid_Distance")
        + 0.15 * scale("Min_Cluster_Pct")
        + 0.10 * scale("Cluster_Size_CV", higher_is_better=False)
        + 0.05 * scale("K", higher_is_better=False)
    )

    return result.sort_values("K").reset_index(drop=True)