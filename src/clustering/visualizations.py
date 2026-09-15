"""
Export comparison and selected-segmentation figures for the K-Means clustering workflow.

This module creates the elbow and silhouette comparison plots used to compare
candidate K values, then visualizes the selected segmentation with a two-dimensional
PCA projection and a cluster-size distribution chart.

Figures are saved directly to disk for use in notebooks, reports, and project documentation.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA

def save_comparison_plots(metrics: pd.DataFrame, output_dir: str | Path) -> None:
    """Save the inertia and silhouette comparison plots for candidate K values."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compare compactness with the elbow curve and separation with silhouette score.
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(metrics["K"], metrics["Inertia"], marker="o")
    axes[0].set(title="Elbow comparison", xlabel="K", ylabel="Inertia")

    axes[1].plot(metrics["K"], metrics["Silhouette"], marker="o")
    axes[1].set(title="Cluster separation", xlabel="K", ylabel="Silhouette score")

    figure.tight_layout()
    figure.savefig(output_dir / "k_comparison.png", dpi=160, bbox_inches="tight")

    plt.close(figure)

def save_selected_plots(matrix, labels, summary: pd.DataFrame, output_dir: str | Path) -> None:
    """Save PCA and cluster-size plots for the selected segmentation."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a reproducible sample so the PCA plot stays readable on large datasets.
    sample_size = min(6000, len(labels))
    indices = pd.Series(range(len(labels))).sample(sample_size, random_state=42).to_numpy()

    # Reduce the encoded clustering matrix to two dimensions for visualization only.
    coordinates = PCA(n_components=2, random_state=42).fit_transform(matrix[indices])

    figure, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=labels[indices], s=8, alpha=0.5)

    ax.set(title="Selected buyer segments in two dimensions", xlabel="PCA 1", ylabel="PCA 2")
    figure.colorbar(scatter, ax=ax, label="Cluster")

    figure.tight_layout()
    figure.savefig(output_dir / "selected_clusters_pca.png", dpi=160, bbox_inches="tight")

    plt.close(figure)

    # Show what percentage of purchase profiles belongs to each selected cluster.
    figure, ax = plt.subplots(figsize=(8, 4.5))

    ax.bar(summary["Cluster"].astype(str), summary["Percentage"])
    ax.set(title="Selected cluster sizes", xlabel="Cluster", ylabel="Profiles (%)")

    figure.tight_layout()
    figure.savefig(output_dir / "selected_cluster_sizes.png", dpi=160, bbox_inches="tight")

    plt.close(figure)