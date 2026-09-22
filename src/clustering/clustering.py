"""Candidate clustering algorithms for market-wide diamond segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture

from .config import SegmentationConfig


@dataclass
class CandidateSolution:
    """Labels and metadata for one clustering candidate."""

    name: str
    method: str
    k: int
    labels: np.ndarray
    model: object
    membership_confidence: float | None = None
    aic: float | None = None
    bic: float | None = None


def run_kmeans_candidates(
    x: np.ndarray,
    config: SegmentationConfig,
) -> Dict[str, CandidateSolution]:
    """Run K-Means for each configured K value."""
    solutions: Dict[str, CandidateSolution] = {}

    for k in config.k_values:
        model = KMeans(
            n_clusters=k,
            n_init=30,
            random_state=config.random_state,
        )
        labels = model.fit_predict(x)
        name = f"KMeans_k{k}"
        solutions[name] = CandidateSolution(
            name=name,
            method="KMeans",
            k=k,
            labels=labels,
            model=model,
        )

    return solutions


def run_gmm_candidates(
    x: np.ndarray,
    config: SegmentationConfig,
) -> Dict[str, CandidateSolution]:
    """Run diagonal-covariance Gaussian Mixture candidates."""
    solutions: Dict[str, CandidateSolution] = {}

    for k in config.k_values:
        model = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            n_init=3,
            random_state=config.random_state,
            reg_covar=1e-6,
        )
        model.fit(x)
        labels = model.predict(x)
        probabilities = model.predict_proba(x)

        name = f"GMM_k{k}"
        solutions[name] = CandidateSolution(
            name=name,
            method="GMM",
            k=k,
            labels=labels,
            model=model,
            membership_confidence=float(probabilities.max(axis=1).mean()),
            aic=float(model.aic(x)),
            bic=float(model.bic(x)),
        )

    return solutions


def run_agglomerative_candidates(
    x: np.ndarray,
    config: SegmentationConfig,
) -> Dict[str, CandidateSolution]:
    """Run scalable hierarchical candidates using sample + centroid assignment.

    Full Ward clustering on ~54k rows is unnecessarily expensive for this
    application. A deterministic sample is clustered hierarchically and the
    remaining rows are assigned to the nearest discovered centroid.
    """
    rng = np.random.default_rng(config.random_state)
    sample_size = min(config.agglomerative_sample_size, len(x))
    sample_idx = rng.choice(
        len(x),
        size=sample_size,
        replace=False,
    )
    sample = x[sample_idx]

    solutions: Dict[str, CandidateSolution] = {}

    for k in config.k_values:
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
        )
        sample_labels = model.fit_predict(sample)

        centers = np.vstack(
            [sample[sample_labels == cluster_id].mean(axis=0) for cluster_id in range(k)]
        )
        labels = euclidean_distances(x, centers).argmin(axis=1)

        name = f"Agglomerative_k{k}"
        solutions[name] = CandidateSolution(
            name=name,
            method="Agglomerative",
            k=k,
            labels=labels,
            model={
                "sample_model": model,
                "centers": centers,
            },
        )

    return solutions


def run_candidate_clustering(
    x: np.ndarray,
    config: SegmentationConfig,
) -> Dict[str, CandidateSolution]:
    """Run all enabled production candidate algorithms."""
    solutions: Dict[str, CandidateSolution] = {}

    if "kmeans" in config.enabled_methods:
        solutions.update(run_kmeans_candidates(x, config))

    if "gmm" in config.enabled_methods:
        solutions.update(run_gmm_candidates(x, config))

    if "agglomerative" in config.enabled_methods:
        solutions.update(run_agglomerative_candidates(x, config))

    if not solutions:
        raise ValueError("No clustering methods are enabled.")

    return solutions
