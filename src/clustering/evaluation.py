"""Evaluation and practical model selection for customer segmentation."""

from __future__ import annotations

from typing import List, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import euclidean_distances

from .clustering import CandidateSolution
from .config import SegmentationConfig

EPS = 1e-9


def _entropy_balance(percentages: np.ndarray) -> float:
    """Measure how evenly observations are distributed across clusters."""
    if len(percentages) <= 1:
        return 0.0

    probs = percentages / percentages.sum()
    entropy = -np.sum(probs * np.log(probs + EPS))
    return float(entropy / np.log(len(probs)))


def _stratified_sample(
    labels: np.ndarray,
    max_samples: int,
    random_state: int,
) -> np.ndarray:
    """Sample observations while retaining representation from each cluster."""
    rng = np.random.default_rng(random_state)
    unique = np.unique(labels)

    if len(labels) <= max_samples:
        return np.arange(len(labels))

    selected: List[int] = []

    for label in unique:
        indices = np.where(labels == label)[0]
        proportional = int(round(max_samples * len(indices) / len(labels)))
        n_take = min(
            len(indices),
            max(min(20, len(indices)), proportional),
        )
        selected.extend(
            rng.choice(
                indices,
                size=n_take,
                replace=False,
            ).tolist()
        )

    selected_array = np.array(
        list(dict.fromkeys(selected)),
        dtype=int,
    )

    if len(selected_array) > max_samples:
        selected_array = rng.choice(
            selected_array,
            size=max_samples,
            replace=False,
        )

    return selected_array


def _separation_statistics(
    x: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Compute explicit between-cluster and within-cluster distances."""
    cluster_ids = np.unique(labels)

    if len(cluster_ids) < 2:
        return np.nan, np.nan, np.nan, np.nan

    centers = []
    within_rms = []

    for cluster_id in cluster_ids:
        points = x[labels == cluster_id]
        center = points.mean(axis=0)
        centers.append(center)

        rms = np.sqrt(
            np.mean(
                np.sum(
                    (points - center) ** 2,
                    axis=1,
                )
            )
        )
        within_rms.append(rms)

    centers_array = np.vstack(centers)
    distances = euclidean_distances(
        centers_array,
        centers_array,
    )
    upper = distances[np.triu_indices(len(centers_array), k=1)]

    avg_centroid_distance = float(np.mean(upper))
    min_centroid_distance = float(np.min(upper))
    avg_within = float(np.mean(within_rms))
    separation_ratio = min_centroid_distance / avg_within if avg_within > EPS else np.nan

    return (
        avg_centroid_distance,
        min_centroid_distance,
        avg_within,
        separation_ratio,
    )


def _kmeans_stability(
    x: np.ndarray,
    k: int,
    config: SegmentationConfig,
) -> float:
    """Estimate K-Means stability using pairwise Adjusted Rand Index."""
    label_sets = []

    for seed in config.stability_seeds:
        model = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=seed,
        )
        label_sets.append(model.fit_predict(x))

    scores = [
        adjusted_rand_score(label_sets[i], label_sets[j])
        for i in range(len(label_sets))
        for j in range(i + 1, len(label_sets))
    ]
    return float(np.mean(scores))


def evaluate_candidates(
    x: np.ndarray,
    solutions: Mapping[str, CandidateSolution],
    config: SegmentationConfig,
) -> pd.DataFrame:
    """Evaluate all clustering candidates using multiple complementary metrics."""
    rows = []

    for solution in solutions.values():
        labels = np.asarray(solution.labels)
        unique, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels) * 100.0

        if len(unique) < 2:
            continue

        sample_idx = _stratified_sample(
            labels,
            max_samples=min(
                config.silhouette_sample_size,
                len(labels),
            ),
            random_state=config.random_state,
        )

        silhouette = silhouette_score(
            x[sample_idx],
            labels[sample_idx],
        )
        davies_bouldin = davies_bouldin_score(x, labels)
        calinski_harabasz = calinski_harabasz_score(x, labels)

        (
            avg_centroid_distance,
            min_centroid_distance,
            avg_within_cluster_rms,
            separation_ratio,
        ) = _separation_statistics(x, labels)

        valid_structure = bool(
            percentages.min() >= config.min_segment_pct
            and percentages.max() <= config.max_segment_pct
        )

        stability = (
            _kmeans_stability(x, solution.k, config) if solution.method == "KMeans" else np.nan
        )

        rows.append(
            {
                "Solution": solution.name,
                "Method": solution.method,
                "K": solution.k,
                "CoveragePct": 100.0,
                "Silhouette": silhouette,
                "DaviesBouldin": davies_bouldin,
                "CalinskiHarabasz": calinski_harabasz,
                "SmallestSegmentPct": float(percentages.min()),
                "LargestSegmentPct": float(percentages.max()),
                "BalanceScore": _entropy_balance(percentages),
                "AvgCentroidDistance": avg_centroid_distance,
                "MinCentroidDistance": min_centroid_distance,
                "AvgWithinClusterRMS": avg_within_cluster_rms,
                "SeparationRatio": separation_ratio,
                "ValidCustomerSegments": valid_structure,
                "StabilityARI": stability,
                "MembershipConfidence": solution.membership_confidence,
                "AIC": solution.aic,
                "BIC": solution.bic,
            }
        )

    if not rows:
        raise ValueError("No valid multi-cluster solutions were produced.")

    result = pd.DataFrame(rows)

    sil_component = np.clip(
        (result["Silhouette"].fillna(-1.0) + 1.0) / 2.0,
        0.0,
        1.0,
    )
    db_component = 1.0 / (1.0 + result["DaviesBouldin"].fillna(999.0))

    separation = result["SeparationRatio"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    if separation.notna().any() and separation.max() > separation.min():
        separation_component = (
            (separation - separation.min()) / (separation.max() - separation.min())
        ).fillna(0.0)
    else:
        separation_component = pd.Series(
            0.5,
            index=result.index,
        )

    stability_or_confidence = (
        result["StabilityARI"].combine_first(result["MembershipConfidence"]).fillna(0.60)
    )
    k_span = max(float(result["K"].max() - result["K"].min()), 1.0)
    result["InterpretabilityScore"] = 1.0 - ((result["K"] - result["K"].min()) / k_span)

    result["CompositeScore"] = (
        config.weight_silhouette * sil_component
        + config.weight_davies_bouldin * db_component
        + config.weight_separation * separation_component
        + config.weight_balance * result["BalanceScore"].fillna(0.0)
        + config.weight_coverage * (result["CoveragePct"].fillna(0.0) / 100.0)
        + config.weight_stability_or_confidence * stability_or_confidence
        + config.weight_interpretability * result["InterpretabilityScore"]
    )

    result.loc[
        ~result["ValidCustomerSegments"],
        "CompositeScore",
    ] -= config.invalid_structure_penalty

    return result.sort_values(
        [
            "ValidCustomerSegments",
            "CompositeScore",
            "Silhouette",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def select_best_solution(
    comparison: pd.DataFrame,
) -> str:
    """Return the name of the best practical segmentation solution."""
    valid = comparison.loc[comparison["ValidCustomerSegments"]]
    if not valid.empty:
        return str(valid.iloc[0]["Solution"])

    return str(comparison.iloc[0]["Solution"])
