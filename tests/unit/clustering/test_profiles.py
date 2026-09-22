"""Unit checks for features, algorithms, profiles, and RAG helpers."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.clustering.clustering import run_candidate_clustering
from src.clustering.config import SegmentationConfig
from src.clustering.feature_engineering import (
    CUSTOMER_PILLARS,
    build_customer_pillars,
    build_segmentation_features,
    split_geometry_anomalies,
)
from src.clustering.profiling import attach_profile_names, build_profiles
from src.clustering.rag import answer_profile_question, filter_diamonds_by_profile
from tests.helpers import make_diamonds


def _prepared(rows=180):
    frame = make_diamonds(rows=rows).drop(columns="Unnamed: 0")
    config = SegmentationConfig(
        k_values=(3,),
        silhouette_sample_size=100,
        agglomerative_sample_size=100,
        stability_seeds=(0, 1),
    )
    engineered = build_segmentation_features(frame, config)
    segmentable, anomalies = split_geometry_anomalies(engineered, config)
    pillars = build_customer_pillars(segmentable)
    matrix = StandardScaler().fit_transform(pillars[list(CUSTOMER_PILLARS)])
    return config, pillars, anomalies, matrix


def test_feature_generation_is_finite_and_screens_geometry_anomalies():
    config, pillars, anomalies, _ = _prepared()
    assert np.isfinite(pillars[list(CUSTOMER_PILLARS)].to_numpy()).all()
    assert len(pillars) + len(anomalies) <= 180
    assert len(CUSTOMER_PILLARS) == 8

    malformed = make_diamonds(rows=30).drop(columns=["price"])
    with pytest.raises(ValueError, match="missing columns"):
        build_segmentation_features(malformed, config)


def test_every_algorithm_returns_labels_and_profiles_are_grounded():
    config, pillars, _, matrix = _prepared()
    solutions = run_candidate_clustering(matrix, config)
    assert set(solution.method for solution in solutions.values()) == {
        "KMeans",
        "GMM",
        "Agglomerative",
    }
    assert all(len(solution.labels) == len(pillars) for solution in solutions.values())

    labels = solutions["KMeans_k3"].labels
    profiles, _, _ = build_profiles(pillars, matrix, labels)
    enriched = attach_profile_names(pillars, labels, profiles)
    assert sum(profile.segment_share_pct for profile in profiles) == pytest.approx(100.0)
    assert all(profile.top_priorities and profile.tradeoffs for profile in profiles)
    assert filter_diamonds_by_profile(enriched, profiles[0].name).shape[0] > 0
    answer = answer_profile_question(f"What is {profiles[0].name}?", profiles)
    assert profiles[0].name in answer
    assert "Product-derived" in answer


def test_kmeans_labels_are_deterministic():
    config, _, _, matrix = _prepared()
    first = run_candidate_clustering(matrix, config)["KMeans_k3"].labels
    second = run_candidate_clustering(matrix, config)["KMeans_k3"].labels
    assert np.array_equal(first, second)
