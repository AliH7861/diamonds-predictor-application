"""Configuration for customer-oriented diamond segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SegmentationConfig:
    """Runtime configuration for the segmentation pipeline.

    Defaults match the experimentation that produced stable, interpretable
    market-wide segments, while remaining configurable for integration.
    """

    random_state: int = 42
    k_values: Tuple[int, ...] = (3, 5, 7, 10)

    # Practical segmentation constraints. These are project heuristics,
    # not universal clustering laws.
    min_segment_pct: float = 3.0
    max_segment_pct: float = 60.0

    silhouette_sample_size: int = 7_000
    agglomerative_sample_size: int = 8_000
    stability_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)

    # Peer-value feature construction.
    min_peer_group_size: int = 30

    # Geometry anomaly screening.
    geometry_low_quantile: float = 0.001
    geometry_high_quantile: float = 0.999
    aspect_ratio_hard_quantile: float = 0.9995
    min_extreme_geometry_flags: int = 2

    # Candidate clustering methods kept in the production comparison.
    enabled_methods: Tuple[str, ...] = (
        "kmeans",
        "gmm",
        "agglomerative",
    )

    # Weighted selection heuristic. Weights sum to 1.0.
    weight_silhouette: float = 0.25
    weight_davies_bouldin: float = 0.15
    weight_separation: float = 0.15
    weight_balance: float = 0.15
    weight_coverage: float = 0.10
    weight_stability_or_confidence: float = 0.10
    weight_interpretability: float = 0.10

    invalid_structure_penalty: float = 0.35
