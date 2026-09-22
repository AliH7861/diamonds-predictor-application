"""Typed models returned by the segmentation package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class SegmentProfile:
    """Human-readable profile for one discovered product segment."""

    cluster_id: int
    name: str
    segment_share_pct: float
    market_tier: str
    top_priorities: List[str]
    tradeoffs: List[str]
    pillar_scores: Dict[str, float]
    typical_diamond: Dict[str, Any]
    summary: str
    nearest_profile_id: Optional[int] = None
    most_distant_profile_id: Optional[int] = None
    limitation: str = (
        "Product-derived buyer-preference archetype; not validated customer psychology."
    )


@dataclass
class SegmentationResult:
    """Complete end-to-end result returned to the host application."""

    selected_solution: str
    selected_method: str
    selected_k: int
    enriched_diamonds: pd.DataFrame
    profiles: List[SegmentProfile]
    comparison: pd.DataFrame
    profile_distances: pd.DataFrame
    pillar_medians: pd.DataFrame
    anomaly_rows: pd.DataFrame
    rag_records: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
