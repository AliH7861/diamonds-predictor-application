"""RAG-facing helpers for customer-preference profiles.

The LLM should explain structured segmentation results, not invent them.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, List, Sequence

import pandas as pd

from .models import SegmentProfile


def build_rag_records(
    profiles: Sequence[SegmentProfile],
) -> List[Dict[str, Any]]:
    """Convert structured profiles into authoritative RAG-ready records."""
    records: List[Dict[str, Any]] = []

    for profile in profiles:
        priorities = ", ".join(profile.top_priorities)
        tradeoffs = ", ".join(profile.tradeoffs)
        typical = profile.typical_diamond

        text = (
            f"Customer profile: {profile.name}. "
            f"Segment share: {profile.segment_share_pct:.2f}%. "
            f"Market tier: {profile.market_tier}. "
            f"Likely priorities: {priorities}. "
            f"Likely tradeoffs: {tradeoffs}. "
            f"Typical diamond: approximately {typical['carat']:.2f} carats, "
            f"${typical['price']:.0f}, {typical['cut']} cut, "
            f"{typical['color']} color, {typical['clarity']} clarity. "
            f"Interpretation: {profile.summary} "
            f"Limitation: {profile.limitation}"
        )

        records.append(
            {
                "id": f"customer_profile_{profile.cluster_id}",
                "type": "customer_profile",
                "cluster_id": profile.cluster_id,
                "profile_name": profile.name,
                "segment_share_pct": profile.segment_share_pct,
                "market_tier": profile.market_tier,
                "top_priorities": profile.top_priorities,
                "tradeoffs": profile.tradeoffs,
                "pillar_scores": profile.pillar_scores,
                "typical_diamond": profile.typical_diamond,
                "nearest_profile_id": profile.nearest_profile_id,
                "most_distant_profile_id": profile.most_distant_profile_id,
                "text": text,
            }
        )

    return records


def profile_lookup(
    profiles: Sequence[SegmentProfile],
    profile_name: str,
) -> SegmentProfile | None:
    """Return an exact case-insensitive profile match for structured lookup."""
    target = profile_name.strip().casefold()

    for profile in profiles:
        if profile.name.casefold() == target:
            return profile

    return None


def filter_diamonds_by_profile(
    enriched_diamonds: pd.DataFrame,
    profile_name: str,
) -> pd.DataFrame:
    """Return diamonds assigned to the requested profile."""
    if "customer_profile_name" not in enriched_diamonds.columns:
        raise ValueError("Expected 'customer_profile_name' in enriched diamonds.")

    target = profile_name.strip().casefold()
    mask = enriched_diamonds["customer_profile_name"].astype(str).str.casefold() == target
    return enriched_diamonds.loc[mask].copy()


def build_profile_context(
    profile: SegmentProfile,
) -> Dict[str, Any]:
    """Return structured context suitable for an existing assistant router."""
    return {
        "intent": "customer_profile",
        "cluster_id": profile.cluster_id,
        "profile_name": profile.name,
        "segment_share_pct": profile.segment_share_pct,
        "market_tier": profile.market_tier,
        "top_priorities": profile.top_priorities,
        "tradeoffs": profile.tradeoffs,
        "pillar_scores": profile.pillar_scores,
        "typical_diamond": profile.typical_diamond,
        "summary": profile.summary,
        "limitation": profile.limitation,
    }


def load_profile_registry(path: str | Path) -> List[SegmentProfile]:
    """Load the authoritative JSON profile registry created during training."""
    registry_path = Path(path)
    if not registry_path.exists():
        return []
    records = json.loads(registry_path.read_text(encoding="utf-8"))
    return [SegmentProfile(**record) for record in records]


def answer_profile_question(question: str, profiles: Sequence[SegmentProfile]) -> str | None:
    """Answer exact and comparative profile questions from structured data."""
    if not profiles:
        return None
    lowered = question.casefold()
    exact = next(
        (profile for profile in profiles if profile.name.casefold() in lowered),
        None,
    )
    pillar_terms = {
        "quality": "quality",
        "visual size": "visualpresence",
        "visible size": "visualpresence",
        "value": "value",
        "budget": "value",
        "rarity": "rarity",
        "milestone": "milestone",
        "proportion": "proportion",
        "balance": "qualitybalance",
    }
    requested_pillar = next(
        (pillar for phrase, pillar in pillar_terms.items() if phrase in lowered), None
    )
    if (
        exact is None
        and requested_pillar
        and any(word in lowered for word in ("profile", "segment", "buyer"))
    ):
        exact = max(
            profiles,
            key=lambda profile: profile.pillar_scores.get(requested_pillar, float("-inf")),
        )
    if exact is None:
        return None

    typical = exact.typical_diamond
    priorities = ", ".join(exact.top_priorities)
    tradeoffs = ", ".join(exact.tradeoffs)
    return (
        f"{exact.name} represents {exact.segment_share_pct:.2f}% of segmented diamonds. "
        f"It is a {exact.market_tier} product profile that emphasizes {priorities}. "
        f"Its main trade-offs are {tradeoffs}. A typical diamond is about "
        f"{typical['carat']:.2f} carats and ${typical['price']:,.0f}, with "
        f"{typical['cut']} cut, {typical['color']} color, and {typical['clarity']} clarity. "
        f"{exact.summary} {exact.limitation}"
    )
