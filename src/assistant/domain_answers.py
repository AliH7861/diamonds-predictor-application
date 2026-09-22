"""Short deterministic answers for fixed project vocabularies."""

from __future__ import annotations

from typing import Sequence

from src.clustering.models import SegmentProfile


CUT_ANSWER = (
    "The dataset has five cut grades: **Fair, Good, Very Good, Premium, and Ideal**. "
    "Cut describes how well the diamond's proportions support brightness and sparkle; "
    "it is different from diamond shape."
)

CLARITY_ANSWER = (
    "This project uses five clarity families: **I, SI, VS, VVS, and IF**. "
    "They progress from more visible inclusions toward internally flawless diamonds."
)

COLOR_ANSWER = (
    "The dataset contains color grades **D through J**. D is the most colorless in this "
    "range, while grades closer to J show more visible warmth."
)


def answer_domain_catalog(
    topic: str,
    profiles: Sequence[SegmentProfile],
) -> str | None:
    """Return a concise authoritative list without retrieval or generation."""
    if topic == "cut":
        return CUT_ANSWER
    if topic == "clarity":
        return CLARITY_ANSWER
    if topic == "color":
        return COLOR_ANSWER
    if topic == "customer_profiles" and profiles:
        names = [profile.name for profile in profiles]
        return (
            "The clustering model found five purchase profiles: **"
            + "**, **".join(names)
            + "**. These describe groups of similar diamonds and possible buying priorities; "
            "they are not verified customer identities or demographics."
        )
    return None
