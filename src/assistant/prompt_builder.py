"""Compact routed evidence into the bounded prompt sent to local Qwen."""

from dataclasses import asdict
import json

import pandas as pd

from .clarity import clarity_family
from .schemas import DiamondQueryPlan, EvidenceRoute


VISIBLE_COLUMNS = [
    "_row_id",
    "price",
    "carat",
    "cut",
    "color",
    "clarity",
    "depth",
    "table",
    "x",
    "y",
    "z",
    "similarity_score",
    "model_price",
    "price_difference_pct",
    "predicted_clarity_family",
    "buyer_segment",
    "buyer_interpretation",
    "why_it_stands_out",
]


def compact_rows(frame: pd.DataFrame, limit: int = 5) -> list[dict]:
    """Keep only useful columns and a bounded number of dataset rows."""
    if frame.empty:
        return []
    columns = [column for column in VISIBLE_COLUMNS if column in frame.columns]
    visible = frame.loc[:, columns].head(limit).round(4).copy()
    if "clarity" in visible:
        visible["clarity"] = visible["clarity"].map(clarity_family)
    return visible.to_dict("records")


def compact_knowledge(details: list[dict], limit: int = 4) -> list[dict]:
    """Keep short facts, source names, and retrieval scores."""
    return [
        {
            "source": item.get("source", "unknown"),
            "section": item.get("section", "Overview"),
            "similarity": item.get("similarity"),
            "retrieval_score": item.get("retrieval_score"),
            "fact": str(item.get("document", ""))[:500],
        }
        for item in details[:limit]
    ]


def build_evidence_payload(
    route: EvidenceRoute,
    plan: DiamondQueryPlan,
    matches: pd.DataFrame,
    similar: pd.DataFrame,
    knowledge: list[dict],
    memory: list[str],
    model_evidence: dict | None = None,
) -> dict:
    """Build the internal JSON-sized evidence object shared by both UI modes."""
    return {
        "intent": route.intent,
        "filters": asdict(plan),
        "top_matches": compact_rows(matches),
        "similar_matches": compact_rows(similar),
        "rag_facts": compact_knowledge(knowledge),
        "model_evidence": model_evidence or {},
        "remembered_preferences": memory[:4],
    }


def build_generation_prompt(question: str, conversation_text: str, evidence: dict) -> str:
    """Separate observed, retrieved, predicted, and state evidence explicitly."""
    dataset_results = evidence.get("top_matches", [])
    dataset_statistics = evidence.get("dataset_statistics", {})
    rag_knowledge = evidence.get("rag_facts", [])
    model_outputs = evidence.get("model_evidence", {})
    current_state = evidence.get("filters", {})
    return (
        f"Recent conversation:\n{conversation_text}\n\n"
        f"Current question:\n{question}\n\n"
        "DATASET RESULTS\n"
        + json.dumps(dataset_results, ensure_ascii=False, separators=(",", ":"))
        + "\n\nDATASET STATISTICS\n"
        + json.dumps(dataset_statistics, ensure_ascii=False, separators=(",", ":"))
        + "\n\nRAG KNOWLEDGE\n"
        + json.dumps(rag_knowledge, ensure_ascii=False, separators=(",", ":"))
        + "\n\nMODEL OUTPUTS\n"
        + json.dumps(model_outputs, ensure_ascii=False, separators=(",", ":"))
        + "\n\nCURRENT SEARCH STATE\n"
        + json.dumps(current_state, ensure_ascii=False, separators=(",", ":"))
    )
