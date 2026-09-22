"""Hybrid router for the assistant's seven product capabilities.

Natural language is interpreted semantically first. Deterministic rules remain as a
safe fallback so the product still works if the local LLM parser fails for a turn.
"""

from __future__ import annotations

import re
from typing import Any

from .clarification import normalize_user_text
from .schemas import EvidenceRoute


SMALL_TALK = {"hi", "hello", "hey", "thanks", "thank you", "got it", "help", "what can you do"}
OUT_OF_SCOPE = (
    "weather",
    "football",
    "basketball",
    "recipe",
    "stock price",
    "medical",
    "legal advice",
    "bitcoin",
    "movie",
    "hockey",
    "2+2",
    "2 + 2",
)
UNAVAILABLE = (
    "who certified",
    "country mined",
    "mined in",
    "type of inclusion",
    "buyer income",
    "sold online",
    "year sold",
)


def _route(
    intent: str,
    family: str,
    action: str,
    reason: str,
    *,
    use_dataset: bool = False,
    use_knowledge: bool = False,
    use_models: bool = False,
) -> EvidenceRoute:
    return EvidenceRoute(
        intent=intent,
        family=family,
        level_1=intent.upper(),
        level_2=action,
        level_3=action,
        confidence=0.99,
        source="semantic+deterministic",
        reason=reason,
        use_dataset=use_dataset,
        use_knowledge=use_knowledge,
        use_models=use_models,
        use_memory=False,
    )


def _active_search(state: dict[str, Any] | None) -> bool:
    state = state or {}
    if state.get("search_active"):
        return True
    return any(
        state.get(key) not in (None, [], "")
        for key in (
            "min_price",
            "max_price",
            "target_price",
            "target_carat",
            "cut",
            "color",
            "clarity",
            "priorities",
            "pending_field",
        )
    )


def _semantic_route(meaning: dict[str, Any] | None) -> EvidenceRoute | None:
    if not meaning:
        return None
    try:
        confidence = float(meaning.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < 0.45:
        return None

    intent = meaning.get("intent")
    reason = meaning.get("reason") or "semantic interpretation"

    if intent == "search":
        return _route("search", "data", "SEARCH", reason, use_dataset=True)
    if intent == "compare":
        return _route("compare", "data", "COMPARE", reason, use_dataset=True)
    if intent == "dataset_analysis":
        return _route(
            "dataset_analysis",
            "data",
            meaning.get("analysis_action") or "STATS",
            reason,
            use_dataset=True,
        )
    if intent == "diamond_knowledge":
        topic = str(meaning.get("knowledge_topic") or "").casefold()
        action = (
            "CUSTOMER_PROFILES"
            if topic in {"customer_profiles", "buyer_profiles", "clustering_profiles"}
            else "RAG_EXPLANATION"
        )
        return _route(
            "diamond_knowledge",
            "knowledge",
            action,
            reason,
            use_knowledge=True,
        )
    if intent == "model_prediction":
        targets = list(meaning.get("model_targets") or [])
        action = (
            "MULTI_MODEL"
            if len(targets) > 1
            else (f"{targets[0].upper()}_PREDICTION" if targets else "MODEL_PREDICTION")
        )
        return _route("model_prediction", "model", action, reason, use_models=True)
    if intent == "small_talk":
        return _route("small_talk", "conversation", "CHAT", reason)
    if intent == "out_of_scope":
        return _route("out_of_scope", "unsupported", "OUT_OF_SCOPE", reason)
    return None


def _analysis_route(text: str) -> EvidenceRoute | None:
    if re.search(
        r"\b(?:dataset|data)\b.{0,30}\b(?:overview|summary|features?|columns?|contain)\b|"
        r"\b(?:overview|summary)\b.{0,20}\b(?:dataset|data)\b|"
        r"\bwhat (?:features?|columns?) (?:are|is) (?:in|available)",
        text,
    ):
        return _route(
            "dataset_analysis",
            "data",
            "DATASET_OVERVIEW",
            "live dataset overview",
            use_dataset=True,
        )

    if not re.search(
        r"\b(?:average|avg|mean|median|mode|count|how many|range|largest|smallest|"
        r"highest|lowest|maximum|minimum|most common|distribution|correlation|correlates?|"
        r"relationship|standard deviation|percentile)\b",
        text,
    ):
        return None

    if re.search(r"\bby\s+(?:cut|color|clarity|carat band)\b", text):
        action = "TREND_GROUP"
    elif re.search(r"\b(?:correlation|correlates?|relationship)\b", text):
        action = "TREND_RELATION"
    elif re.search(r"\b(?:most common|distribution|mode)\b", text):
        action = "DISTRIBUTION"
    elif re.search(r"\b(?:largest|smallest|highest|lowest|maximum|minimum)\b", text):
        action = "EXTREME"
    elif "range" in text:
        action = "RANGE"
    else:
        action = "STATS"
    return _route(
        "dataset_analysis",
        "data",
        action,
        "explicit dataset calculation",
        use_dataset=True,
    )


def _fallback_route(
    question: str,
    conversation_text: str = "",
    state: dict[str, Any] | None = None,
) -> EvidenceRoute:
    del conversation_text  # structured state, not a text marker, controls continuity
    text = normalize_user_text(question).casefold().strip()
    plain = re.sub(r"[^\w\s+']", "", text).strip()
    active = _active_search(state)

    if plain in SMALL_TALK:
        return _route("small_talk", "conversation", "CHAT", "short conversational phrase")

    if any(term in text for term in OUT_OF_SCOPE):
        return _route("out_of_scope", "unsupported", "OUT_OF_SCOPE", "non-diamond request")

    if any(term in text for term in UNAVAILABLE):
        return _route(
            "out_of_scope", "unsupported", "UNAVAILABLE_FIELD", "field absent from dataset"
        )

    if re.search(r"\b(?:predict|prediction|classify|estimate)\b", text):
        targets: list[str] = []
        if "price" in text:
            targets.append("price")
        if "clarity" in text:
            targets.append("clarity")
        if re.search(r"\b(?:cluster|segment|profile)\b", text):
            targets.append("cluster")
        if targets:
            action = "MULTI_MODEL" if len(targets) > 1 else f"{targets[0].upper()}_PREDICTION"
            return _route(
                "model_prediction",
                "model",
                action,
                "explicit saved-model request: " + ", ".join(targets),
                use_models=True,
            )

    if re.search(
        r"\b(?:customer|buyer).{0,25}(?:types?|groups?|segments?|profiles?)\b|"
        r"\bhow many.{0,25}(?:customer|buyer).{0,20}(?:types?|groups?|segments?|profiles?)\b",
        text,
    ):
        return _route(
            "diamond_knowledge",
            "knowledge",
            "CUSTOMER_PROFILES",
            "customer profile question",
            use_knowledge=True,
        )

    # Explanations must be checked before generic phrases such as "I want".
    if (
        re.search(
            r"\b(?:what does|what is|why is|why does|how does|explain|define|meaning|"
            r"types? of|grades? of|different types? of|understand)\b",
            text,
        )
        and re.search(
            r"\b(?:diamond|diamonds|carat|cut|cuts|clarity|color|colours?|colors?|depth|"
            r"table|price per carat|cluster|clustering|quality)\b",
            text,
        )
        and not re.search(
            r"\b(?:average|avg|median|mode|range|count|how many|minimum|maximum|"
            r"largest|smallest|most common|distribution|correlation|standard deviation|percentile)\b|"
            r"\bmean\s+(?:price|carat|depth|table|x|y|z)\b",
            text,
        )
    ):
        return _route(
            "diamond_knowledge",
            "knowledge",
            "RAG_EXPLANATION",
            "diamond-domain explanation",
            use_knowledge=True,
        )

    analysis = _analysis_route(text)
    if analysis and not re.search(
        r"\b(?:find|show|recommend|give me|looking for|search for)\b", text
    ):
        return analysis

    if re.search(r"\bcompare(?: these| them| the)?\b|side by side", text):
        if re.search(r"\b(?:first|second|third|fourth|fifth|these|them|results?|options?)\b", text):
            return _route(
                "compare",
                "data",
                "COMPARE",
                "comparison of displayed dataframe rows",
                use_dataset=True,
            )

    if active and (
        re.fullmatch(r"\$?\s*[\d,.]+(?:\s*(?:ct|carat|carats?))?", text)
        or re.fullmatch(
            r"(?:price|budget|size|carat|cut|color|clarity|value|balance|fair|good|"
            r"very good|premium|ideal|i|si|vs|vvs|if|[d-j])",
            text,
        )
        or re.search(
            r"\b(?:different|another|cheaper|bigger|larger|smaller|more colorless|"
            r"maximum budget|max budget|under|below|at least|between|make it|make them|"
            r"focus on|important|matters)\b",
            text,
        )
    ):
        return _route("search", "data", "SEARCH", "active search refinement", use_dataset=True)

    strong_search = bool(
        re.search(
            r"\b(?:find|show|search|recommend|looking for|options?|matches?|"
            r"give me\s+\d+|give me\s+(?:one|two|three|four|five|six|seven|eight|nine|ten))\b",
            text,
        )
        or re.search(r"\bdiamonds?\b", text)
        and (
            "$" in text
            or re.search(
                r"\b(?:between|under|below|budget|range|carat|ct|clear|clean|bigger|cheaper)\b",
                text,
            )
        )
        or "$" in text
        and not re.search(r"\b(?:range|average|median|mean|minimum|maximum)\b", text)
        or re.search(r"\b\d*\.?\d+\s*(?:ct|carat|carats?)\b", text)
    )
    if strong_search:
        invalid = re.search(r"\b(?:clarity\s+)?(vi|iv|vsi|vvs3|vs3|si3)\b", text)
        if invalid:
            return _route(
                "search",
                "data",
                "INVALID_CONSTRAINT",
                f"unsupported clarity token: {invalid.group(1).upper()}",
                use_dataset=False,
            )
        return _route(
            "search", "data", "SEARCH", "natural-language dataset search", use_dataset=True
        )

    # Last diamond-domain catch-all is knowledge, not out-of-scope.
    if re.search(
        r"\b(?:clarity|carat|cut|color|depth|table|diamond|diamonds|price per carat|quality)\b",
        text,
    ):
        return _route(
            "diamond_knowledge",
            "knowledge",
            "RAG_EXPLANATION",
            "diamond-domain explanation",
            use_knowledge=True,
        )

    return _route(
        "out_of_scope",
        "unsupported",
        "OUT_OF_SCOPE",
        "request has no supported diamond capability",
    )


def route_question(
    question: str,
    conversation_text: str = "",
    *,
    state: dict[str, Any] | None = None,
    understanding: dict[str, Any] | None = None,
) -> EvidenceRoute:
    """Choose one capability using semantic meaning first and safe rules second."""
    semantic = _semantic_route(understanding)
    if semantic is not None:
        return semantic
    return _fallback_route(question, conversation_text, state)
