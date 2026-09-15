"""Choose the smallest useful set of evidence sources for each question."""

from .schemas import EvidenceRoute


def route_question(question: str, conversation_text: str = "") -> EvidenceRoute:
    """Classify user intent with auditable rules before retrieval or inference."""
    text = f"{conversation_text}\n{question}".casefold()
    current = question.casefold()

    if any(phrase in current for phrase in ("similar", "like this", "same quality")):
        return EvidenceRoute(
            "similarity_search", use_dataset=True, use_similarity=True,
            use_knowledge=True, use_models=True,
        )
    if any(phrase in current for phrase in ("what would this cost", "predict price", "price estimate")):
        return EvidenceRoute(
            "price_prediction", use_knowledge=True, use_models=True
        )
    if any(phrase in current for phrase in ("predict clarity", "classify clarity", "clarity family")):
        return EvidenceRoute(
            "clarity_prediction", use_knowledge=True, use_models=True
        )
    if any(phrase in current for phrase in ("compare", "difference", "trade-off", "tradeoff", "why")):
        return EvidenceRoute(
            "comparison", use_dataset=any(word in text for word in ("$", "budget", "carat")),
            use_knowledge=True, use_models="this diamond" in current,
        )
    if any(word in text for word in (
        "i want", "find", "recommend", "looking for", "budget", "buy", "show options", "$",
    )):
        return EvidenceRoute(
            "recommendation", use_dataset=True, use_knowledge=True, use_models=True
        )
    return EvidenceRoute("general_knowledge", use_knowledge=True)
