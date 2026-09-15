"""Choose the smallest useful set of evidence sources for each question."""

from .schemas import EvidenceRoute
from .clarification import normalize_user_text


def route_question(question: str, conversation_text: str = "") -> EvidenceRoute:
    """Classify user intent with auditable rules before retrieval or inference."""
    text = normalize_user_text(f"{conversation_text}\n{question}").casefold()
    current = normalize_user_text(question).casefold()

    unavailable_fields = (
        "who certified", "country was", "country mined", "mined in",
        "type of inclusion", "married customer", "buyer's income", "buyer income",
        "sold online", "sold in store", "year was", "year sold",
    )
    if any(phrase in current for phrase in unavailable_fields):
        return EvidenceRoute("unsupported_dataset_field", use_memory=False)
    if current.startswith("how many") or "count diamonds" in current:
        return EvidenceRoute("dataset_count", use_dataset=True, use_memory=False)
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
    buying_terms = (
        "i want", "find", "recommend", "looking for", "budget", "buy", "show options", "$",
        "good diamond", "best value", "give me",
    )
    follow_up_terms = (
        "same", "actually", "make it", "under", "below", "no more than",
        "carat", "cut", "clarity", "color", "don't care", "dont care", "remove",
    )
    continues_search = (
        "saved search preferences:" in text
        and any(term in current for term in follow_up_terms)
    )
    if any(word in current for word in buying_terms) or continues_search:
        needs_explanation = any(
            word in current
            for word in ("explain", "why", "trade-off", "tradeoff", "priorit")
        )
        return EvidenceRoute(
            "recommendation",
            use_dataset=True,
            use_knowledge=needs_explanation,
            use_models=needs_explanation,
            use_memory=needs_explanation,
        )
    return EvidenceRoute("general_knowledge", use_knowledge=True)
