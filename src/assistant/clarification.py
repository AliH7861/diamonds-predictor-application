"""Fast, deterministic intake for diamond buying requests."""

import re

from .schemas import DiamondQueryPlan


BUYING_MARKERS = ("i want", "find", "recommend", "looking for", "budget", "buy")
CUT_GRADES = ("Fair", "Good", "Very Good", "Premium", "Ideal")
CLARITY_GRADES = ("I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF")


def _last_match(pattern: str, text: str) -> str | None:
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return matches[-1] if matches else None


def build_buying_plan(question: str, conversation_text: str) -> DiamondQueryPlan | None:
    """Return a verified buying plan, or None when the user is asking a factual question."""
    transcript = f"{conversation_text}\n{question}".strip()
    if not any(marker in transcript.casefold() for marker in BUYING_MARKERS):
        return None

    # Accept the ways people commonly type a budget: "$4,000", "4000$", or
    # "budget 4000". A number at the start of a short follow-up is also a
    # budget when the assistant just asked for one.
    money_text = _last_match(
        r"(?:\$\s*([\d,]+(?:\.\d+)?)|([\d,]+(?:\.\d+)?)\s*\$)", transcript
    )
    if isinstance(money_text, tuple):
        money_text = next((value for value in money_text if value), None)
    budget_text = _last_match(
        r"\b(?:maximum\s+)?budget(?:\s+(?:is|of|for))?\s*\$?\s*([\d,]+(?:\.\d+)?)",
        transcript,
    )
    if budget_text:
        money_text = budget_text
    if re.search(r"maximum budget", conversation_text, flags=re.IGNORECASE):
        follow_up_budget = re.match(r"\s*\$?\s*([\d,]{3,}(?:\.\d+)?)\b", question)
        if follow_up_budget:
            money_text = follow_up_budget.group(1)
    budget = float(money_text.replace(",", "")) if money_text else None

    # "Carrot" is a frequent speech-to-text or typing mistake for "carat".
    carat_text = _last_match(
        r"\b(\d+(?:\.\d+)?)\s*(?:carats?|carrots?|ct)\b", transcript
    )
    carat = float(carat_text) if carat_text else None
    carat_tolerance = 0.15
    if carat is None:
        lowered = transcript.casefold()
        if re.search(r"\bmedium(?:-sized|\s+size(?:d)?)?\b", lowered):
            carat, carat_tolerance = 0.75, 0.25
        elif re.search(r"\bsmall(?:-sized|\s+size(?:d)?)?\b", lowered):
            carat, carat_tolerance = 0.40, 0.15
        elif re.search(r"\blarge(?:-sized|\s+size(?:d)?)?\b", lowered):
            carat, carat_tolerance = 1.25, 0.25

    def stated_number(label: str) -> float | None:
        value = _last_match(
            rf"\b{label}\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)", transcript
        )
        return float(value) if value else None

    cut = None
    for grade in sorted(CUT_GRADES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(grade)}\b", transcript, flags=re.IGNORECASE):
            cut = grade
    clarity = None
    for grade in CLARITY_GRADES:
        if re.search(rf"\b{grade}\b", transcript, flags=re.IGNORECASE):
            clarity = grade
    lowered = transcript.casefold()
    if clarity is None and re.search(r"\b(?:high|excellent|very good)\s+clarity\b", lowered):
        clarity = "VVS"
    elif clarity is None and (
        "balance between clarity and price" in lowered
        or "balance of clarity and price" in lowered
    ):
        clarity = "VS"
    color = _last_match(r"\bcolor(?:\s+grade)?\s+([D-J])\b", transcript)

    priorities = []
    for priority in ("clarity", "cut", "size", "color", "value"):
        if priority in lowered and any(
            word in lowered for word in ("matter", "priorit", "prefer", "balance")
        ):
            priorities.append(priority)
    if "balance" in lowered and "price" in lowered and "value" not in priorities:
        priorities.append("value")

    missing = []
    if budget is None:
        missing.append("your maximum budget")
    if carat is None:
        missing.append("the carat size you are targeting")
    if "clarity" in priorities and clarity is None:
        missing.append("the clarity grade or range you would accept")
    elif cut is None and clarity is None:
        missing.append("a preferred cut or clarity grade")

    clarification = None
    if missing:
        clarification = "Before I search, what is " + ", and ".join(missing) + "?"

    min_price = None
    max_price = budget
    if budget is not None and "around" in question.casefold() and "budget" not in question.casefold():
        min_price, max_price = budget * 0.9, budget * 1.1

    query_parts = ["diamond buying trade-offs"]
    if priorities:
        query_parts.append("priorities " + " ".join(priorities))
    if clarity:
        query_parts.append(f"clarity {clarity}")
    if cut:
        query_parts.append(f"cut {cut}")
    knowledge_queries = [" ".join(query_parts), "carat price cut color clarity comparison"]

    return DiamondQueryPlan(
        search_dataset=True,
        min_price=min_price,
        max_price=max_price,
        target_carat=carat,
        carat_tolerance=carat_tolerance,
        depth=stated_number("depth"),
        table=stated_number("table"),
        x=stated_number("x"),
        y=stated_number("y"),
        z=stated_number("z"),
        cut=cut,
        color=color.upper() if color else None,
        clarity=clarity,
        priorities=priorities,
        needs_clarification=bool(missing),
        clarifying_question=clarification,
        knowledge_queries=[] if missing else knowledge_queries,
    )


def parse_model_inputs(text: str, intent: str) -> tuple[dict, list[str]]:
    """Extract raw ANN inputs from natural language and list anything still missing."""
    def number(label: str) -> float | None:
        found = _last_match(rf"\b(?:{label})\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)", text)
        return float(found) if found else None

    carat = _last_match(r"\b(\d+(?:\.\d+)?)\s*(?:carats?|ct)\b", text)
    values = {
        "carat": float(carat) if carat else number(r"carats?|ct"),
        "depth": number("depth"),
        "table": number("table"),
        "x": number("x"),
        "y": number("y"),
        "z": number("z"),
    }
    money = _last_match(r"\$\s*([\d,]+(?:\.\d+)?)", text)
    values["price"] = float(money.replace(",", "")) if money else number("price")
    for column, options in (("cut", CUT_GRADES), ("clarity", CLARITY_GRADES)):
        values[column] = next(
            (item for item in sorted(options, key=len, reverse=True)
             if re.search(rf"\b{re.escape(item)}\b", text, flags=re.IGNORECASE)),
            None,
        )
    color = _last_match(r"\bcolor(?:\s+grade)?\s+([D-J])\b", text)
    color = color or _last_match(r"\b([D-J])\s+color\b", text)
    values["color"] = color.upper() if color else None

    required = ["carat", "cut", "color", "depth", "table", "x", "y", "z"]
    required.append("clarity" if intent == "price_prediction" else "price")
    clean = {key: value for key, value in values.items() if value is not None}
    return clean, [key for key in required if key not in clean]
