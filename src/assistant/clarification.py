"""Fast, deterministic intake for diamond buying requests."""

import re

from .clarity import clarity_family
from .schemas import DiamondQueryPlan


BUYING_MARKERS = (
    "i want", "find", "recommend", "looking for", "budget", "buy", "give me",
    "good diamond", "best value", "how many", "below", "under", "carat",
)
CUT_GRADES = ("Fair", "Good", "Very Good", "Premium", "Ideal")
CLARITY_GRADES = ("I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF")
AMOUNT_PATTERN = r"\d[\d,]*(?:\.\d+)?(?:\s*k)?"


def _last_match(pattern: str, text: str) -> str | None:
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return matches[-1] if matches else None


def normalize_user_text(text: str) -> str:
    """Repair a small set of frequent buying-query spelling mistakes."""
    replacements = {
        r"\b(?:diamnd|dimond)\b": "diamond",
        r"\b(?:carrat|carret|carrot)\b": "carat",
        r"\b(?:claraty|clarty|clrty)\b": "clarity",
        r"\b(?:ideel|idel)\b": "ideal",
        r"\bcoler\b": "color",
        r"\bless then\b": "less than",
    }
    normalized = text
    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


def _amount(value: str) -> float:
    """Convert values such as ``3,000`` and ``4k`` to dollars."""
    cleaned = value.casefold().replace(",", "").replace(" ", "")
    multiplier = 1000 if cleaned.endswith("k") else 1
    if multiplier == 1000:
        cleaned = cleaned[:-1]
    return float(cleaned) * multiplier


def _last_budget(text: str) -> float | None:
    """Find the latest clearly price-like amount without mistaking carats for dollars."""
    patterns = (
        rf"\$\s*(?P<amount>{AMOUNT_PATTERN})",
        rf"(?P<amount>{AMOUNT_PATTERN})\s*\$",
        rf"\b(?:maximum\s+)?budget(?:\s+(?:is|of|for))?\s*\$?\s*(?P<amount>{AMOUNT_PATTERN})",
        rf"\b(?:under|below|less\s+than|no\s+more\s+than|up\s+to|max(?:imum)?)\s*\$?\s*(?P<amount>{AMOUNT_PATTERN})",
        r"\b(?:like|around|about)\s+\$?\s*(?P<amount>\d+(?:\.\d+)?\s*k)\b",
    )
    candidates = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = _amount(match.group("amount"))
            if value >= 100:
                candidates.append((match.start(), value))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _last_choice(text: str, choices: tuple[str, ...]) -> str | None:
    """Return the most recently stated category, preferring longer names."""
    matches = []
    for choice in sorted(choices, key=len, reverse=True):
        for match in re.finditer(rf"\b{re.escape(choice)}\b", text, flags=re.IGNORECASE):
            matches.append((match.start(), choice))
    return max(matches, default=(0, None), key=lambda item: item[0])[1]


def build_buying_plan(
    question: str,
    conversation_text: str,
    require_recommendation_details: bool = True,
) -> DiamondQueryPlan | None:
    """Return a verified buying plan, or None when the user is asking a factual question."""
    normalized_question = normalize_user_text(question)
    transcript = normalize_user_text(f"{conversation_text}\n{question}".strip())
    if not any(marker in transcript.casefold() for marker in BUYING_MARKERS):
        return None

    # Accept the ways people commonly type a budget: "$4,000", "4000$", or
    # "budget 4000". A number at the start of a short follow-up is also a
    # budget when the assistant just asked for one.
    budget = _last_budget(transcript)
    if re.search(r"maximum budget", conversation_text, flags=re.IGNORECASE):
        follow_up_budget = re.match(rf"\s*\$?\s*({AMOUNT_PATTERN})\b", normalized_question)
        if follow_up_budget:
            possible_budget = _amount(follow_up_budget.group(1))
            if possible_budget >= 100:
                budget = possible_budget

    carat_range = re.findall(
        r"\bbetween\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*(?:carats?|ct)\b",
        transcript,
        flags=re.IGNORECASE,
    )
    carat_tolerance = 0.15
    if carat_range:
        low, high = map(float, carat_range[-1])
        low, high = sorted((low, high))
        carat = round((low + high) / 2, 4)
        carat_tolerance = round((high - low) / 2, 4)
    else:
        carat_text = _last_match(r"\b(\d+(?:\.\d+)?)\s*(?:carats?|ct)\b", transcript)
        carat = float(carat_text) if carat_text else None
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

    cut = _last_choice(transcript, CUT_GRADES)
    clarity = _last_choice(transcript, CLARITY_GRADES)
    if clarity is None:
        clarity = _last_choice(transcript, ("VVS", "VS", "SI"))
    if clarity is not None:
        clarity = clarity_family(clarity)
    lowered = transcript.casefold()
    if clarity is None and re.search(r"\b(?:high|excellent|very good)\s+clarity\b", lowered):
        clarity = "VVS"
    elif clarity is None and (
        "balance between clarity and price" in lowered
        or "balance of clarity and price" in lowered
    ):
        clarity = "VS"
    color = _last_match(r"\bcolor(?:\s+grade)?\s+([D-J])\b", transcript)

    no_clarity_preference = bool(re.search(
        r"\b(?:don['’]?t\s+care(?:\s+that\s+much)?\s+about|remove|forget)\s+(?:the\s+)?clarity",
        normalized_question,
        flags=re.IGNORECASE,
    ))
    no_cut_preference = bool(re.search(
        r"\b(?:don['’]?t\s+care(?:\s+that\s+much)?\s+about|remove|forget)\s+(?:the\s+)?cut",
        normalized_question,
        flags=re.IGNORECASE,
    ))
    no_color_preference = bool(re.search(
        r"\b(?:don['’]?t\s+care(?:\s+that\s+much)?\s+about|remove|forget)\s+(?:the\s+)?color",
        normalized_question,
        flags=re.IGNORECASE,
    ))
    if no_clarity_preference:
        clarity = None
    if no_cut_preference:
        cut = None
    if no_color_preference:
        color = None

    priorities = []
    for priority in ("clarity", "cut", "size", "color", "value"):
        if priority in lowered and any(
            word in lowered for word in ("matter", "priorit", "prefer", "balance")
        ):
            priorities.append(priority)
    if "balance" in lowered and "price" in lowered and "value" not in priorities:
        priorities.append("value")
    if no_clarity_preference and "clarity" in priorities:
        priorities.remove("clarity")
    if no_cut_preference and "cut" in priorities:
        priorities.remove("cut")
    if no_color_preference and "color" in priorities:
        priorities.remove("color")

    missing = []
    if require_recommendation_details:
        # One concrete constraint is enough to start a useful search. The
        # assistant should not force buyers through a form-like interview when
        # they have already supplied a budget, size, or quality preference.
        has_search_constraint = any((budget, carat, cut, color, clarity))
        if not has_search_constraint:
            missing.append("your budget, preferred carat size, or quality preference")
        elif "clarity" in priorities and clarity is None and not no_clarity_preference:
            missing.append("the clarity grade or range you would accept")

    clarification = None
    if missing:
        clarification = "Before I search, what is " + ", and ".join(missing) + "?"

    min_price = None
    max_price = budget
    target_price = None
    if budget is not None and "around" in question.casefold() and "budget" not in question.casefold():
        min_price, max_price = budget * 0.9, budget * 1.1
        target_price = budget

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
        target_price=target_price,
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
