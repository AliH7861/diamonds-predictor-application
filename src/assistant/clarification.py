"""Fast, deterministic intake for diamond buying requests."""

import json
import re
from typing import Any

from .clarity import clarity_family
from .schemas import DiamondQueryPlan


BUYING_MARKERS = (
    "i want",
    "find",
    "recommend",
    "looking for",
    "budget",
    "buy",
    "give me",
    "good diamond",
    "best value",
    "how many",
    "below",
    "under",
    "carat",
)
CUT_GRADES = ("Fair", "Good", "Very Good", "Premium", "Ideal")
CLARITY_GRADES = ("I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF")
CLARITY_FAMILY_DEFAULTS = {
    "I": "I1",
    "SI": "SI1",
    "VS": "VS2",
    "VVS": "VVS2",
    "IF": "IF",
}
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
        r"\bmedain\b": "median",
        r"\bprce\b": "price",
        r"\bclrity\b": "clarity",
        r"\bcolr\b": "color",
        r"\bwhats\b": "what is",
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
        rf"\b(?P<amount>{AMOUNT_PATTERN})\s+(?:max|maximum|ceiling|cap)\b",
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


def _last_cut(text: str) -> str | None:
    """Extract a cut grade without mistaking phrases such as "good value" for a grade."""
    choices = "Very Good|Premium|Ideal|Fair|Good"
    explicit = re.findall(
        rf"\b({choices})\s+cut\b|\bcut\s+(?:grade\s+)?({choices})\b",
        text,
        flags=re.IGNORECASE,
    )
    if explicit:
        value = explicit[-1][0] or explicit[-1][1]
        return next(item for item in CUT_GRADES if item.casefold() == value.casefold())
    # Bare premium/ideal/fair labels are normally grades. Bare "good" is too
    # ambiguous in natural language and is accepted only beside the word cut.
    return _last_choice(text, ("Very Good", "Premium", "Ideal", "Fair"))


def _last_clarity(text: str) -> str | None:
    """Extract a detailed grade or one of the five buyer-facing families."""
    grade = _last_choice(text, CLARITY_GRADES)
    if grade is None:
        grade = _last_choice(text, ("VVS", "VS", "SI"))
    if grade is None and text.strip(" .,!?").casefold() in {"i", "i clarity", "clarity i"}:
        grade = "I"
    return clarity_family(grade) if grade is not None else None


def _last_color(text: str) -> str | None:
    """Extract an explicit D-J color reference from one piece of text."""
    color = _last_match(r"\bcolor(?:\s+grade)?\s+([D-J])\b", text)
    color = color or _last_match(r"\b([D-J])\s+color\b", text)
    color = color or _last_match(r"\b([D-J])(?:\+|\s+or\s+better\b)", text)
    color = color or _last_match(r"(?:^|[/,;]\s*)([D-J])(?=\s*(?:[/,;]|$))", text)
    return color.upper() if color else None


def _carat_request(text: str) -> tuple[float | None, float]:
    """Read the latest explicit carat target or range from one piece of text."""
    candidates: list[tuple[int, float, float]] = []
    range_pattern = (
        r"(?:\bbetween\s+)?(\d*\.\d+|\d+(?:\.\d+)?)\s*(?:-|to|and)\s*"
        r"(\d*\.\d+|\d+(?:\.\d+)?)\s*(?:carats?|ct)\b"
    )
    range_spans = []
    for match in re.finditer(range_pattern, text, flags=re.IGNORECASE):
        low, high = sorted((float(match.group(1)), float(match.group(2))))
        candidates.append((match.start(), round((low + high) / 2, 4), round((high - low) / 2, 4)))
        range_spans.append(match.span())
    single_pattern = r"(?<![\w.])(\d*\.\d+|\d+(?:\.\d+)?)\s*(?:carats?|ct)\b"
    for match in re.finditer(single_pattern, text, flags=re.IGNORECASE):
        if any(start <= match.start() and match.end() <= end for start, end in range_spans):
            continue
        candidates.append((match.start(), float(match.group(1)), 0.15))
    if not candidates:
        return None, 0.15
    _, target, tolerance = max(candidates, key=lambda item: item[0])
    return target, tolerance


def build_buying_plan(
    question: str,
    conversation_text: str,
    require_recommendation_details: bool = True,
) -> DiamondQueryPlan | None:
    """Return a verified buying plan, or None when the user is asking a factual question."""
    normalized_question = normalize_user_text(question)
    transcript = normalize_user_text(f"{conversation_text}\n{question}".strip())
    if require_recommendation_details and not any(
        marker in transcript.casefold() for marker in BUYING_MARKERS
    ):
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

    # Current-turn evidence wins over previous chat text. This prevents numbers
    # quoted in an earlier assistant answer from overriding a correction such as
    # "Make it 0.75 carat."
    carat, carat_tolerance = _carat_request(normalized_question)
    if carat is None:
        carat, carat_tolerance = _carat_request(transcript)
    if carat is None:
        lowered = transcript.casefold()
        if re.search(r"\bmedium(?:-sized|\s+size(?:d)?)?\b", lowered):
            carat, carat_tolerance = 0.75, 0.25
        elif re.search(r"\bsmall(?:-sized|\s+size(?:d)?)?\b", lowered):
            carat, carat_tolerance = 0.40, 0.15
        elif re.search(r"\blarge(?:-sized|\s+size(?:d)?)?\b", lowered):
            carat, carat_tolerance = 1.25, 0.25

    def stated_number(label: str) -> float | None:
        value = _last_match(rf"\b{label}\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)", transcript)
        return float(value) if value else None

    cut = _last_cut(normalized_question) or _last_cut(transcript)
    clarity = _last_clarity(normalized_question) or _last_clarity(transcript)
    lowered = transcript.casefold()
    if clarity is None and re.search(r"\b(?:high|excellent|very good)\s+clarity\b", lowered):
        clarity = "VVS"
    elif clarity is None and (
        "balance between clarity and price" in lowered or "balance of clarity and price" in lowered
    ):
        clarity = "VS"
    color = _last_color(normalized_question) or _last_color(transcript)

    no_clarity_preference = bool(
        re.search(
            r"\b(?:don['’]?t\s+care(?:\s+that\s+much)?\s+about|remove|forget)\s+(?:the\s+)?clarity",
            normalized_question,
            flags=re.IGNORECASE,
        )
    )
    no_cut_preference = bool(
        re.search(
            r"\b(?:don['’]?t\s+care(?:\s+that\s+much)?\s+about|remove|forget)\s+(?:the\s+)?cut",
            normalized_question,
            flags=re.IGNORECASE,
        )
    )
    no_color_preference = bool(
        re.search(
            r"\b(?:don['’]?t\s+care(?:\s+that\s+much)?\s+about|remove|forget)\s+(?:the\s+)?color",
            normalized_question,
            flags=re.IGNORECASE,
        )
    )
    if no_clarity_preference:
        clarity = None
    if no_cut_preference:
        cut = None
    if no_color_preference:
        color = None

    priority_mentions = []
    for priority in ("clarity", "cut", "size", "color", "value", "balance"):
        patterns = (
            rf"\b{priority}\b.{{0,24}}\b(?:matters?|priority|prioritize|preferred?|first|main|top)\b",
            rf"\b(?:prioritize|preferred?|main|top)\b.{{0,24}}\b{priority}\b",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, lowered):
                priority_mentions.append((match.start(), priority))
    priorities = [max(priority_mentions)[1]] if priority_mentions else []
    if "balance" in lowered and "price" in lowered and "value" not in priorities:
        priorities.append("value")
    current_lower = normalized_question.casefold()
    current_priorities = [
        priority
        for priority in ("clarity", "cut", "size", "color", "value", "balance")
        if re.search(
            rf"\b{priority}\b.{{0,24}}\b(?:matters?|priority|prioritize|first|main|top)\b|"
            rf"\b(?:prioritize|main|top)\b.{{0,24}}\b{priority}\b",
            current_lower,
        )
    ]
    if current_priorities:
        priorities = [current_priorities[-1]]
    if re.search(r"\bbigger (?:stone|diamond)|\bsize comes first\b", current_lower):
        priorities = ["size"]
    if re.search(r"\bclean(?: look|liness)? matters\b", current_lower):
        priorities = ["clarity"]
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
    approximate_price = re.search(
        rf"\b(?:around|about)\s+\$\s*(?:{AMOUNT_PATTERN})\b|"
        rf"\b(?:around|about)\s+(?:{AMOUNT_PATTERN})\s+dollars?\b|"
        rf"\b(?:price|cost)\s+(?:is\s+)?(?:around|about)\s+(?:{AMOUNT_PATTERN})\b|"
        rf"\b(?:around|about)\s+(?:the\s+)?(?:price|cost)\s+\$?\s*(?:{AMOUNT_PATTERN})\b",
        question,
        flags=re.IGNORECASE,
    )
    if budget is not None and approximate_price and "budget" not in question.casefold():
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
        color=color,
        clarity=clarity,
        priorities=priorities,
        needs_clarification=bool(missing),
        clarifying_question=clarification,
        knowledge_queries=[] if missing else knowledge_queries,
    )


def parse_model_inputs(text: str, intent: str) -> tuple[dict[str, float | str], list[str]]:
    """Extract raw ANN inputs from natural language and list anything still missing."""

    def number(label: str) -> float | None:
        found = _last_match(rf"\b(?:{label})\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)", text)
        return float(found) if found else None

    carat = _last_match(r"\b(\d+(?:\.\d+)?)\s*(?:carats?|ct)\b", text)
    values: dict[str, float | str | None] = {
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
            (
                item
                for item in sorted(options, key=len, reverse=True)
                if re.search(rf"\b{re.escape(item)}\b", text, flags=re.IGNORECASE)
            ),
            None,
        )
    if values["clarity"] is None:
        family = _last_choice(text, tuple(CLARITY_FAMILY_DEFAULTS))
        values["clarity"] = CLARITY_FAMILY_DEFAULTS.get(family) if family else None
    color = _last_match(r"\bcolor(?:\s+grade)?\s+([D-J])\b", text)
    color = color or _last_match(r"\b([D-J])\s+color\b", text)
    # Natural diamond descriptions commonly omit the word "color", for
    # example "1 ct Ideal G VS2". A standalone D-J token is safe here because
    # model inference already requires the other physical diamond fields.
    color = color or _last_match(r"\b([D-J])\b", text)
    values["color"] = color.upper() if color else None

    required = ["carat", "cut", "color", "depth", "table", "x", "y", "z"]
    if intent in {"price_prediction", "regression"}:
        required.append("clarity")
    elif intent in {"cluster_prediction", "clustering"}:
        required.extend(["clarity", "price"])
    clean = {key: value for key, value in values.items() if value is not None}
    return clean, [key for key in required if key not in clean]


# -----------------------------------------------------------------------------
# Semantic understanding for free-form, multi-turn chat
# -----------------------------------------------------------------------------

SUPPORTED_INTENTS = {
    "search",
    "compare",
    "dataset_analysis",
    "diamond_knowledge",
    "model_prediction",
    "small_talk",
    "out_of_scope",
}
SEMANTIC_SEARCH_FIELDS = {
    "min_price",
    "max_price",
    "target_price",
    "target_carat",
    "carat_tolerance",
    "cut",
    "color",
    "clarity",
    "priorities",
    "pending_field",
}
ANALYSIS_ACTIONS = {
    "DATASET_OVERVIEW",
    "RANGE",
    "STATS",
    "DISTRIBUTION",
    "EXTREME",
    "TREND_RELATION",
    "TREND_GROUP",
}


def _semantic_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    clean = value.casefold().replace("$", "").replace(",", "").replace(" ", "")
    multiplier = 1000.0 if clean.endswith("k") else 1.0
    clean = clean.removesuffix("k")
    try:
        return float(clean) * multiplier
    except ValueError:
        return None


def _json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object from common local-LLM response formats."""
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.I | re.S).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S).strip()
    decoder = json.JSONDecoder()
    for start, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("No JSON object found in semantic-parser response.")


def _clean_semantic(raw: dict[str, Any]) -> dict[str, Any]:
    intent = str(raw.get("intent", "")).strip().casefold()
    if intent not in SUPPORTED_INTENTS:
        intent = "out_of_scope"
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    incoming = raw.get("search_updates")
    incoming = incoming if isinstance(incoming, dict) else {}
    updates: dict[str, Any] = {}
    for field in SEMANTIC_SEARCH_FIELDS:
        if field not in incoming:
            continue
        value = incoming[field]
        if field in {"min_price", "max_price", "target_price", "target_carat", "carat_tolerance"}:
            number = _semantic_number(value)
            if number is not None and number >= 0:
                updates[field] = number
        elif field == "priorities":
            values = [value] if isinstance(value, str) else value
            if isinstance(values, list):
                allowed = {"price", "size", "cut", "color", "clarity", "value", "balance"}
                updates[field] = [
                    str(item).casefold() for item in values if str(item).casefold() in allowed
                ]
        elif field == "cut" and isinstance(value, str):
            match = next((item for item in CUT_GRADES if item.casefold() == value.casefold()), None)
            if match:
                updates[field] = match
        elif field == "color" and isinstance(value, str) and value.upper() in set("DEFGHIJ"):
            updates[field] = value.upper()
        elif field == "clarity" and isinstance(value, str):
            family = value.upper()
            if family in {"I", "SI", "VS", "VVS", "IF"}:
                updates[field] = family
        elif field == "pending_field" and value in {
            None,
            "price",
            "carat",
            "cut",
            "color",
            "clarity",
            "priority",
        }:
            updates[field] = value

    relative = raw.get("relative_change")
    if not isinstance(relative, dict):
        relative = {}
    relative = {
        str(key).casefold(): str(value).casefold()
        for key, value in relative.items()
        if str(key).casefold() in {"price", "carat", "cut", "color", "clarity"}
        and str(value).casefold() in {"increase", "decrease", "clear", "keep"}
    }

    model_targets = raw.get("model_targets")
    if isinstance(model_targets, str):
        model_targets = [model_targets]
    if not isinstance(model_targets, list):
        model_targets = []
    model_targets = [
        str(value).casefold()
        for value in model_targets
        if str(value).casefold() in {"price", "clarity", "cluster"}
    ]

    result_count = raw.get("result_count")
    try:
        result_count = int(result_count) if result_count is not None else None
    except (TypeError, ValueError):
        result_count = None
    if result_count is not None and not 1 <= result_count <= 20:
        result_count = None

    action = str(raw.get("analysis_action") or "").upper() or None
    if action not in ANALYSIS_ACTIONS:
        action = None

    return {
        "intent": intent,
        "confidence": confidence,
        "reason": str(raw.get("reason", ""))[:300],
        "search_updates": updates,
        "relative_change": relative,
        "result_count": result_count,
        "compare_displayed": bool(raw.get("compare_displayed", False)),
        "analysis_action": action,
        "knowledge_topic": str(raw.get("knowledge_topic", "")).strip().casefold()[:100],
        "knowledge_query": str(raw.get("knowledge_query", "")).strip()[:500],
        "model_targets": model_targets,
        "needs_clarification": bool(raw.get("needs_clarification", False)),
        "clarifying_question": str(raw.get("clarifying_question", "")).strip()[:300],
        "preserve_search_state": bool(raw.get("preserve_search_state", True)),
    }


def _deterministic_semantic_hints(
    question: str,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reliable facts from the current turn. These override LLM guesses."""
    text = normalize_user_text(question).strip()
    lower = text.casefold().replace("’", "'")
    state = dict(state or {})
    updates: dict[str, Any] = {}
    priorities: list[str] = []

    # Result count: "give me 3 diamonds", "show three options".
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    count_match = re.search(
        r"\b(?:find|show|give|recommend|return)?\s*(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:different\s+)?(?:diamonds?|options?|matches?)\b",
        lower,
    )
    result_count = None
    if count_match:
        token = count_match.group(1)
        result_count = int(token) if token.isdigit() else words[token]
        if not 1 <= result_count <= 20:
            result_count = None

    # Price range accepts natural forms with or without dollar signs.
    price_range = re.search(
        rf"\b(?:between|from|range\s+(?:of\s+)?)\s*\$?\s*({AMOUNT_PATTERN})\s*"
        rf"(?:and|to|-)\s*\$?\s*({AMOUNT_PATTERN})\b",
        lower,
    )
    if price_range:
        low, high = sorted((_amount(price_range.group(1)), _amount(price_range.group(2))))
        if low >= 100 and high >= 100:
            updates.update(min_price=low, max_price=high, target_price=None)
    else:
        budget = _last_budget(text)
        if budget is None:
            # "a diamond for $2000" and a lone "$2000" are valid search budgets.
            money = re.findall(rf"\$\s*({AMOUNT_PATTERN})", lower)
            if money:
                possible = _amount(money[-1])
                if possible >= 100:
                    budget = possible
        if budget is not None:
            updates["max_price"] = budget
            updates["target_price"] = None

    carat, tolerance = _carat_request(text)
    if carat is not None:
        updates["target_carat"] = carat
        updates["carat_tolerance"] = tolerance

    cut = _last_cut(text)
    clarity = _last_clarity(text)
    color = _last_color(text)
    if cut:
        updates["cut"] = cut
    if clarity:
        updates["clarity"] = clarity
    if color:
        updates["color"] = color

    # Explicit removals.
    for field in ("clarity", "cut", "color"):
        if re.search(
            rf"\b(?:forget|remove|clear|drop)\s+(?:the\s+)?{field}\b|"
            rf"\bdon't care(?:\s+that much)?\s+about\s+(?:the\s+)?{field}\b",
            lower,
        ):
            updates[field] = None

    # Natural preference language is ranking, not a hard grade filter.
    preference_phrases = {
        "clarity": (
            r"\bclarity\b.{0,35}\b(?:important|matters?|priority|focus|first|most)\b|"
            r"\b(?:important|care|focus|priorit(?:y|ize)|most)\b.{0,35}\bclarity\b|"
            r"\b(?:clear|clean[- ]?looking|cleanliness)\s+diamond\b"
        ),
        "size": (
            r"\bsize\b.{0,35}\b(?:important|matters?|priority|focus|first|most)\b|"
            r"\b(?:important|care|focus|priorit(?:y|ize)|most)\b.{0,35}\bsize\b|"
            r"\b(?:bigger|larger)\s+(?:stone|diamond)\b"
        ),
        "cut": r"\bcut\b.{0,35}\b(?:important|matters?|priority|focus|first|most)\b|\b(?:focus|priorit(?:y|ize))\b.{0,35}\bcut\b",
        "color": r"\bcolor\b.{0,35}\b(?:important|matters?|priority|focus|first|most)\b|\b(?:focus|priorit(?:y|ize))\b.{0,35}\bcolor\b",
        "price": r"\bprice\b.{0,35}\b(?:important|matters?|priority|focus|first|most)\b|\b(?:focus|priorit(?:y|ize))\b.{0,35}\bprice\b",
        "value": r"\b(?:best\s+value|value\s+matters?|prioritize\s+value)\b",
        "balance": r"\b(?:balanced?|trade[- ]?off)\b",
    }
    for priority, pattern in preference_phrases.items():
        if re.search(pattern, lower):
            priorities.append(priority)

    # A one-word answer is deterministic when the previous turn asked for priority.
    plain = re.sub(r"[^a-z]+", " ", lower).strip()
    if state.get("pending_field") == "priority" and plain in {
        "price",
        "size",
        "cut",
        "color",
        "clarity",
        "value",
        "balance",
    }:
        priorities = [plain]
        updates["pending_field"] = None
    elif state.get("pending_field") == "price":
        numeric = re.fullmatch(r"\s*\$?\s*([\d,]+(?:\.\d+)?(?:\s*k)?)\s*", lower)
        if numeric:
            possible = _amount(numeric.group(1))
            if possible >= 100:
                updates["max_price"] = possible
                updates["pending_field"] = None
    elif state.get("pending_field") == "carat":
        numeric = re.fullmatch(r"\s*(\d*\.\d+|\d+(?:\.\d+)?)\s*", lower)
        if numeric:
            updates["target_carat"] = float(numeric.group(1))
            updates["carat_tolerance"] = 0.10
            updates["pending_field"] = None

    if priorities:
        updates["priorities"] = [priorities[-1]]

    relative: dict[str, str] = {}
    if re.search(r"\b(?:cheaper|less expensive|lower price)\b", lower):
        relative["price"] = "decrease"
    if re.search(r"\b(?:bigger|larger|more carat)\b", lower):
        relative["carat"] = "increase"
    if re.search(r"\b(?:smaller|less carat)\b", lower):
        relative["carat"] = "decrease"

    # Strong deterministic intent hints. The LLM can fill the conversational gaps.
    intent = ""
    confidence = 0.0
    reason = ""
    if re.fullmatch(r"(?:hi|hello|hey|thanks|thank you|got it)", lower.strip(" .!?")):
        intent, confidence, reason = "small_talk", 0.99, "short conversational phrase"
    elif re.search(r"\b(?:compare|versus|vs\.?|side by side)\b", lower) and re.search(
        r"\b(?:first|second|third|fourth|fifth|these|them|results?|options?)\b", lower
    ):
        intent, confidence, reason = "compare", 0.98, "comparison of displayed results"
    elif (
        re.search(
            r"\b(?:what does|what is|why is|why does|explain|define|meaning|types? of|grades? of|"
            r"different types? of|how does)\b",
            lower,
        )
        and re.search(
            r"\b(?:diamond|carat|cut|clarity|color|colour|depth|table|price per carat|cluster|clustering)\b",
            lower,
        )
        and not re.search(
            r"\b(?:average|avg|median|mode|range|count|how many|minimum|maximum|"
            r"largest|smallest|most common|distribution|correlation|standard deviation|percentile)\b|"
            r"\bmean\s+(?:price|carat|depth|table|x|y|z)\b",
            lower,
        )
    ):
        intent, confidence, reason = "diamond_knowledge", 0.96, "diamond concept explanation"
    elif re.search(r"\b(?:predict|prediction|classify|estimate)\b", lower) and re.search(
        r"\b(?:price|clarity|cluster|segment)\b", lower
    ):
        intent, confidence, reason = "model_prediction", 0.96, "saved-model request"
    elif (
        updates
        or result_count is not None
        or relative
        or re.search(
            r"\b(?:find|show|search|recommend|looking for|give me|options?|matches?)\b", lower
        )
    ):
        intent, confidence, reason = "search", 0.88, "diamond search or refinement"
    elif re.search(
        r"\b(?:average|avg|median|mode|range|count|how many|minimum|maximum|"
        r"largest|smallest|most common|distribution|correlation|standard deviation|percentile)\b|"
        r"\bmean\s+(?:price|carat|depth|table|x|y|z)\b",
        lower,
    ) and not re.search(r"\btypes?\b", lower):
        intent, confidence, reason = "dataset_analysis", 0.92, "dataset calculation request"

    model_targets = []
    if intent == "model_prediction":
        if "price" in lower:
            model_targets.append("price")
        if "clarity" in lower:
            model_targets.append("clarity")
        if re.search(r"\b(?:cluster|segment)\b", lower):
            model_targets.append("cluster")

    knowledge_topic = ""
    if intent == "diamond_knowledge":
        for topic in ("clarity", "cut", "color", "carat", "depth", "table"):
            if re.search(rf"\b{topic}\b", lower):
                knowledge_topic = topic
                break
        if re.search(r"\b(?:customer|buyer).{0,25}(?:segment|profile|type|group)", lower):
            knowledge_topic = "customer_profiles"

    action = None
    if intent == "dataset_analysis":
        if "range" in lower:
            action = "RANGE"
        elif "distribution" in lower or "mode" in lower or "most common" in lower:
            action = "DISTRIBUTION"
        elif "correlation" in lower:
            action = "TREND_RELATION"
        elif re.search(r"\b(?:largest|smallest|minimum|maximum)\b", lower):
            action = "EXTREME"
        else:
            action = "STATS"

    return {
        "intent": intent,
        "confidence": confidence,
        "reason": reason,
        "search_updates": updates,
        "relative_change": relative,
        "result_count": result_count,
        "compare_displayed": intent == "compare",
        "analysis_action": action,
        "knowledge_topic": knowledge_topic,
        "knowledge_query": text if intent == "diamond_knowledge" else "",
        "model_targets": model_targets,
        "needs_clarification": False,
        "clarifying_question": "",
        "preserve_search_state": True,
    }


def _merge_semantic(primary: dict[str, Any], hints: dict[str, Any]) -> dict[str, Any]:
    """Merge LLM interpretation with deterministic facts from the literal message."""
    merged = dict(primary)
    if hints.get("intent"):
        # Literal high-confidence patterns win over an LLM misclassification.
        # Free-form messages that do not match these patterns still rely on the LLM.
        merged["intent"] = hints["intent"]
        merged["confidence"] = max(
            float(merged.get("confidence", 0.0)),
            float(hints.get("confidence", 0.0)),
        )
        merged["reason"] = hints.get("reason", merged.get("reason", ""))

    updates = dict(merged.get("search_updates") or {})
    updates.update(hints.get("search_updates") or {})
    merged["search_updates"] = updates

    relative = dict(merged.get("relative_change") or {})
    relative.update(hints.get("relative_change") or {})
    merged["relative_change"] = relative

    if hints.get("result_count") is not None:
        merged["result_count"] = hints["result_count"]
    if hints.get("analysis_action"):
        merged["analysis_action"] = hints["analysis_action"]
    if hints.get("knowledge_topic"):
        merged["knowledge_topic"] = hints["knowledge_topic"]
    if hints.get("model_targets"):
        merged["model_targets"] = hints["model_targets"]
    if hints.get("compare_displayed"):
        merged["compare_displayed"] = True
    return _clean_semantic(merged)


class SemanticUnderstanding:
    """Constrained natural-language parser. It never executes tools or invents data."""

    def __init__(self, llm):
        self.llm = llm

    def interpret(
        self,
        question: str,
        *,
        conversation: list[dict] | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = dict(state or {})
        recent = [
            {"role": str(item.get("role", "")), "content": str(item.get("content", ""))[:700]}
            for item in (conversation or [])[-6:]
            if isinstance(item, dict)
        ]
        hints = _deterministic_semantic_hints(question, state)

        system = """You are a semantic parser for a diamond decision assistant.
Return JSON only. Do NOT answer the user and do NOT invent values.

Supported intents: search, compare, dataset_analysis, diamond_knowledge,
model_prediction, small_talk, out_of_scope.

Understand normal conversational language, typos, and follow-ups. Use STRUCTURED STATE and
RECENT CONVERSATION only to resolve references. A knowledge/statistics detour does not erase
an active search. If the assistant asked for a field, a short answer can fill that pending field.

Important distinctions:
- Search constraints are hard filters: price ranges, carat ranges, explicit cut/color/clarity grades.
- Preferences are rankings: 'clarity matters', 'I care about size', 'clear/clean-looking diamond',
  'best value', 'focus on cut'. Put these in priorities; do not invent a grade.
- 'What does clarity mean?', 'why is clarity important?', 'types of cut' are diamond_knowledge.
- Mean/median/range/count/distribution/correlation are dataset_analysis.
- 'compare first and third' after shown rows is compare.
- Saved model requests explicitly say predict/classify/estimate and name price, clarity, or cluster.
- If a message is unrelated to diamonds, use out_of_scope.
- Clarify only when the request is genuinely impossible to interpret; do not turn search into a form.

Return exactly these keys:
{
  "intent": "search|compare|dataset_analysis|diamond_knowledge|model_prediction|small_talk|out_of_scope",
  "confidence": 0.0,
  "reason": "short reason",
  "search_updates": {
    "min_price": null, "max_price": null, "target_price": null,
    "target_carat": null, "carat_tolerance": null,
    "cut": null, "color": null, "clarity": null,
    "priorities": [], "pending_field": null
  },
  "relative_change": {},
  "result_count": null,
  "compare_displayed": false,
  "analysis_action": null,
  "knowledge_topic": "",
  "knowledge_query": "",
  "model_targets": [],
  "needs_clarification": false,
  "clarifying_question": "",
  "preserve_search_state": true
}"""
        prompt = (
            "STRUCTURED STATE:\n"
            + json.dumps(state, default=str, separators=(",", ":"))
            + "\nRECENT CONVERSATION:\n"
            + json.dumps(recent, default=str, separators=(",", ":"))
            + "\nCURRENT MESSAGE:\n"
            + question
        )
        try:
            if hasattr(self.llm, "structured"):
                raw = self.llm.structured(system, prompt)
            else:
                raw = _json_object(self.llm.complete(system, prompt))
            if not isinstance(raw, dict):
                raise TypeError("Semantic parser did not return an object.")
            parsed = _clean_semantic(raw)
            return _merge_semantic(parsed, hints)
        except Exception:
            # The assistant still works if Ollama/semantic parsing fails for one turn.
            fallback = dict(hints)
            if not fallback.get("intent"):
                fallback.update(
                    intent="out_of_scope",
                    confidence=0.0,
                    reason="semantic parser unavailable; deterministic router fallback",
                )
            return _clean_semantic(fallback)
