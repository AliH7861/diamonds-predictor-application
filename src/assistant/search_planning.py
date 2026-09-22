"""Build one deterministic search plan from interpreted language and chat state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from .clarification import normalize_user_text
from .clarity import clarity_family
from .schemas import DiamondQueryPlan


AMOUNT = r"\d[\d,]*(?:\.\d+)?(?:\s*k)?"
PLAN_STATE_FIELDS = (
    "min_price",
    "max_price",
    "target_price",
    "target_carat",
    "carat_tolerance",
    "cut",
    "color",
    "clarity",
    "priorities",
)
STATE_FIELDS = (*PLAN_STATE_FIELDS, "search_active", "pending_field")
CUTS = ("Very Good", "Premium", "Ideal", "Fair", "Good")
INVALID_CLARITIES = ("VI", "IV", "VSI", "VVS3", "VS3", "SI3")


def _amount(value: str) -> float:
    clean = value.casefold().replace(",", "").replace(" ", "")
    multiplier = 1000 if clean.endswith("k") else 1
    return float(clean.removesuffix("k")) * multiplier


def _last(pattern: str, text: str) -> re.Match[str] | None:
    matches = list(re.finditer(pattern, text, re.I))
    return matches[-1] if matches else None


def _cut(text: str) -> str | None:
    explicit = _last(
        r"\b(Very Good|Premium|Ideal|Fair|Good)\s+cut\b|"
        r"\bcut(?:\s+grade)?\s+(Very Good|Premium|Ideal|Fair|Good)\b",
        text,
    )
    if explicit:
        value = explicit.group(1) or explicit.group(2)
        return next(item for item in CUTS if item.casefold() == value.casefold())
    bare = _last(r"\b(Very Good|Premium|Ideal|Fair)\b", text)
    if bare:
        return next(item for item in CUTS if item.casefold() == bare.group(1).casefold())
    return None


def _clarity(text: str) -> str | None:
    match = _last(r"\b(I1|SI1|SI2|VS1|VS2|VVS1|VVS2|IF|SI|VS|VVS)\b", text)
    return clarity_family(match.group(1)) if match else None


def _color(text: str) -> str | None:
    match = _last(
        r"\bcolor(?:\s+grade)?\s+([D-J])\b|\b([D-J])\s+color\b",
        text,
    )
    return (match.group(1) or match.group(2)).upper() if match else None


def _has_search_value(values: dict[str, Any]) -> bool:
    return any(
        values.get(field) not in (None, [], "")
        for field in (
            "min_price",
            "max_price",
            "target_price",
            "target_carat",
            "cut",
            "color",
            "clarity",
            "priorities",
        )
    )


def _natural_priorities(lower: str) -> list[str]:
    found: list[tuple[int, str]] = []
    patterns = {
        "clarity": (
            r"\bclarity\b.{0,35}\b(?:matters?|important|priority|focus|first|most)\b|"
            r"\b(?:care|focus|priorit(?:y|ize)|important|most)\b.{0,35}\bclarity\b|"
            r"\b(?:clear|clean[- ]?looking|cleanliness)\s+diamond\b"
        ),
        "size": (
            r"\bsize\b.{0,35}\b(?:matters?|important|priority|focus|first|most)\b|"
            r"\b(?:care|focus|priorit(?:y|ize)|important|most)\b.{0,35}\bsize\b|"
            r"\b(?:bigger|larger)\s+(?:stone|diamond)\b"
        ),
        "cut": r"\bcut\b.{0,35}\b(?:matters?|important|priority|focus|first|most)\b|\b(?:focus|priorit(?:y|ize))\b.{0,35}\bcut\b",
        "color": r"\bcolor\b.{0,35}\b(?:matters?|important|priority|focus|first|most)\b|\b(?:focus|priorit(?:y|ize))\b.{0,35}\bcolor\b",
        "price": r"\bprice\b.{0,35}\b(?:matters?|important|priority|focus|first|most)\b|\b(?:focus|priorit(?:y|ize))\b.{0,35}\bprice\b",
        "value": r"\b(?:best\s+value|value\s+matters?|prioritize\s+value)\b",
        "balance": r"\b(?:balanced?|trade[- ]?off)\b",
    }
    for priority, pattern in patterns.items():
        for match in re.finditer(pattern, lower, re.I):
            found.append((match.start(), priority))
    return [max(found)[1]] if found else []


@dataclass
class SearchInterpretation:
    """Parsed, validated request plus compact state and ranking behavior."""

    plan: DiamondQueryPlan
    state: dict[str, Any]
    raw_fields: dict[str, Any]
    accepted_fields: dict[str, Any]
    dropped_fields: dict[str, str]
    strategy: str
    exclude_ids: list[int]
    clarification: str | None = None

    def audit(self, previous_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "raw_criteria": self.raw_fields,
            "accepted_criteria": self.accepted_fields,
            "dropped_candidates": self.dropped_fields,
            "problems": [self.clarification] if self.clarification else [],
            "previous_state": {
                key: previous_state[key]
                for key in (*STATE_FIELDS, "last_result_ids")
                if key in previous_state
            },
            "resulting_state": dict(self.state),
            "ranking_strategy": self.strategy,
            "hard_filters": asdict(self.plan),
        }


def build_search_plan(
    question: str,
    state: dict[str, Any] | None = None,
    *,
    semantic_updates: dict[str, Any] | None = None,
    relative_change: dict[str, str] | None = None,
) -> SearchInterpretation:
    """Interpret one search turn while preserving validated state from the same chat."""
    text = normalize_user_text(question).strip()
    lower = text.casefold().replace("’", "'")
    previous = dict(state or {})

    values = {field: previous.get(field) for field in PLAN_STATE_FIELDS}
    values["priorities"] = list(values.get("priorities") or [])
    search_active = bool(previous.get("search_active")) or _has_search_value(values)
    pending_field = previous.get("pending_field")

    raw: dict[str, Any] = {}
    accepted: dict[str, Any] = {}
    dropped: dict[str, str] = {}

    # An explicit reset starts a fresh search without leaking previous criteria.
    if re.search(
        r"\b(?:new search|start over|reset search|clear search|forget everything)\b", lower
    ):
        values = {field: None for field in PLAN_STATE_FIELDS}
        values["priorities"] = []
        search_active = True
        pending_field = None

    # Semantic parsing handles free-form language. Values are still revalidated below.
    for field, value in (semantic_updates or {}).items():
        if field not in (*PLAN_STATE_FIELDS, "pending_field"):
            continue
        if field == "priorities":
            if isinstance(value, str):
                value = [value]
            if isinstance(value, list):
                allowed = {"price", "size", "cut", "color", "clarity", "value", "balance"}
                clean = [str(item).casefold() for item in value if str(item).casefold() in allowed]
                if clean:
                    values[field] = clean
                    raw[field] = clean
                    accepted[field] = clean
        elif field == "pending_field":
            if value in {None, "price", "carat", "cut", "color", "clarity", "priority"}:
                pending_field = value
        elif field in {"min_price", "max_price", "target_price", "target_carat", "carat_tolerance"}:
            try:
                number = float(value)
            except (TypeError, ValueError):
                dropped[field] = "not numeric"
            else:
                if number >= 0:
                    values[field] = number
                    raw[field] = number
                    accepted[field] = number
        elif field == "cut" and value in CUTS:
            values[field] = value
            raw[field] = value
            accepted[field] = value
        elif field == "color" and isinstance(value, str) and value.upper() in set("DEFGHIJ"):
            values[field] = value.upper()
            raw[field] = values[field]
            accepted[field] = values[field]
        elif (
            field == "clarity"
            and isinstance(value, str)
            and value.upper() in {"I", "SI", "VS", "VVS", "IF"}
        ):
            values[field] = value.upper()
            raw[field] = values[field]
            accepted[field] = values[field]

    invalid = _last(r"\b(?:clarity\s+)?(VI|IV|VSI|VVS3|VS3|SI3)\b", text)
    if invalid:
        token = invalid.group(1).upper()
        message = (
            f"{token} isn't a clarity grade in this dataset. "
            "Available clarity families are I, SI, VS, VVS, and IF."
        )
        result_state = {
            key: value
            for key, value in previous.items()
            if key in (*STATE_FIELDS, "last_result_ids")
        }
        result_state["search_active"] = True
        result_state["pending_field"] = "clarity"
        dropped["clarity"] = f"{token} is not supported"
        return SearchInterpretation(
            DiamondQueryPlan(
                search_dataset=False,
                needs_clarification=True,
                clarifying_question=message,
            ),
            result_state,
            {**raw, "clarity": token},
            accepted,
            dropped,
            "none",
            [],
            message,
        )

    # Explicit removals beat inherited state and semantic guesses.
    for field in ("clarity", "cut", "color"):
        if re.search(
            rf"\b(?:forget|remove|clear|drop)\s+(?:the\s+)?{field}\b|"
            rf"\bdon't care(?:\s+that much)?\s+about\s+(?:the\s+)?{field}\b",
            lower,
        ):
            values[field] = None
            raw[field] = None
            accepted[field] = None

    # Price parsing. A natural range does not need dollar signs.
    price_range = _last(
        rf"\b(?:between|from|range\s+(?:of\s+)?)\s*\$?\s*({AMOUNT})\s*"
        rf"(?:and|to|-)\s*\$?\s*({AMOUNT})\b|"
        rf"\b(?:price|budget).{{0,20}}(?:between|from)\s*\$?\s*({AMOUNT})\s*"
        rf"(?:and|to|-)\s*\$?\s*({AMOUNT})\b",
        text,
    )
    approximate = _last(
        rf"\b(?:around|about|near|close(?:r)?\s+to|priced?\s+at)\s+\$?\s*({AMOUNT})\b",
        text,
    )
    maximum = _last(
        rf"(?:\b(?:under|below|less than|up to|max(?:imum)?(?:\s+budget)?(?:\s+of)?|"
        rf"budget(?:\s+of)?|for)\s*\$\s*({AMOUNT})\b|"
        rf"\b(?:under|below|less than|up to|max(?:imum)?(?:\s+budget)?(?:\s+of)?|budget(?:\s+of)?)\s*\$?\s*({AMOUNT})\b|"
        rf"\$\s*({AMOUNT})\s+budget\b)",
        text,
    )
    minimum = _last(rf"\b(?:over|above|at least|minimum(?:\s+of)?)\s*\$?\s*({AMOUNT})\b", text)
    lone_money = _last(rf"\$\s*({AMOUNT})\b", text)

    if price_range:
        first = price_range.group(1) or price_range.group(3)
        second = price_range.group(2) or price_range.group(4)
        low, high = sorted((_amount(first), _amount(second)))
        if low >= 100 and high >= 100:
            values.update(min_price=low, max_price=high, target_price=None)
            raw.update(min_price=low, max_price=high)
            accepted.update(min_price=low, max_price=high)
    elif approximate:
        target = _amount(approximate.group(1))
        if target >= 100:
            values.update(target_price=target, min_price=None, max_price=None)
            raw["target_price"] = target
            accepted["target_price"] = target
    else:
        if maximum:
            group = next(value for value in maximum.groups() if value is not None)
            value = _amount(group)
            if value >= 100:
                values["max_price"] = value
                values["target_price"] = None
                raw["max_price"] = value
                accepted["max_price"] = value
        elif lone_money:
            value = _amount(lone_money.group(1))
            if value >= 100:
                # In a search turn, "$2000" or "diamond for $2000" means a ceiling.
                values["max_price"] = value
                values["target_price"] = None
                raw["max_price"] = value
                accepted["max_price"] = value
        if minimum:
            value = _amount(minimum.group(1))
            if value >= 100:
                values["min_price"] = value
                raw["min_price"] = value
                accepted["min_price"] = value

    # Short numeric answer to an earlier price question.
    if pending_field == "price":
        answer = re.fullmatch(rf"\s*\$?\s*({AMOUNT})\s*", text, re.I)
        if answer:
            value = _amount(answer.group(1))
            if value >= 100:
                values["max_price"] = value
                values["target_price"] = None
                raw["max_price"] = value
                accepted["max_price"] = value
                pending_field = None

    carat_range = _last(
        r"\b(?:between|from)?\s*(\d*\.\d+|\d+(?:\.\d+)?)\s*(?:and|to|-)\s*"
        r"(\d*\.\d+|\d+(?:\.\d+)?)\s*(?:ct|carats?)\b",
        text,
    )
    carat = _last(r"\b(\d*\.\d+|\d+(?:\.\d+)?)\s*(?:ct|carats?)\b", text)
    if carat_range:
        low, high = sorted((float(carat_range.group(1)), float(carat_range.group(2))))
        values["target_carat"] = round((low + high) / 2, 4)
        values["carat_tolerance"] = round((high - low) / 2, 4)
        raw.update(target_carat=values["target_carat"], carat_tolerance=values["carat_tolerance"])
        accepted.update(
            target_carat=values["target_carat"], carat_tolerance=values["carat_tolerance"]
        )
    elif carat:
        values["target_carat"] = float(carat.group(1))
        values["carat_tolerance"] = 0.10
        raw.update(target_carat=values["target_carat"], carat_tolerance=0.10)
        accepted.update(target_carat=values["target_carat"], carat_tolerance=0.10)
    elif pending_field == "carat":
        answer = re.fullmatch(r"\s*(\d*\.\d+|\d+(?:\.\d+)?)\s*", text)
        if answer:
            values["target_carat"] = float(answer.group(1))
            values["carat_tolerance"] = 0.10
            raw.update(target_carat=values["target_carat"], carat_tolerance=0.10)
            accepted.update(target_carat=values["target_carat"], carat_tolerance=0.10)
            pending_field = None

    for field, extractor in (("cut", _cut), ("clarity", _clarity), ("color", _color)):
        value = extractor(text)
        if value:
            values[field] = value
            raw[field] = value
            accepted[field] = value
            if pending_field == field:
                pending_field = None

    priorities = _natural_priorities(lower)
    plain = re.sub(r"[^a-z]+", " ", lower).strip()
    if pending_field == "priority" and plain in {
        "price",
        "size",
        "cut",
        "color",
        "clarity",
        "value",
        "balance",
    }:
        priorities = [plain]
        pending_field = None
    # Also accept a one-word priority during an already-active search. This is
    # what makes a conversational reply like "clarity" useful instead of looping.
    elif search_active and plain in {
        "price",
        "size",
        "cut",
        "color",
        "clarity",
        "value",
        "balance",
    }:
        priorities = [plain]

    if priorities:
        values["priorities"] = [priorities[-1]]
        raw["priorities"] = values["priorities"]
        accepted["priorities"] = values["priorities"]

    relative_change = dict(relative_change or {})
    strategy = "balanced"
    if re.search(r"\b(?:different|another|new)\s+(?:ones?|options?|diamonds?)\b", lower):
        strategy = (
            "different_diverse"
            if values.get("min_price") is not None and values.get("max_price") is not None
            else "different"
        )
    elif "lower end" in lower:
        strategy = "lower_end"
    elif "middle" in lower or "midpoint" in lower:
        strategy = "middle"
    elif re.search(r"\b(?:bigger|larger)\b", lower) or relative_change.get("carat") == "increase":
        strategy = "bigger"
    elif (
        re.search(r"\b(?:cheaper|lower price|less expensive)\b", lower)
        or relative_change.get("price") == "decrease"
    ):
        strategy = "cheaper"
    elif values.get("target_price") is not None:
        strategy = "target_price"
    elif values.get("min_price") is not None and values.get("max_price") is not None:
        strategy = "diverse_range"
    elif values.get("max_price") is not None:
        strategy = "budget_top"

    meaningful = _has_search_value(values)
    clarification = None
    if not meaningful:
        if re.fullmatch(r"(?:price|price only)", lower):
            clarification = "What is your maximum budget or target price?"
            pending_field = "price"
        elif re.fullmatch(r"(?:size|carat|size only)", lower):
            clarification = "What carat size or carat range would you like?"
            pending_field = "carat"
        else:
            clarification = "What price, size, cut, color, or clarity preference should I use?"
            pending_field = "priority"

    search_active = True
    compact_state = {key: value for key, value in values.items() if value not in (None, [], "")}
    compact_state["search_active"] = True
    if pending_field:
        compact_state["pending_field"] = pending_field
    if previous.get("last_result_ids"):
        compact_state["last_result_ids"] = list(previous.get("last_result_ids") or [])

    exclude_ids = (
        list(previous.get("last_result_ids") or []) if strategy.startswith("different") else []
    )

    plan = DiamondQueryPlan(
        search_dataset=meaningful,
        min_price=values.get("min_price"),
        max_price=values.get("max_price"),
        target_price=values.get("target_price"),
        target_carat=values.get("target_carat"),
        carat_tolerance=float(values.get("carat_tolerance") or 0.10),
        cut=values.get("cut"),
        color=values.get("color"),
        clarity=values.get("clarity"),
        priorities=list(values.get("priorities") or []),
        needs_clarification=bool(clarification),
        clarifying_question=clarification,
    )
    return SearchInterpretation(
        plan, compact_state, raw, accepted, dropped, strategy, exclude_ids, clarification
    )


def state_from_plan(plan: DiamondQueryPlan) -> dict[str, Any]:
    """Return compact public state for a validated search plan."""
    data = asdict(plan)
    state = {
        field: data[field] for field in PLAN_STATE_FIELDS if data.get(field) not in (None, [], "")
    }
    if state:
        state["search_active"] = True
    return state
