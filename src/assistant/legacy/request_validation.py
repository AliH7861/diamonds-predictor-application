"""Field-level evidence checks for structured diamond search criteria."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import re
from typing import Any

from ..clarity import clarity_family
from ..schemas import DiamondQueryPlan
from .text_preprocessing import CanonicalMessage


@dataclass(frozen=True)
class FieldEvidence:
    value: Any
    source: str
    evidence: str
    confidence: float = 0.99


@dataclass
class RequestEnvelope:
    message: CanonicalMessage
    candidates: dict[str, Any]
    accepted: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, FieldEvidence] = field(default_factory=dict)
    dropped: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    def public_dict(self) -> dict[str, Any]:
        """Return JSON-safe developer evidence."""
        return {
            "raw_text": self.message.raw,
            "canonical_text": self.message.canonical,
            "candidate_criteria": self.candidates,
            "accepted_criteria": self.accepted,
            "field_evidence": {key: asdict(value) for key, value in self.evidence.items()},
            "dropped_candidates": self.dropped,
            "problems": self.problems,
        }


STATE_FIELDS = {
    "min_price": "min_price",
    "max_price": "max_price",
    "target_price": "target_price",
    "target_carat": "target_carat",
    "carat_tolerance": "carat_tolerance",
    "depth": "depth",
    "table": "table",
    "x": "x",
    "y": "y",
    "z": "z",
    "cut": "cut",
    "color": "color",
    "clarity": "clarity",
    "priorities": "priorities",
}


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 1e-9
    return left == right


def _number_present(text: str, value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    compact = text.casefold().replace(",", "").replace("$", "")
    for match in re.finditer(r"(?<![\w.])(\d*\.\d+|\d+(?:\.\d+)?)(\s*k)?(?!\d)", compact):
        found = float(match.group(1)) * (1000 if match.group(2) else 1)
        if abs(found - number) <= 1e-9:
            return True
    return False


def _explicit_support(field_name: str, value: Any, text: str, state: dict) -> str | None:
    lower = text.casefold()
    if field_name in {"min_price", "max_price", "target_price"}:
        if _number_present(text, value) and (
            "$" in text
            or re.search(
                r"\b(?:budget|price|under|below|around|about|max(?:imum)?|less than)\b", lower
            )
            or state.get("pending_buying")
        ):
            return "explicit budget or price expression"
    elif field_name == "target_carat":
        ranges = re.findall(
            r"(?:\bbetween\s+)?(\d*\.\d+|\d+(?:\.\d+)?)\s*(?:-|to|and)\s*(\d*\.\d+|\d+(?:\.\d+)?)\s*(?:ct|carat|carats)\b",
            lower,
        )
        if ranges:
            low, high = map(float, ranges[-1])
            if abs(float(value) - ((low + high) / 2)) <= 1e-9:
                return "midpoint derived from explicit carat range"
        if re.search(r"\b(?:small|medium|large)(?:-sized|\s+size(?:d)?)?\b", lower):
            return "explicit descriptive size"
        if _number_present(text, value) and re.search(r"(?:ct|carat|carats)\b", lower):
            return "explicit carat expression"
    elif field_name == "carat_tolerance":
        if re.search(
            r"(?:\bbetween\b.+\band\b|\d\s*(?:-|to)\s*\d).+\b(?:ct|carat|carats)\b", lower
        ):
            return "explicit carat range"
        if re.search(r"\b(?:small|medium|large)(?:-sized|\s+size(?:d)?)?\b", lower):
            return "size-range mapping"
    elif field_name in {"depth", "table", "x", "y", "z"}:
        if re.search(rf"\b{field_name}\s*(?:is|=|:)?\s*{re.escape(f'{float(value):g}')}\b", lower):
            return f"explicit {field_name} value"
    elif field_name == "cut" and re.search(rf"\b{re.escape(str(value))}\b", text, re.I):
        return "explicit cut grade"
    elif field_name == "color" and re.search(
        rf"\b(?:color\s+)?{re.escape(str(value))}(?:\s+color)?\b", text, re.I
    ):
        return "explicit color grade"
    elif field_name == "clarity":
        stated = re.findall(r"\b(?:I1|SI1|SI2|VS1|VS2|VVS1|VVS2|IF|I|SI|VS|VVS)\b", text, re.I)
        if any(clarity_family(item) == value for item in stated):
            return "explicit clarity family"
        if value == "VVS" and re.search(r"\b(?:high|excellent|very good)\s+clarity\b", lower):
            return "explicit high-clarity preference"
        if value == "VS" and re.search(r"\bbalance\b.{0,30}\bclarity\b", lower):
            return "explicit clarity/value balance"
    elif field_name == "priorities":
        supported = [item for item in value if re.search(rf"\b{re.escape(item)}\b", lower)]
        if supported:
            return "explicit preference words: " + ", ".join(supported)
    return None


def validate_plan(
    message: CanonicalMessage,
    plan: DiamondQueryPlan,
    prior_state: dict | None = None,
) -> tuple[DiamondQueryPlan, RequestEnvelope]:
    """Freeze only criteria supported by current text or validated prior state."""
    state = dict(prior_state or {})
    candidates = {
        key: value
        for key, value in asdict(plan).items()
        if key
        not in {"search_dataset", "needs_clarification", "clarifying_question", "knowledge_queries"}
        and value not in (None, "", [], {})
    }
    envelope = RequestEnvelope(message, deepcopy(candidates))
    accepted: dict[str, Any] = {}
    for field_name, value in candidates.items():
        state_name = STATE_FIELDS.get(field_name)
        if state_name and state_name in state and _same(state[state_name], value):
            accepted[field_name] = deepcopy(value)
            envelope.evidence[field_name] = FieldEvidence(
                value, "prior_state", "unchanged validated conversation state", 0.95
            )
            continue
        support = _explicit_support(field_name, value, message.canonical, state)
        if field_name == "carat_tolerance" and "target_carat" in accepted:
            support = support or "tolerance derived from supported carat target"
        if support:
            accepted[field_name] = deepcopy(value)
            envelope.evidence[field_name] = FieldEvidence(value, "current_text", support)
        else:
            envelope.dropped[field_name] = "No field-level evidence in current text or prior state"

    minimum, maximum = accepted.get("min_price"), accepted.get("max_price")
    if minimum is not None and maximum is not None and minimum > maximum:
        envelope.problems.append("minimum price exceeds maximum price")
    target, tolerance = accepted.get("target_carat"), accepted.get("carat_tolerance")
    if target is not None and target <= 0:
        envelope.problems.append("target carat must be positive")
    if tolerance is not None and tolerance < 0:
        envelope.problems.append("carat tolerance cannot be negative")
    stated_range = re.search(
        r"\bbetween\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*(?:ct|carat|carats)\b",
        message.canonical,
        re.I,
    )
    if stated_range and float(stated_range.group(1)) > float(stated_range.group(2)):
        envelope.problems.append("carat range is reversed")
    if re.search(r"(?<!\d)-\d+(?:\.\d+)?\s*(?:ct|carat|carats)\b", message.canonical, re.I):
        envelope.problems.append("carat cannot be negative")

    clean = asdict(plan)
    for field_name in candidates:
        clean[field_name] = accepted.get(field_name)
    clean["priorities"] = list(accepted.get("priorities") or [])
    clean["knowledge_queries"] = list(plan.knowledge_queries)
    if envelope.problems:
        clean["needs_clarification"] = True
        clean["clarifying_question"] = (
            "I found conflicting search details. Please restate the budget or size range."
        )
    envelope.accepted = accepted
    return DiamondQueryPlan.from_dict(clean), envelope
