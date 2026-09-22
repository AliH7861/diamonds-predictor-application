"""Transactional delta updates for frozen validated conversation state."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import re
from typing import Any

from .request_validation import RequestEnvelope


@dataclass(frozen=True)
class StateDelta:
    field: str
    operation: str
    old_value: Any
    new_value: Any
    evidence: str


def apply_state_transaction(previous: dict | None, envelope: RequestEnvelope) -> tuple[dict, dict]:
    """Commit only current-message evidence; preserve every other validated value."""
    before = deepcopy(previous or {})
    text = envelope.message.canonical.casefold()
    turn = int(before.get("_turn", 0)) + 1
    history = deepcopy(before.get("_history", []))
    if re.search(r"\b(?:undo|go back|revert)\b", text) and history:
        restored = deepcopy(history[-1])
        restored["_history"] = history[:-1]
        restored["_turn"] = turn
        audit = {
            "previous_validated_state": before,
            "candidate_delta": [
                {
                    "field": "*",
                    "operation": "UNDO",
                    "old_value": before,
                    "new_value": restored,
                    "evidence": "explicit undo request",
                }
            ],
            "accepted_delta": [
                {
                    "field": "*",
                    "operation": "UNDO",
                    "old_value": before,
                    "new_value": restored,
                    "evidence": "explicit undo request",
                }
            ],
            "rejected_delta": [],
            "resulting_state": restored,
            "committed": True,
        }
        return restored, audit
    result = deepcopy(before)
    accepted = []
    for field_name, evidence in envelope.evidence.items():
        if evidence.source != "current_text":
            continue
        old = result.get(field_name)
        operation = "SET" if old is None else ("KEEP" if old == evidence.value else "REPLACE")
        result[field_name] = deepcopy(evidence.value)
        accepted.append(StateDelta(field_name, operation, old, evidence.value, evidence.evidence))
    aliases = {
        "budget": "max_price",
        "price": "max_price",
        "carat": "target_carat",
        "size": "target_carat",
        "cut": "cut",
        "color": "color",
        "clarity": "clarity",
    }
    for alias, field_name in aliases.items():
        if re.search(
            rf"\b(?:remove|forget|clear|don't care about|dont care about)\s+(?:the\s+)?{alias}\b",
            text,
        ):
            old = result.pop(field_name, None)
            accepted.append(
                StateDelta(field_name, "REMOVE", old, None, f"explicit removal of {alias}")
            )
    rejected = []
    if envelope.problems:
        rejected = [asdict(item) for item in accepted]
        result, accepted = before, []
    changed = any(item.operation in {"SET", "REPLACE", "REMOVE"} for item in accepted)
    if changed:
        clean_before = {key: value for key, value in before.items() if not key.startswith("_")}
        history.append(clean_before)
        result["_history"] = history[-20:]
    provenance = deepcopy(before.get("_provenance", {}))
    for item in accepted:
        provenance[item.field] = {
            "evidence": item.evidence,
            "operation": item.operation,
        }
    if provenance:
        result["_provenance"] = provenance
    for item in accepted:
        result.setdefault("_provenance", {}).setdefault(item.field, {}).update(
            {
                "confidence": 0.99,
                "last_modified_turn": turn,
            }
        )
    result["_turn"] = turn
    result["pending_buying"] = bool(envelope.problems)
    audit = {
        "previous_validated_state": before,
        "candidate_delta": [asdict(item) for item in accepted] + rejected,
        "accepted_delta": [asdict(item) for item in accepted],
        "rejected_delta": rejected,
        "resulting_state": result,
        "committed": not envelope.problems,
    }
    return result, audit
