"""JSON-safe transport helpers for the local Diamond Assistant HTTP API."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
from typing import Any, cast

import pandas as pd

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is optional to serialization logic
    np = None


MESSAGE_KEYS = {
    "role",
    "content",
    "matches",
    "similar_matches",
    "conversation_state",
    "state",
    "evidence",
    "route",
    "plan",
    "request_id",
    "metadata",
}


def decode_conversation(value: Any) -> list[dict]:
    """Normalize browser conversation payloads without dropping displayed-row metadata."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Conversation must be a JSON array.")

    output: list[dict] = []
    for item in value[-50:]:
        if isinstance(item, str):
            content = item.strip()
            if content:
                output.append({"role": "user", "content": content})
            continue
        if not isinstance(item, dict):
            continue

        role = str(item.get("role") or item.get("sender") or "user").casefold()
        if role not in {"user", "assistant", "system"}:
            role = "user"
        content = item.get("content", item.get("text", item.get("message", "")))
        if content is None:
            content = ""
        message = {"role": role, "content": str(content)}

        # Preserve fields comparison/state code may use. Unknown UI-only fields are ignored.
        for key in MESSAGE_KEYS - {"role", "content"}:
            if key in item:
                message[key] = item[key]
        output.append(message)
    return output


def encode_conversation(value: list[dict] | None) -> list[dict]:
    """Convert Streamlit conversation messages into JSON-safe dictionaries."""
    encoded = _json_safe(value or [])
    return encoded if isinstance(encoded, list) else []


def normalize_state(value: Any) -> dict:
    """Accept only object-shaped conversation state; never silently accept a list/string."""
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError("Conversation state must be a JSON object.")
    return dict(value)


def extract_state_from_conversation(conversation: list[dict] | None) -> dict:
    """Recover the most recent state if the UI stored it on an assistant message."""
    for item in reversed(conversation or []):
        if not isinstance(item, dict):
            continue
        for key in ("conversation_state", "state"):
            candidate = item.get(key)
            if isinstance(candidate, dict):
                return dict(candidate)
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            candidate = metadata.get("conversation_state") or metadata.get("state")
            if isinstance(candidate, dict):
                return dict(candidate)
    return {}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if np is not None and isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, pd.DataFrame):
        return [_json_safe(item) for item in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(cast(Any, value)))
    if isinstance(value, type):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return number if math.isfinite(number) else None


def encode_result(result: dict) -> dict:
    """Convert pandas/numpy/dataclass values into JSON-safe builtins."""
    if not isinstance(result, dict):
        raise TypeError("Assistant result must be a dictionary.")
    return _json_safe(result)


def decode_result(result: dict) -> dict:
    """Restore DataFrame result fields expected by the optional Streamlit UI."""
    if not isinstance(result, dict):
        raise TypeError("Assistant result must be a dictionary.")
    decoded = dict(result)
    for key in ("matches", "similar_matches"):
        value = decoded.get(key)
        if isinstance(value, list):
            decoded[key] = pd.DataFrame(value)
    return decoded
