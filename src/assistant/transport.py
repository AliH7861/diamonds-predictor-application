"""JSON-safe transport helpers for the separated assistant frontend and backend."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


DATAFRAME_KEYS = ("matches", "similar_matches")


def _json_value(value: Any) -> Any:
    """Convert NumPy, Pandas, and nested values into standard JSON values."""
    if isinstance(value, pd.DataFrame):
        clean = value.replace({np.nan: None})
        return clean.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return _json_value(value.to_dict())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def encode_result(result: dict[str, Any]) -> dict[str, Any]:
    """Encode an assistant result for an HTTP response."""
    return _json_value(result)


def decode_result(result: dict[str, Any]) -> dict[str, Any]:
    """Restore transported result rows to DataFrames expected by the UI."""
    decoded = dict(result)
    for key in DATAFRAME_KEYS:
        value = decoded.get(key, [])
        if not isinstance(value, pd.DataFrame):
            decoded[key] = pd.DataFrame(value or [])
    return decoded


def encode_conversation(
    conversation: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Keep visible chat and compact row evidence needed by similarity follow-ups."""
    encoded: list[dict[str, Any]] = []
    for message in conversation or []:
        item: dict[str, Any] = {
            "role": str(message.get("role", "user")),
            "content": str(message.get("content", "")),
        }
        result = message.get("result")
        if isinstance(result, dict):
            item["result"] = {key: _json_value(result.get(key, [])) for key in DATAFRAME_KEYS}
        encoded.append(item)
    return encoded


def decode_conversation(
    conversation: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Restore any earlier row evidence before backend orchestration."""
    decoded = []
    for message in conversation or []:
        item = dict(message)
        if isinstance(item.get("result"), dict):
            item["result"] = decode_result(item["result"])
        decoded.append(item)
    return decoded
