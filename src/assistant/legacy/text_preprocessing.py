"""Canonical text preprocessing shared by every assistant control stage."""

from dataclasses import dataclass
import re

from ..clarification import normalize_user_text


@dataclass(frozen=True)
class CanonicalMessage:
    """Preserve the raw message alongside one normalized representation."""

    raw: str
    canonical: str


def canonicalize_message(message: str) -> CanonicalMessage:
    """Normalize common input mistakes without changing the user's meaning."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Question must contain text.")
    canonical = normalize_user_text(message)
    canonical = canonical.replace("’", "'")
    canonical = re.sub(r"\s+", " ", canonical).strip()
    return CanonicalMessage(raw=message, canonical=canonical)
