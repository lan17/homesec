"""Safe identifiers for talk backend registration and diagnostics."""

from __future__ import annotations

import re

TALK_BACKEND_ID_PATTERN_TEXT = r"^[a-z][a-z0-9_]{0,63}$"
_TALK_BACKEND_ID_PATTERN = re.compile(TALK_BACKEND_ID_PATTERN_TEXT)
_UNSAFE_TALK_BACKEND_REASON_PATTERN = re.compile(
    r"://|authorization|bearer|basic|digest|password|passwd|secret|token|api[_-]?key|"
    r"\bsdp\b|(?:^|\s)(?:v=0|m=|a=)",
    re.IGNORECASE,
)


def normalize_talk_backend_id(value: str) -> str:
    """Normalize an operator/plugin provided talk backend identifier."""
    return value.strip().lower()


def validate_talk_backend_id(value: str) -> str:
    """Validate that a talk backend name is safe to expose in diagnostics."""
    if not _TALK_BACKEND_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "talk backend names must use lowercase letters, numbers, and underscores, "
            "start with a letter, and be at most 64 characters"
        )
    return value


def sanitize_talk_backend_id(value: object) -> str | None:
    """Return a safe backend identifier for public diagnostics, or None."""
    if not isinstance(value, str):
        return None
    normalized = normalize_talk_backend_id(value)
    try:
        return validate_talk_backend_id(normalized)
    except ValueError:
        return None


def sanitize_talk_backend_reason(value: object) -> str | None:
    """Return a safe short backend diagnostic reason, or None."""
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    reason = value.strip()
    if not reason or len(reason) > 256 or "\n" in reason or "\r" in reason:
        return None
    if _UNSAFE_TALK_BACKEND_REASON_PATTERN.search(reason):
        return None
    return reason
