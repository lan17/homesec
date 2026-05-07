"""Safe identifiers for talk backend registration and diagnostics."""

from __future__ import annotations

import re

TALK_BACKEND_ID_PATTERN_TEXT = r"^[a-z][a-z0-9_]{0,63}$"
_TALK_BACKEND_ID_PATTERN = re.compile(TALK_BACKEND_ID_PATTERN_TEXT)


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
