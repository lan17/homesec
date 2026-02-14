"""Cursor token helpers for API pagination."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone

from homesec.models.clip import ClipListCursor


class CursorDecodeError(ValueError):
    """Raised when cursor token cannot be decoded or validated."""


def encode_clip_cursor(cursor: ClipListCursor) -> str:
    """Encode clip cursor into a URL-safe opaque token."""
    payload = {
        "created_at": cursor.created_at.astimezone(timezone.utc).isoformat(),
        "clip_id": cursor.clip_id,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_clip_cursor(token: str) -> ClipListCursor:
    """Decode URL-safe token into clip cursor."""
    try:
        padded = token + "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(raw)
    except Exception as exc:  # pragma: no cover - defensive
        raise CursorDecodeError("Invalid cursor token") from exc

    if not isinstance(payload, dict):
        raise CursorDecodeError("Invalid cursor token")

    created_at_raw = payload.get("created_at")
    clip_id_raw = payload.get("clip_id")
    if not isinstance(created_at_raw, str) or not isinstance(clip_id_raw, str):
        raise CursorDecodeError("Invalid cursor token")
    if not clip_id_raw:
        raise CursorDecodeError("Invalid cursor token")

    try:
        created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CursorDecodeError("Invalid cursor token") from exc

    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise CursorDecodeError("Invalid cursor token")

    return ClipListCursor(
        created_at=created_at.astimezone(timezone.utc),
        clip_id=clip_id_raw,
    )
