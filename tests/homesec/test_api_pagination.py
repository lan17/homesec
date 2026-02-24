"""Tests for API cursor pagination helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from homesec.api.pagination import CursorDecodeError, decode_clip_cursor, encode_clip_cursor
from homesec.models.clip import ClipListCursor


def test_encode_decode_clip_cursor_round_trip_normalizes_to_utc() -> None:
    # Given: A cursor with timezone-aware timestamp
    cursor = ClipListCursor(
        created_at=datetime.fromisoformat("2026-02-15T12:00:00+02:00"),
        clip_id="clip-123",
    )

    # When: Encoding and decoding cursor token
    token = encode_clip_cursor(cursor)
    decoded = decode_clip_cursor(token)

    # Then: Cursor round-trips with UTC-normalized datetime
    assert decoded.clip_id == "clip-123"
    assert decoded.created_at.tzinfo is UTC
    assert decoded.created_at.isoformat() == "2026-02-15T10:00:00+00:00"


def test_decode_clip_cursor_rejects_non_mapping_payload() -> None:
    # Given: A base64 token whose JSON payload is not an object
    token = "WyJub3QtYS1tYXBwaW5nIl0"

    # When: Decoding the cursor token
    with pytest.raises(CursorDecodeError):
        decode_clip_cursor(token)

    # Then: The token is rejected as invalid


def test_decode_clip_cursor_rejects_empty_clip_id() -> None:
    # Given: A token with empty clip_id field
    token = "eyJjbGlwX2lkIjoiIiwiY3JlYXRlZF9hdCI6IjIwMjYtMDItMTVUMTI6MDA6MDBaIn0"

    # When: Decoding the cursor token
    with pytest.raises(CursorDecodeError):
        decode_clip_cursor(token)

    # Then: The token is rejected as invalid


def test_decode_clip_cursor_rejects_invalid_created_at_format() -> None:
    # Given: A token with invalid created_at format
    token = "eyJjbGlwX2lkIjoiY2xpcC0xIiwiY3JlYXRlZF9hdCI6Im5vdC1hLWRhdGUifQ"

    # When: Decoding the cursor token
    with pytest.raises(CursorDecodeError):
        decode_clip_cursor(token)

    # Then: The token is rejected as invalid


def test_decode_clip_cursor_rejects_naive_timestamp() -> None:
    # Given: A token whose created_at timestamp has no timezone
    token = "eyJjbGlwX2lkIjoiY2xpcC0xIiwiY3JlYXRlZF9hdCI6IjIwMjYtMDItMTVUMTI6MDA6MDAifQ"

    # When: Decoding the cursor token
    with pytest.raises(CursorDecodeError):
        decode_clip_cursor(token)

    # Then: The token is rejected as invalid
