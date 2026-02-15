"""Unit tests for media token helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from homesec.api.media_tokens import (
    MediaTokenError,
    issue_clip_media_token,
    validate_clip_media_token,
)


def test_issue_and_validate_clip_media_token_round_trip() -> None:
    """Issued token should validate for the same clip."""
    # Given a clip ID and API key
    issued_at = datetime(2026, 2, 15, 12, 0, tzinfo=UTC)

    # When issuing and validating token for same clip
    token, expires_at = issue_clip_media_token(
        api_key="secret-key",
        clip_id="clip-1",
        ttl_s=600,
        now=issued_at,
    )
    payload = validate_clip_media_token(
        api_key="secret-key",
        token=token,
        clip_id="clip-1",
        now=issued_at,
    )

    # Then payload is accepted with expected expiry metadata
    assert payload.clip_id == "clip-1"
    assert payload.scope == "clip_media"
    assert expires_at > issued_at


def test_validate_clip_media_token_rejects_mismatched_clip_id() -> None:
    """Token should be rejected when requested clip does not match."""
    # Given a token issued for clip-1
    issued_at = datetime(2026, 2, 15, 12, 0, tzinfo=UTC)
    token, _ = issue_clip_media_token(
        api_key="secret-key",
        clip_id="clip-1",
        ttl_s=600,
        now=issued_at,
    )

    # When validating token for a different clip
    with pytest.raises(MediaTokenError):
        validate_clip_media_token(
            api_key="secret-key",
            token=token,
            clip_id="clip-2",
            now=issued_at,
        )

    # Then validation fails


def test_validate_clip_media_token_rejects_expired_token() -> None:
    """Token should be rejected once it has expired."""
    # Given a short-lived token
    issued_at = datetime(2026, 2, 15, 12, 0, tzinfo=UTC)
    token, _ = issue_clip_media_token(
        api_key="secret-key",
        clip_id="clip-1",
        ttl_s=60,
        now=issued_at,
    )

    # When validating after expiry
    with pytest.raises(MediaTokenError):
        validate_clip_media_token(
            api_key="secret-key",
            token=token,
            clip_id="clip-1",
            now=datetime(2026, 2, 15, 12, 2, tzinfo=UTC),
        )

    # Then validation fails


def test_validate_clip_media_token_rejects_tampered_signature() -> None:
    """Token should be rejected if payload is altered without re-signing."""
    # Given a valid token
    issued_at = datetime(2026, 2, 15, 12, 0, tzinfo=UTC)
    token, _ = issue_clip_media_token(
        api_key="secret-key",
        clip_id="clip-1",
        ttl_s=600,
        now=issued_at,
    )

    # When signature segment is tampered
    token_parts = token.split(".")
    assert len(token_parts) == 3
    tampered_token = f"{token_parts[0]}.{token_parts[1]}.invalidsignature"

    with pytest.raises(MediaTokenError):
        validate_clip_media_token(
            api_key="secret-key",
            token=tampered_token,
            clip_id="clip-1",
            now=issued_at,
        )

    # Then validation fails
