"""Unit tests for media token helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import homesec.api.media_tokens as media_tokens
from homesec.api.media_tokens import (
    MediaTokenError,
    MediaTokenErrorCode,
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


def test_validate_clip_media_token_rejects_invalid_version() -> None:
    """Token should be rejected when version prefix is not supported."""
    # Given a valid token issued for clip-1
    issued_at = datetime(2026, 2, 15, 12, 0, tzinfo=UTC)
    token, _ = issue_clip_media_token(
        api_key="secret-key",
        clip_id="clip-1",
        ttl_s=600,
        now=issued_at,
    )

    # When mutating the token version segment
    token_parts = token.split(".")
    assert len(token_parts) == 3
    invalid_version_token = f"v2.{token_parts[1]}.{token_parts[2]}"

    # Then validation fails with malformed-version classification
    with pytest.raises(MediaTokenError) as exc_info:
        validate_clip_media_token(
            api_key="secret-key",
            token=invalid_version_token,
            clip_id="clip-1",
            now=issued_at,
        )
    assert exc_info.value.code is MediaTokenErrorCode.MALFORMED


def test_validate_clip_media_token_rejects_scope_mismatch() -> None:
    """Token should be rejected when scope does not match clip-media scope."""
    # Given a forged token payload signed with a non-media scope
    issued_at = datetime(2026, 2, 15, 12, 0, tzinfo=UTC)
    payload_json = b'{"clip_id":"clip-1","exp":1897396800,"scope":"other_scope"}'
    payload_segment = media_tokens._base64url_encode(payload_json)
    signature = media_tokens._base64url_encode(
        media_tokens._sign("secret-key", media_tokens._signing_input(payload_segment))
    )
    scope_mismatch_token = f"{media_tokens.TOKEN_VERSION}.{payload_segment}.{signature}"

    # When validating token for clip-1
    with pytest.raises(MediaTokenError) as exc_info:
        validate_clip_media_token(
            api_key="secret-key",
            token=scope_mismatch_token,
            clip_id="clip-1",
            now=issued_at,
        )

    # Then validation fails with scope mismatch
    assert exc_info.value.code is MediaTokenErrorCode.SCOPE_MISMATCH


def test_validate_clip_media_token_rejects_non_json_payload() -> None:
    """Token should be rejected when payload is not valid JSON."""
    # Given a signed token whose payload is not JSON
    issued_at = datetime(2026, 2, 15, 12, 0, tzinfo=UTC)
    payload_segment = media_tokens._base64url_encode(b"not-json")
    signature = media_tokens._base64url_encode(
        media_tokens._sign("secret-key", media_tokens._signing_input(payload_segment))
    )
    malformed_token = f"{media_tokens.TOKEN_VERSION}.{payload_segment}.{signature}"

    # When validating the token
    with pytest.raises(MediaTokenError) as exc_info:
        validate_clip_media_token(
            api_key="secret-key",
            token=malformed_token,
            clip_id="clip-1",
            now=issued_at,
        )

    # Then validation fails with malformed payload classification
    assert exc_info.value.code is MediaTokenErrorCode.MALFORMED


def test_validate_clip_media_token_rejects_non_object_payload() -> None:
    """Token should be rejected when decoded payload is not an object."""
    # Given a signed token with a JSON array payload
    issued_at = datetime(2026, 2, 15, 12, 0, tzinfo=UTC)
    payload_segment = media_tokens._base64url_encode(b'["clip-1"]')
    signature = media_tokens._base64url_encode(
        media_tokens._sign("secret-key", media_tokens._signing_input(payload_segment))
    )
    malformed_token = f"{media_tokens.TOKEN_VERSION}.{payload_segment}.{signature}"

    # When validating the token
    with pytest.raises(MediaTokenError) as exc_info:
        validate_clip_media_token(
            api_key="secret-key",
            token=malformed_token,
            clip_id="clip-1",
            now=issued_at,
        )

    # Then validation fails with malformed payload classification
    assert exc_info.value.code is MediaTokenErrorCode.MALFORMED


def test_validate_clip_media_token_rejects_wrong_payload_types() -> None:
    """Token should be rejected when payload field types are invalid."""
    # Given a signed token with invalid payload field types
    issued_at = datetime(2026, 2, 15, 12, 0, tzinfo=UTC)
    payload_json = b'{"clip_id":1,"scope":"clip_media","exp":"bad"}'
    payload_segment = media_tokens._base64url_encode(payload_json)
    signature = media_tokens._base64url_encode(
        media_tokens._sign("secret-key", media_tokens._signing_input(payload_segment))
    )
    malformed_token = f"{media_tokens.TOKEN_VERSION}.{payload_segment}.{signature}"

    # When validating the token
    with pytest.raises(MediaTokenError) as exc_info:
        validate_clip_media_token(
            api_key="secret-key",
            token=malformed_token,
            clip_id="clip-1",
            now=issued_at,
        )

    # Then validation fails with malformed payload classification
    assert exc_info.value.code is MediaTokenErrorCode.MALFORMED
