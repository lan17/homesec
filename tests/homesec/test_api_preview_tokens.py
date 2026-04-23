"""Unit tests for preview token helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import homesec.api.preview_tokens as preview_tokens
from homesec.api.preview_tokens import (
    PreviewTokenError,
    PreviewTokenErrorCode,
    issue_camera_preview_token,
    validate_camera_preview_token,
)


def test_issue_and_validate_camera_preview_token_round_trip() -> None:
    """Issued preview token should validate for the same camera."""
    # Given: A camera name and API key
    issued_at = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)

    # When: Issuing and validating a preview token
    token, expires_at = issue_camera_preview_token(
        api_key="secret-key",
        camera_name="front-door",
        ttl_s=60,
        now=issued_at,
    )
    payload = validate_camera_preview_token(
        api_key="secret-key",
        token=token,
        camera_name="front-door",
        now=issued_at,
    )

    # Then: The payload round-trips with preview scope
    assert payload.camera_name == "front-door"
    assert payload.scope == "camera_preview"
    assert expires_at > issued_at


def test_validate_camera_preview_token_rejects_camera_mismatch() -> None:
    """Preview token should be rejected for another camera."""
    # Given: A token issued for front-door
    issued_at = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)
    token, _ = issue_camera_preview_token(
        api_key="secret-key",
        camera_name="front-door",
        now=issued_at,
    )

    # When: Validating the token for another camera
    with pytest.raises(PreviewTokenError) as exc_info:
        validate_camera_preview_token(
            api_key="secret-key",
            token=token,
            camera_name="back-yard",
            now=issued_at,
        )

    # Then: Validation fails with camera mismatch
    assert exc_info.value.code is PreviewTokenErrorCode.CAMERA_MISMATCH


def test_validate_camera_preview_token_rejects_expired_token() -> None:
    """Preview token should be rejected after expiry."""
    # Given: A short-lived preview token
    issued_at = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)
    token, _ = issue_camera_preview_token(
        api_key="secret-key",
        camera_name="front-door",
        ttl_s=10,
        now=issued_at,
    )

    # When: Validating after the token expires
    with pytest.raises(PreviewTokenError) as exc_info:
        validate_camera_preview_token(
            api_key="secret-key",
            token=token,
            camera_name="front-door",
            now=datetime(2026, 4, 22, 12, 1, tzinfo=UTC),
        )

    # Then: Validation fails with expiry classification
    assert exc_info.value.code is PreviewTokenErrorCode.EXPIRED


def test_validate_camera_preview_token_rejects_scope_mismatch() -> None:
    """Preview token should reject payloads with another scope."""
    # Given: A forged token signed with a non-preview scope
    issued_at = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)
    payload_json = b'{"camera_name":"front-door","exp":1897396800,"scope":"other_scope"}'
    payload_segment = preview_tokens._base64url_encode(payload_json)
    signature = preview_tokens._base64url_encode(
        preview_tokens._sign("secret-key", preview_tokens._signing_input(payload_segment))
    )
    forged_token = f"{preview_tokens.TOKEN_VERSION}.{payload_segment}.{signature}"

    # When: Validating the forged token
    with pytest.raises(PreviewTokenError) as exc_info:
        validate_camera_preview_token(
            api_key="secret-key",
            token=forged_token,
            camera_name="front-door",
            now=issued_at,
        )

    # Then: Validation fails with scope mismatch
    assert exc_info.value.code is PreviewTokenErrorCode.SCOPE_MISMATCH


def test_issue_camera_preview_token_reports_actual_validation_expiry_boundary() -> None:
    """Preview token expiry metadata should match the server-side validation cutoff."""
    # Given: A sub-second issue time that would otherwise advertise extra token lifetime
    issued_at = datetime(2026, 4, 22, 12, 0, 0, 900000, tzinfo=UTC)

    # When: Issuing a one-second preview token
    token, expires_at = issue_camera_preview_token(
        api_key="secret-key",
        camera_name="front-door",
        ttl_s=1,
        now=issued_at,
    )

    # Then: The advertised expiry matches the encoded exp boundary and stays valid until then
    assert expires_at == datetime(2026, 4, 22, 12, 0, 1, tzinfo=UTC)
    payload = validate_camera_preview_token(
        api_key="secret-key",
        token=token,
        camera_name="front-door",
        now=expires_at - timedelta(microseconds=1),
    )
    assert payload.exp == int(expires_at.timestamp())

    with pytest.raises(PreviewTokenError) as exc_info:
        validate_camera_preview_token(
            api_key="secret-key",
            token=token,
            camera_name="front-door",
            now=expires_at,
        )

    assert exc_info.value.code is PreviewTokenErrorCode.EXPIRED
