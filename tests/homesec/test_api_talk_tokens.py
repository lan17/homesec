"""Unit tests for talk stream token helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import homesec.api.talk_tokens as talk_tokens
from homesec.api.talk_tokens import (
    TalkTokenError,
    TalkTokenErrorCode,
    issue_camera_talk_token,
    validate_camera_talk_token,
)


def test_issue_and_validate_camera_talk_token_round_trip() -> None:
    """Issued talk token should validate for the same camera/session pair."""
    # Given: A fixed issuance time and matching camera/session identity
    issued_at = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)

    # When: Issuing and validating a camera talk stream token
    token, expires_at = issue_camera_talk_token(
        api_key="secret-key",
        camera_name="front-door",
        session_id="session-1",
        ttl_s=60,
        now=issued_at,
    )
    payload = validate_camera_talk_token(
        api_key="secret-key",
        token=token,
        camera_name="front-door",
        session_id="session-1",
        now=issued_at,
    )

    # Then: The validated payload is bound to the requested camera/session
    assert payload.camera_name == "front-door"
    assert payload.session_id == "session-1"
    assert payload.scope == "camera_talk_stream"
    assert expires_at > issued_at


def test_validate_camera_talk_token_rejects_camera_mismatch() -> None:
    """Talk token should be rejected for another camera."""
    # Given: A token issued for the front-door camera
    issued_at = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)
    token, _ = issue_camera_talk_token(
        api_key="secret-key",
        camera_name="front-door",
        session_id="session-1",
        now=issued_at,
    )

    # When: Validating the token for a different camera
    with pytest.raises(TalkTokenError) as exc_info:
        validate_camera_talk_token(
            api_key="secret-key",
            token=token,
            camera_name="back-yard",
            session_id="session-1",
            now=issued_at,
        )

    # Then: Validation fails with a stable camera-mismatch code
    assert exc_info.value.code is TalkTokenErrorCode.CAMERA_MISMATCH


def test_validate_camera_talk_token_rejects_session_mismatch() -> None:
    """Talk token should be rejected for another session on the same camera."""
    # Given: A token issued for one talk session
    issued_at = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)
    token, _ = issue_camera_talk_token(
        api_key="secret-key",
        camera_name="front-door",
        session_id="session-1",
        now=issued_at,
    )

    # When: Validating the token against another session on the same camera
    with pytest.raises(TalkTokenError) as exc_info:
        validate_camera_talk_token(
            api_key="secret-key",
            token=token,
            camera_name="front-door",
            session_id="session-2",
            now=issued_at,
        )

    # Then: Validation fails with a stable session-mismatch code
    assert exc_info.value.code is TalkTokenErrorCode.SESSION_MISMATCH


def test_validate_camera_talk_token_rejects_expired_token() -> None:
    """Talk token should be rejected after expiry."""
    # Given: A short-lived token issued at a fixed time
    issued_at = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)
    token, _ = issue_camera_talk_token(
        api_key="secret-key",
        camera_name="front-door",
        session_id="session-1",
        ttl_s=10,
        now=issued_at,
    )

    # When: Validating the token after its expiry boundary
    with pytest.raises(TalkTokenError) as exc_info:
        validate_camera_talk_token(
            api_key="secret-key",
            token=token,
            camera_name="front-door",
            session_id="session-1",
            now=datetime(2026, 4, 22, 12, 1, tzinfo=UTC),
        )

    # Then: Validation fails with a stable expired-token code
    assert exc_info.value.code is TalkTokenErrorCode.EXPIRED


def test_validate_camera_talk_token_rejects_scope_mismatch() -> None:
    """Talk token should reject payloads with another signed scope."""
    # Given: A correctly signed token payload with the wrong purpose scope
    issued_at = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)
    payload_json = (
        b'{"camera_name":"front-door","exp":1897396800,'
        b'"scope":"other_scope","session_id":"session-1"}'
    )
    payload_segment = talk_tokens._base64url_encode(payload_json)
    signature = talk_tokens._base64url_encode(
        talk_tokens._sign("secret-key", talk_tokens._signing_input(payload_segment))
    )
    forged_token = f"{talk_tokens.TOKEN_VERSION}.{payload_segment}.{signature}"

    # When: Validating that token as a camera talk stream token
    with pytest.raises(TalkTokenError) as exc_info:
        validate_camera_talk_token(
            api_key="secret-key",
            token=forged_token,
            camera_name="front-door",
            session_id="session-1",
            now=issued_at,
        )

    # Then: Validation fails with a stable scope-mismatch code
    assert exc_info.value.code is TalkTokenErrorCode.SCOPE_MISMATCH


def test_issue_camera_talk_token_reports_actual_validation_expiry_boundary() -> None:
    """Talk token expiry metadata should match the validation cutoff."""
    # Given: A token issued close to the next whole-second expiry boundary
    issued_at = datetime(2026, 4, 22, 12, 0, 0, 900000, tzinfo=UTC)

    # When: Issuing a one-second token and validating immediately before expiry
    token, expires_at = issue_camera_talk_token(
        api_key="secret-key",
        camera_name="front-door",
        session_id="session-1",
        ttl_s=1,
        now=issued_at,
    )

    assert expires_at == datetime(2026, 4, 22, 12, 0, 1, tzinfo=UTC)
    payload = validate_camera_talk_token(
        api_key="secret-key",
        token=token,
        camera_name="front-door",
        session_id="session-1",
        now=expires_at - timedelta(microseconds=1),
    )
    assert payload.exp == int(expires_at.timestamp())

    # Then: Validation rejects the token exactly at the published expiry time
    with pytest.raises(TalkTokenError) as exc_info:
        validate_camera_talk_token(
            api_key="secret-key",
            token=token,
            camera_name="front-door",
            session_id="session-1",
            now=expires_at,
        )

    assert exc_info.value.code is TalkTokenErrorCode.EXPIRED
