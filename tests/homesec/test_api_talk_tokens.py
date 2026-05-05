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
    issued_at = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)

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

    assert payload.camera_name == "front-door"
    assert payload.session_id == "session-1"
    assert payload.scope == "camera_talk_stream"
    assert expires_at > issued_at


def test_validate_camera_talk_token_rejects_camera_mismatch() -> None:
    """Talk token should be rejected for another camera."""
    issued_at = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)
    token, _ = issue_camera_talk_token(
        api_key="secret-key",
        camera_name="front-door",
        session_id="session-1",
        now=issued_at,
    )

    with pytest.raises(TalkTokenError) as exc_info:
        validate_camera_talk_token(
            api_key="secret-key",
            token=token,
            camera_name="back-yard",
            session_id="session-1",
            now=issued_at,
        )

    assert exc_info.value.code is TalkTokenErrorCode.CAMERA_MISMATCH


def test_validate_camera_talk_token_rejects_session_mismatch() -> None:
    """Talk token should be rejected for another session on the same camera."""
    issued_at = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)
    token, _ = issue_camera_talk_token(
        api_key="secret-key",
        camera_name="front-door",
        session_id="session-1",
        now=issued_at,
    )

    with pytest.raises(TalkTokenError) as exc_info:
        validate_camera_talk_token(
            api_key="secret-key",
            token=token,
            camera_name="front-door",
            session_id="session-2",
            now=issued_at,
        )

    assert exc_info.value.code is TalkTokenErrorCode.SESSION_MISMATCH


def test_validate_camera_talk_token_rejects_expired_token() -> None:
    """Talk token should be rejected after expiry."""
    issued_at = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)
    token, _ = issue_camera_talk_token(
        api_key="secret-key",
        camera_name="front-door",
        session_id="session-1",
        ttl_s=10,
        now=issued_at,
    )

    with pytest.raises(TalkTokenError) as exc_info:
        validate_camera_talk_token(
            api_key="secret-key",
            token=token,
            camera_name="front-door",
            session_id="session-1",
            now=datetime(2026, 4, 22, 12, 1, tzinfo=UTC),
        )

    assert exc_info.value.code is TalkTokenErrorCode.EXPIRED


def test_validate_camera_talk_token_rejects_scope_mismatch() -> None:
    """Talk token should reject payloads with another signed scope."""
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

    with pytest.raises(TalkTokenError) as exc_info:
        validate_camera_talk_token(
            api_key="secret-key",
            token=forged_token,
            camera_name="front-door",
            session_id="session-1",
            now=issued_at,
        )

    assert exc_info.value.code is TalkTokenErrorCode.SCOPE_MISMATCH


def test_issue_camera_talk_token_reports_actual_validation_expiry_boundary() -> None:
    """Talk token expiry metadata should match the validation cutoff."""
    issued_at = datetime(2026, 4, 22, 12, 0, 0, 900000, tzinfo=UTC)

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

    with pytest.raises(TalkTokenError) as exc_info:
        validate_camera_talk_token(
            api_key="secret-key",
            token=token,
            camera_name="front-door",
            session_id="session-1",
            now=expires_at,
        )

    assert exc_info.value.code is TalkTokenErrorCode.EXPIRED
