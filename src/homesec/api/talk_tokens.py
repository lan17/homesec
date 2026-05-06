"""Talk stream token creation and verification helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

TOKEN_VERSION = "v1"
TOKEN_SCOPE = "camera_talk_stream"
DEFAULT_TALK_TOKEN_TTL_S = 30
_SIGNING_CONTEXT = b"homesec-talk-token:v1"


class TalkTokenErrorCode(StrEnum):
    """Stable reasons a talk stream token may be rejected."""

    MALFORMED = "MALFORMED"
    INVALID_SIGNATURE = "INVALID_SIGNATURE"
    CAMERA_MISMATCH = "CAMERA_MISMATCH"
    SESSION_MISMATCH = "SESSION_MISMATCH"
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    EXPIRED = "EXPIRED"


class TalkTokenError(ValueError):
    """Raised when talk stream token validation fails."""

    def __init__(self, code: TalkTokenErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class TalkTokenPayload:
    camera_name: str
    session_id: str
    scope: str
    exp: int


def issue_camera_talk_token(
    *,
    api_key: str,
    camera_name: str,
    session_id: str,
    ttl_s: int = DEFAULT_TALK_TOKEN_TTL_S,
    now: datetime | None = None,
) -> tuple[str, datetime]:
    """Issue a signed short-lived token for a camera talk WebSocket stream."""
    issued_at = now or datetime.now(UTC)
    expiry_ts = int((issued_at + timedelta(seconds=ttl_s)).timestamp())
    payload = TalkTokenPayload(
        camera_name=camera_name,
        session_id=session_id,
        scope=TOKEN_SCOPE,
        exp=expiry_ts,
    )

    payload_json = json.dumps(
        {
            "camera_name": payload.camera_name,
            "session_id": payload.session_id,
            "scope": payload.scope,
            "exp": payload.exp,
        },
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8")
    payload_segment = _base64url_encode(payload_json)
    signing_input = _signing_input(payload_segment)
    signature = _base64url_encode(_sign(api_key, signing_input))
    token = f"{TOKEN_VERSION}.{payload_segment}.{signature}"
    return token, datetime.fromtimestamp(expiry_ts, UTC)


def validate_camera_talk_token(
    *,
    api_key: str,
    token: str,
    camera_name: str,
    session_id: str,
    now: datetime | None = None,
) -> TalkTokenPayload:
    """Validate a camera talk token and return decoded payload."""
    token_parts = token.split(".")
    if len(token_parts) != 3:
        raise TalkTokenError(TalkTokenErrorCode.MALFORMED, "Token format is invalid")

    version, payload_segment, signature_segment = token_parts
    if version != TOKEN_VERSION:
        raise TalkTokenError(TalkTokenErrorCode.MALFORMED, "Token version is invalid")

    expected_signature = _base64url_encode(_sign(api_key, _signing_input(payload_segment)))
    if not hmac.compare_digest(signature_segment, expected_signature):
        raise TalkTokenError(
            TalkTokenErrorCode.INVALID_SIGNATURE,
            "Token signature is invalid",
        )

    payload = _decode_payload(payload_segment)
    if payload.scope != TOKEN_SCOPE:
        raise TalkTokenError(
            TalkTokenErrorCode.SCOPE_MISMATCH,
            "Token scope is invalid",
        )
    if payload.camera_name != camera_name:
        raise TalkTokenError(
            TalkTokenErrorCode.CAMERA_MISMATCH,
            "Token camera name is invalid",
        )
    if payload.session_id != session_id:
        raise TalkTokenError(
            TalkTokenErrorCode.SESSION_MISMATCH,
            "Token session id is invalid",
        )

    now_ts = int((now or datetime.now(UTC)).timestamp())
    if payload.exp <= now_ts:
        raise TalkTokenError(TalkTokenErrorCode.EXPIRED, "Token has expired")
    return payload


def _decode_payload(payload_segment: str) -> TalkTokenPayload:
    try:
        raw_payload = _base64url_decode(payload_segment)
        payload_obj = json.loads(raw_payload.decode("utf-8"))
    except Exception as exc:
        raise TalkTokenError(
            TalkTokenErrorCode.MALFORMED,
            "Token payload is malformed",
        ) from exc

    if not isinstance(payload_obj, dict):
        raise TalkTokenError(
            TalkTokenErrorCode.MALFORMED,
            "Token payload is malformed",
        )

    camera_name = payload_obj.get("camera_name")
    session_id = payload_obj.get("session_id")
    scope = payload_obj.get("scope")
    exp = payload_obj.get("exp")
    if (
        not isinstance(camera_name, str)
        or not isinstance(session_id, str)
        or not isinstance(scope, str)
        or not isinstance(exp, int)
    ):
        raise TalkTokenError(
            TalkTokenErrorCode.MALFORMED,
            "Token payload is malformed",
        )

    return TalkTokenPayload(
        camera_name=camera_name,
        session_id=session_id,
        scope=scope,
        exp=exp,
    )


def _derive_signing_key(api_key: str) -> bytes:
    return hmac.new(api_key.encode("utf-8"), _SIGNING_CONTEXT, hashlib.sha256).digest()


def _sign(api_key: str, message: bytes) -> bytes:
    return hmac.new(_derive_signing_key(api_key), message, hashlib.sha256).digest()


def _signing_input(payload_segment: str) -> bytes:
    return f"{TOKEN_VERSION}.{payload_segment}".encode()


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))
