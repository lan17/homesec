"""Media token creation and verification helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

TOKEN_VERSION = "v1"
TOKEN_SCOPE = "clip_media"
DEFAULT_MEDIA_TOKEN_TTL_S = 600
_SIGNING_CONTEXT = b"homesec-media-token:v1"


class MediaTokenErrorCode(StrEnum):
    MALFORMED = "MALFORMED"
    INVALID_SIGNATURE = "INVALID_SIGNATURE"
    CLIP_MISMATCH = "CLIP_MISMATCH"
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    EXPIRED = "EXPIRED"


class MediaTokenError(ValueError):
    """Raised when media token validation fails."""

    def __init__(self, code: MediaTokenErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class MediaTokenPayload:
    clip_id: str
    scope: str
    exp: int


def issue_clip_media_token(
    *,
    api_key: str,
    clip_id: str,
    ttl_s: int = DEFAULT_MEDIA_TOKEN_TTL_S,
    now: datetime | None = None,
) -> tuple[str, datetime]:
    """Issue a signed short-lived token for clip media playback."""
    issued_at = now or datetime.now(UTC)
    expiry_dt = issued_at + timedelta(seconds=ttl_s)
    payload = MediaTokenPayload(
        clip_id=clip_id,
        scope=TOKEN_SCOPE,
        exp=int(expiry_dt.timestamp()),
    )

    payload_json = json.dumps(
        {
            "clip_id": payload.clip_id,
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
    return token, expiry_dt


def validate_clip_media_token(
    *,
    api_key: str,
    token: str,
    clip_id: str,
    now: datetime | None = None,
) -> MediaTokenPayload:
    """Validate clip media token and return decoded payload."""
    token_parts = token.split(".")
    if len(token_parts) != 3:
        raise MediaTokenError(MediaTokenErrorCode.MALFORMED, "Token format is invalid")

    version, payload_segment, signature_segment = token_parts
    if version != TOKEN_VERSION:
        raise MediaTokenError(MediaTokenErrorCode.MALFORMED, "Token version is invalid")

    expected_signature = _base64url_encode(_sign(api_key, _signing_input(payload_segment)))
    if not hmac.compare_digest(signature_segment, expected_signature):
        raise MediaTokenError(
            MediaTokenErrorCode.INVALID_SIGNATURE,
            "Token signature is invalid",
        )

    payload = _decode_payload(payload_segment)
    if payload.scope != TOKEN_SCOPE:
        raise MediaTokenError(MediaTokenErrorCode.SCOPE_MISMATCH, "Token scope is invalid")
    if payload.clip_id != clip_id:
        raise MediaTokenError(MediaTokenErrorCode.CLIP_MISMATCH, "Token clip ID is invalid")

    now_ts = int((now or datetime.now(UTC)).timestamp())
    if payload.exp <= now_ts:
        raise MediaTokenError(MediaTokenErrorCode.EXPIRED, "Token has expired")
    return payload


def _decode_payload(payload_segment: str) -> MediaTokenPayload:
    try:
        raw_payload = _base64url_decode(payload_segment)
        payload_obj = json.loads(raw_payload.decode("utf-8"))
    except Exception as exc:
        raise MediaTokenError(
            MediaTokenErrorCode.MALFORMED,
            "Token payload is malformed",
        ) from exc

    if not isinstance(payload_obj, dict):
        raise MediaTokenError(MediaTokenErrorCode.MALFORMED, "Token payload is malformed")

    clip_id = payload_obj.get("clip_id")
    scope = payload_obj.get("scope")
    exp = payload_obj.get("exp")
    if not isinstance(clip_id, str) or not isinstance(scope, str) or not isinstance(exp, int):
        raise MediaTokenError(MediaTokenErrorCode.MALFORMED, "Token payload is malformed")

    return MediaTokenPayload(clip_id=clip_id, scope=scope, exp=exp)


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
