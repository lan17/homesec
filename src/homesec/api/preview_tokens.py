"""Preview token creation and verification helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

TOKEN_VERSION = "v1"
TOKEN_SCOPE = "camera_preview"
DEFAULT_PREVIEW_TOKEN_TTL_S = 60
_SIGNING_CONTEXT = b"homesec-preview-token:v1"


class PreviewTokenErrorCode(StrEnum):
    MALFORMED = "MALFORMED"
    INVALID_SIGNATURE = "INVALID_SIGNATURE"
    CAMERA_MISMATCH = "CAMERA_MISMATCH"
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    EXPIRED = "EXPIRED"


class PreviewTokenError(ValueError):
    """Raised when preview token validation fails."""

    def __init__(self, code: PreviewTokenErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class PreviewTokenPayload:
    camera_name: str
    scope: str
    exp: int


def issue_camera_preview_token(
    *,
    api_key: str,
    camera_name: str,
    ttl_s: int = DEFAULT_PREVIEW_TOKEN_TTL_S,
    now: datetime | None = None,
) -> tuple[str, datetime]:
    """Issue a signed short-lived token for camera preview playback."""
    issued_at = now or datetime.now(UTC)
    expiry_dt = issued_at + timedelta(seconds=ttl_s)
    payload = PreviewTokenPayload(
        camera_name=camera_name,
        scope=TOKEN_SCOPE,
        exp=int(expiry_dt.timestamp()),
    )

    payload_json = json.dumps(
        {
            "camera_name": payload.camera_name,
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


def validate_camera_preview_token(
    *,
    api_key: str,
    token: str,
    camera_name: str,
    now: datetime | None = None,
) -> PreviewTokenPayload:
    """Validate a camera preview token and return decoded payload."""
    token_parts = token.split(".")
    if len(token_parts) != 3:
        raise PreviewTokenError(PreviewTokenErrorCode.MALFORMED, "Token format is invalid")

    version, payload_segment, signature_segment = token_parts
    if version != TOKEN_VERSION:
        raise PreviewTokenError(PreviewTokenErrorCode.MALFORMED, "Token version is invalid")

    expected_signature = _base64url_encode(_sign(api_key, _signing_input(payload_segment)))
    if not hmac.compare_digest(signature_segment, expected_signature):
        raise PreviewTokenError(
            PreviewTokenErrorCode.INVALID_SIGNATURE,
            "Token signature is invalid",
        )

    payload = _decode_payload(payload_segment)
    if payload.scope != TOKEN_SCOPE:
        raise PreviewTokenError(
            PreviewTokenErrorCode.SCOPE_MISMATCH,
            "Token scope is invalid",
        )
    if payload.camera_name != camera_name:
        raise PreviewTokenError(
            PreviewTokenErrorCode.CAMERA_MISMATCH,
            "Token camera name is invalid",
        )

    now_ts = int((now or datetime.now(UTC)).timestamp())
    if payload.exp <= now_ts:
        raise PreviewTokenError(PreviewTokenErrorCode.EXPIRED, "Token has expired")
    return payload


def _decode_payload(payload_segment: str) -> PreviewTokenPayload:
    try:
        raw_payload = _base64url_decode(payload_segment)
        payload_obj = json.loads(raw_payload.decode("utf-8"))
    except Exception as exc:
        raise PreviewTokenError(
            PreviewTokenErrorCode.MALFORMED,
            "Token payload is malformed",
        ) from exc

    if not isinstance(payload_obj, dict):
        raise PreviewTokenError(
            PreviewTokenErrorCode.MALFORMED,
            "Token payload is malformed",
        )

    camera_name = payload_obj.get("camera_name")
    scope = payload_obj.get("scope")
    exp = payload_obj.get("exp")
    if not isinstance(camera_name, str) or not isinstance(scope, str) or not isinstance(exp, int):
        raise PreviewTokenError(
            PreviewTokenErrorCode.MALFORMED,
            "Token payload is malformed",
        )

    return PreviewTokenPayload(camera_name=camera_name, scope=scope, exp=exp)


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
