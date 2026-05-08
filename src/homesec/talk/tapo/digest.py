"""HTTP Digest helpers for the Tapo local talk endpoint."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Literal


class TapoDigestError(ValueError):
    """Raised when a Tapo Digest challenge cannot be parsed or used."""


@dataclass(frozen=True, slots=True)
class TapoDigestChallenge:
    """Parsed Digest challenge from the Tapo local stream endpoint."""

    scheme: str
    params: dict[str, str]

    @property
    def realm(self) -> str | None:
        """Return the Digest realm."""
        return self.params.get("realm")

    @property
    def nonce(self) -> str | None:
        """Return the Digest nonce."""
        return self.params.get("nonce")

    @property
    def encrypt_type(self) -> str | None:
        """Return the Tapo encryption hint when provided."""
        return self.params.get("encrypt_type")

    @property
    def preferred_hash_kind(self) -> Literal["sha256", "md5"]:
        """Return the preferred credential hash kind for this challenge."""
        return "sha256" if self.encrypt_type == "3" else "md5"


def parse_www_authenticate(header: str) -> TapoDigestChallenge:
    """Parse a Tapo WWW-Authenticate Digest challenge."""
    stripped = header.strip()
    scheme, separator, rest = stripped.partition(" ")
    if not separator:
        return TapoDigestChallenge(scheme=stripped.lower(), params={})
    return TapoDigestChallenge(
        scheme=scheme.lower(),
        params=_parse_auth_params(rest),
    )


def parse_digest_authorization(header: str) -> dict[str, str]:
    """Parse a Digest Authorization header into lower-case keys."""
    stripped = header.strip()
    if stripped.lower().startswith("digest "):
        stripped = stripped.partition(" ")[2]
    return _parse_auth_params(stripped)


def build_digest_authorization_header(
    *,
    challenge: TapoDigestChallenge,
    method: str,
    uri: str,
    username: str,
    password_material: str,
    nonce_count: int = 1,
    cnonce: str = "homesec",
) -> str:
    """Build a Digest Authorization header for the Tapo local endpoint."""
    if challenge.scheme != "digest":
        raise TapoDigestError("Tapo local endpoint requires Digest authentication")

    params = challenge.params
    realm = params.get("realm")
    nonce = params.get("nonce")
    if not realm or not nonce:
        raise TapoDigestError("Tapo Digest challenge requires realm and nonce")

    algorithm = params.get("algorithm", "MD5")
    if algorithm.upper() != "MD5":
        raise TapoDigestError("Tapo Digest challenge uses an unsupported algorithm")

    qop = _select_qop(params.get("qop"))
    nc_value = f"{nonce_count:08x}"
    ha1 = _md5_hex(f"{username}:{realm}:{password_material}")
    ha2 = _md5_hex(f"{method}:{uri}")
    if qop is None:
        response = _md5_hex(f"{ha1}:{nonce}:{ha2}")
    else:
        response = _md5_hex(f"{ha1}:{nonce}:{nc_value}:{cnonce}:{qop}:{ha2}")

    items: list[tuple[str, str, bool]] = [
        ("username", username, True),
        ("realm", realm, True),
        ("nonce", nonce, True),
        ("uri", uri, True),
        ("response", response, True),
        ("algorithm", algorithm, False),
    ]
    opaque = params.get("opaque")
    if opaque is not None:
        items.append(("opaque", opaque, True))
    if qop is not None:
        items.extend(
            [
                ("qop", qop, False),
                ("nc", nc_value, False),
                ("cnonce", cnonce, True),
            ]
        )

    rendered: list[str] = []
    for key, value, should_quote in items:
        if should_quote:
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            rendered.append(f'{key}="{escaped}"')
        else:
            rendered.append(f"{key}={value}")
    return "Digest " + ", ".join(rendered)


def digest_authorization_matches(
    *,
    authorization_header: str | None,
    challenge: TapoDigestChallenge,
    method: str,
    uri: str,
    username: str,
    password_material: str,
) -> bool:
    """Return whether an Authorization header matches the expected Digest response."""
    if authorization_header is None or not authorization_header.lower().startswith("digest "):
        return False
    actual = parse_digest_authorization(authorization_header)
    if actual.get("username") != username:
        return False
    if actual.get("realm") != challenge.realm or actual.get("nonce") != challenge.nonce:
        return False
    if actual.get("uri") != uri:
        return False

    qop = actual.get("qop")
    nc = actual.get("nc")
    cnonce = actual.get("cnonce")
    try:
        if qop is not None:
            if qop != "auth":
                return False
            if nc is None or cnonce is None:
                return False
            try:
                nonce_count = int(nc, 16)
            except ValueError:
                return False
            if nonce_count <= 0:
                return False
            expected = build_digest_authorization_header(
                challenge=challenge,
                method=method,
                uri=uri,
                username=username,
                password_material=password_material,
                nonce_count=nonce_count,
                cnonce=cnonce,
            )
        else:
            expected = build_digest_authorization_header(
                challenge=challenge,
                method=method,
                uri=uri,
                username=username,
                password_material=password_material,
            )
    except TapoDigestError:
        return False
    return actual.get("response") == parse_digest_authorization(expected).get("response")


def _parse_auth_params(value: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for match in re.finditer(r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|[^,]+)', value):
        key = match.group(1).lower()
        raw = match.group(2).strip()
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        params[key] = raw
    return params


def _select_qop(qop_value: str | None) -> str | None:
    if qop_value is None:
        return None
    options = [item.strip().lower() for item in qop_value.split(",")]
    if "auth" in options:
        return "auth"
    raise TapoDigestError("Tapo Digest challenge uses an unsupported qop")


def _md5_hex(value: str) -> str:
    return hashlib.md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()
