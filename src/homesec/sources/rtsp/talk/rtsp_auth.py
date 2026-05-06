"""RTSP authentication helpers with credential-safe formatting."""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass
from urllib.parse import quote, unquote, urlsplit, urlunsplit


@dataclass(frozen=True, slots=True)
class RTSPCredentials:
    """Username/password credentials extracted from configuration or URL userinfo."""

    username: str
    password: str


@dataclass(frozen=True, slots=True)
class RTSPAuthChallenge:
    """Parsed WWW-Authenticate challenge."""

    scheme: str
    params: dict[str, str]

    @property
    def realm(self) -> str | None:
        return self.params.get("realm")


def parse_www_authenticate(header: str) -> RTSPAuthChallenge:
    """Parse a Basic or Digest WWW-Authenticate challenge."""
    stripped = header.strip()
    scheme, separator, rest = stripped.partition(" ")
    if not separator:
        return RTSPAuthChallenge(scheme=stripped.lower(), params={})
    return RTSPAuthChallenge(scheme=scheme.lower(), params=_parse_auth_params(rest))


def build_authorization_header(
    *,
    challenge: RTSPAuthChallenge,
    method: str,
    uri: str,
    credentials: RTSPCredentials,
    nonce_count: int = 1,
    cnonce: str = "homesec",
) -> str:
    """Build an Authorization header for Basic or Digest RTSP auth."""
    if challenge.scheme == "basic":
        raw = f"{credentials.username}:{credentials.password}".encode()
        return "Basic " + base64.b64encode(raw).decode("ascii")
    if challenge.scheme == "digest":
        return _build_digest_authorization_header(
            challenge=challenge,
            method=method,
            uri=uri,
            credentials=credentials,
            nonce_count=nonce_count,
            cnonce=cnonce,
        )
    raise ValueError(f"Unsupported RTSP auth scheme: {challenge.scheme}")


def redact_rtsp_url(url: str) -> str:
    """Return an RTSP URL with username/password redacted."""
    parsed = urlsplit(url)
    if parsed.username is None and parsed.password is None:
        return url
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))


def split_rtsp_url_credentials(url: str) -> tuple[str, RTSPCredentials | None]:
    """Remove userinfo from a URL and return extracted credentials if present."""
    parsed = urlsplit(url)
    if parsed.username is None:
        return url, None
    password = unquote(parsed.password or "")
    credentials = RTSPCredentials(username=unquote(parsed.username), password=password)
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    clean = urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))
    return clean, credentials


def request_uri_without_credentials(url: str) -> str:
    """Normalize an RTSP request URI so credentials never cross logs/wire userinfo."""
    clean, _credentials = split_rtsp_url_credentials(url)
    return clean


def quote_digest_value(value: str) -> str:
    """Quote a Digest auth value for deterministic tests."""
    return quote(value, safe="")


def _build_digest_authorization_header(
    *,
    challenge: RTSPAuthChallenge,
    method: str,
    uri: str,
    credentials: RTSPCredentials,
    nonce_count: int,
    cnonce: str,
) -> str:
    params = challenge.params
    realm = params.get("realm")
    nonce = params.get("nonce")
    if not realm or not nonce:
        raise ValueError("Digest challenge requires realm and nonce")

    algorithm = params.get("algorithm", "MD5")
    if algorithm.upper() not in {"MD5", "MD5-SESS"}:
        raise ValueError(f"Unsupported Digest algorithm: {algorithm}")

    username = credentials.username
    ha1 = _md5_hex(f"{username}:{realm}:{credentials.password}")
    if algorithm.upper() == "MD5-SESS":
        ha1 = _md5_hex(f"{ha1}:{nonce}:{cnonce}")
    ha2 = _md5_hex(f"{method}:{uri}")

    qop = _select_qop(params.get("qop"))
    nc_value = f"{nonce_count:08x}"
    if qop:
        response = _md5_hex(f"{ha1}:{nonce}:{nc_value}:{cnonce}:{qop}:{ha2}")
    else:
        response = _md5_hex(f"{ha1}:{nonce}:{ha2}")

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
    if qop:
        items.extend(
            [
                ("qop", qop, False),
                ("nc", nc_value, False),
                ("cnonce", cnonce, True),
            ]
        )

    rendered = []
    for key, value, should_quote in items:
        if should_quote:
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            rendered.append(f'{key}="{escaped}"')
        else:
            rendered.append(f"{key}={value}")
    return "Digest " + ", ".join(rendered)


def _parse_auth_params(value: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for match in re.finditer(r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|[^,]+)', value):
        key = match.group(1).lower()
        raw = match.group(2).strip()
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1]
            raw = raw.replace('\\"', '"').replace("\\\\", "\\")
        params[key] = raw
    return params


def _select_qop(qop_value: str | None) -> str | None:
    if not qop_value:
        return None
    options = [item.strip().lower() for item in qop_value.split(",")]
    if "auth" in options:
        return "auth"
    return None


def _md5_hex(value: str) -> str:
    return hashlib.md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()
