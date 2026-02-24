"""Shared redaction helpers for API responses."""

from __future__ import annotations

from urllib.parse import SplitResult, urlsplit, urlunsplit

REDACTED = "***redacted***"
_SENSITIVE_KEY_TOKENS = (
    "password",
    "secret",
    "token",
    "api_key",
    "credential",
    "dsn",
    "private_key",
    "connection_string",
    "passphrase",
    "bearer",
)


def redact_url_credentials(url: str) -> str:
    """Redact username/password components from a URL string."""
    parts = urlsplit(url)
    if parts.username is None and parts.password is None:
        return url

    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"

    redacted = SplitResult(
        scheme=parts.scheme,
        netloc=f"{REDACTED}@{host}",
        path=parts.path,
        query=parts.query,
        fragment=parts.fragment,
    )
    return urlunsplit(redacted)


def is_sensitive_key(key: str) -> bool:
    """Return True when a payload key should be redacted."""
    normalized = key.lower()
    if normalized.endswith("_env"):
        return False
    return any(token in normalized for token in _SENSITIVE_KEY_TOKENS)


def redact_config(value: object, *, key: str | None = None) -> object:
    """Recursively redact sensitive values from config-like payloads."""
    if isinstance(value, dict):
        return {
            nested_key: redact_config(nested_value, key=nested_key)
            for nested_key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [redact_config(item, key=key) for item in value]
    if isinstance(value, str):
        if key is not None and is_sensitive_key(key):
            return REDACTED
        if key is not None and "url" in key.lower():
            return redact_url_credentials(value)
    return value
