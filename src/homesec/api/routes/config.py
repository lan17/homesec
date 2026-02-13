"""Configuration endpoints."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from urllib.parse import SplitResult, urlsplit, urlunsplit

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from homesec.api.dependencies import get_homesec_app

if TYPE_CHECKING:
    from homesec.app import Application

router = APIRouter(tags=["config"])
REDACTED = "***redacted***"


class ConfigResponse(BaseModel):
    """Returns the full config (secrets shown as env var names, not values)."""

    config: dict[str, object]


def _redact_url_credentials(url: str) -> str:
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


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower()
    if normalized.endswith("_env"):
        return False
    return any(
        token in normalized
        for token in ("password", "secret", "token", "api_key", "credential", "dsn")
    )


def _redact_config(value: object, *, key: str | None = None) -> object:
    if isinstance(value, dict):
        return {
            nested_key: _redact_config(nested_value, key=nested_key)
            for nested_key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_redact_config(item, key=key) for item in value]
    if isinstance(value, str):
        if key is not None and _is_sensitive_key(key):
            return REDACTED
        if key is not None and "url" in key.lower():
            return _redact_url_credentials(value)
    return value


@router.get("/api/v1/config", response_model=ConfigResponse)
async def get_config(app: Application = Depends(get_homesec_app)) -> ConfigResponse:
    """Return full configuration."""
    config = await asyncio.to_thread(app.config_manager.get_config)
    payload = config.model_dump(mode="json")
    redacted = _redact_config(payload)
    if not isinstance(redacted, dict):
        return ConfigResponse(config={})
    return ConfigResponse(config=redacted)
