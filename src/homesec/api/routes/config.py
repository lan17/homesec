"""Configuration endpoints."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from homesec.api.dependencies import get_homesec_app
from homesec.api.redaction import is_sensitive_key, redact_config, redact_url_credentials

if TYPE_CHECKING:
    from homesec.app import Application

router = APIRouter(tags=["config"])


class ConfigResponse(BaseModel):
    """Returns the full config (secrets shown as env var names, not values)."""

    config: dict[str, object]


# Backwards-compatible aliases used by existing tests and route-local callers.
_redact_url_credentials = redact_url_credentials
_is_sensitive_key = is_sensitive_key
_redact_config = redact_config


@router.get("/api/v1/config", response_model=ConfigResponse)
async def get_config(app: Application = Depends(get_homesec_app)) -> ConfigResponse:
    """Return full configuration."""
    config = await asyncio.to_thread(app.config_manager.get_config)
    payload = config.model_dump(mode="json")
    redacted = _redact_config(payload)
    if not isinstance(redacted, dict):
        return ConfigResponse(config={})
    return ConfigResponse(config=redacted)
