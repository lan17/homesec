"""Configuration endpoints."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from homesec.api.dependencies import get_homesec_app

if TYPE_CHECKING:
    from homesec.app import Application

router = APIRouter(tags=["config"])


class ConfigResponse(BaseModel):
    """Returns the full config (secrets shown as env var names, not values)."""

    config: dict[str, object]


@router.get("/api/v1/config", response_model=ConfigResponse)
async def get_config(app: Application = Depends(get_homesec_app)) -> ConfigResponse:
    """Return full configuration."""
    config = await asyncio.to_thread(app.config_manager.get_config)
    return ConfigResponse(config=config.model_dump(mode="json"))
