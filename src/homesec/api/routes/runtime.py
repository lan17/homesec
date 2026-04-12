"""Runtime control-plane endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from homesec.api.dependencies import get_homesec_app
from homesec.api.runtime_reload import RuntimeReloadResponse, request_runtime_reload
from homesec.runtime.models import RuntimeState

if TYPE_CHECKING:
    from homesec.app import Application

router = APIRouter(tags=["runtime"])


class RuntimeStatusResponse(BaseModel):
    state: RuntimeState
    generation: int
    reload_in_progress: bool
    active_config_version: str | None
    last_reload_at: datetime | None
    last_reload_error: str | None


@router.get("/api/v1/runtime/status", response_model=RuntimeStatusResponse)
async def get_runtime_status(app: Application = Depends(get_homesec_app)) -> RuntimeStatusResponse:
    """Return runtime-manager status."""
    status = app.get_runtime_status()
    return RuntimeStatusResponse(
        state=status.state,
        generation=status.generation,
        reload_in_progress=status.reload_in_progress,
        active_config_version=status.active_config_version,
        last_reload_at=status.last_reload_at,
        last_reload_error=status.last_reload_error,
    )


@router.post("/api/v1/runtime/reload", response_model=RuntimeReloadResponse, status_code=202)
async def reload_runtime(
    app: Application = Depends(get_homesec_app),
) -> RuntimeReloadResponse:
    """Trigger runtime reload and return async acceptance outcome."""
    return await request_runtime_reload(app)
