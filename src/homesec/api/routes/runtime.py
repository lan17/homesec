"""Runtime control-plane endpoints."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel

from homesec.api.dependencies import RuntimeRoutesApp, get_runtime_routes_app
from homesec.api.errors import APIError, APIErrorCode
from homesec.runtime.errors import RuntimeReloadConfigError
from homesec.runtime.models import RuntimeState

router = APIRouter(tags=["runtime"])


class RuntimeStatusResponse(BaseModel):
    state: RuntimeState
    generation: int
    reload_in_progress: bool
    active_config_version: str | None
    last_reload_at: datetime | None
    last_reload_error: str | None


class RuntimeReloadResponse(BaseModel):
    accepted: bool
    message: str
    target_generation: int


@router.get("/api/v1/runtime/status", response_model=RuntimeStatusResponse)
async def get_runtime_status(
    app: RuntimeRoutesApp = Depends(get_runtime_routes_app),
) -> RuntimeStatusResponse:
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
    app: RuntimeRoutesApp = Depends(get_runtime_routes_app),
) -> RuntimeReloadResponse:
    """Trigger runtime reload and return async acceptance outcome."""
    try:
        request = await app.request_runtime_reload()
    except RuntimeReloadConfigError as exc:
        raise APIError(
            str(exc),
            status_code=exc.status_code,
            error_code=exc.error_code,
        ) from exc

    if not request.accepted:
        raise APIError(
            request.message,
            status_code=status.HTTP_409_CONFLICT,
            error_code=APIErrorCode.RELOAD_IN_PROGRESS,
            extra={"target_generation": request.target_generation},
        )

    return RuntimeReloadResponse(
        accepted=True,
        message="Runtime reload accepted",
        target_generation=request.target_generation,
    )
