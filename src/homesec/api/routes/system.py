"""System control endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from homesec.api.dependencies import get_homesec_app
from homesec.runtime.errors import RuntimeReloadConfigError

if TYPE_CHECKING:
    from homesec.app import Application

router = APIRouter(tags=["system"])


class SystemRestartResponse(BaseModel):
    accepted: bool
    message: str
    target_generation: int


@router.post("/api/v1/system/restart", response_model=SystemRestartResponse, status_code=202)
async def restart_system(
    app: Application = Depends(get_homesec_app),
) -> SystemRestartResponse | JSONResponse:
    """Restart runtime subprocess while preserving FastAPI control-plane."""
    try:
        request = await app.request_system_restart()
    except RuntimeReloadConfigError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": str(exc),
                "error_code": exc.error_code,
            },
        )

    if not request.accepted:
        return JSONResponse(
            status_code=409,
            content={
                "detail": request.message,
                "error_code": "RESTART_IN_PROGRESS",
                "target_generation": request.target_generation,
            },
        )

    return SystemRestartResponse(
        accepted=True,
        message="Runtime restart accepted",
        target_generation=request.target_generation,
    )
