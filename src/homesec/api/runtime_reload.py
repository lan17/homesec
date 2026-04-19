"""Shared API helpers for runtime reload requests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import status
from pydantic import BaseModel

from homesec.api.errors import APIError, APIErrorCode
from homesec.runtime.errors import RuntimeReloadConfigError

if TYPE_CHECKING:
    from homesec.app import Application


class RuntimeReloadResponse(BaseModel):
    """API response payload for accepted runtime reload requests."""

    accepted: bool
    message: str
    target_generation: int


async def request_runtime_reload(app: Application) -> RuntimeReloadResponse:
    """Request a runtime reload and map domain/runtime errors into API errors."""
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


async def reload_runtime_if_requested(
    *,
    apply_changes: bool,
    app: Application,
) -> RuntimeReloadResponse | None:
    """Request a runtime reload only when a settings route opts into apply-now behavior."""
    if not apply_changes:
        return None

    return await request_runtime_reload(app)
