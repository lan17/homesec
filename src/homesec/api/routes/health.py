"""Health and diagnostics endpoints."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from homesec.api.dependencies import get_homesec_app

if TYPE_CHECKING:
    from homesec.app import Application

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    pipeline: str
    postgres: str
    cameras_online: int


class ComponentStatus(BaseModel):
    status: str
    error: str | None = None
    latency_ms: float | None = None


class CameraStatus(BaseModel):
    healthy: bool
    enabled: bool
    last_heartbeat: float | None


class DiagnosticsResponse(BaseModel):
    status: str
    uptime_seconds: float
    postgres: ComponentStatus
    storage: ComponentStatus
    cameras: dict[str, CameraStatus]


@router.get("/api/v1/health", response_model=HealthResponse)
async def get_health(app: Application = Depends(get_homesec_app)) -> HealthResponse | JSONResponse:
    """Basic health check."""
    if app.bootstrap_mode:
        return HealthResponse(
            status="bootstrap",
            pipeline="stopped",
            postgres="unavailable",
            cameras_online=0,
        )

    pipeline_running = app.pipeline_running
    postgres_ok = await app.repository.ping()
    cameras_online = sum(1 for source in app.sources if source.is_healthy())

    if not pipeline_running:
        status = "unhealthy"
    elif postgres_ok:
        status = "healthy"
    else:
        status = "degraded"

    response = HealthResponse(
        status=status,
        pipeline="running" if pipeline_running else "stopped",
        postgres="connected" if postgres_ok else "unavailable",
        cameras_online=cameras_online,
    )

    if not pipeline_running:
        return JSONResponse(status_code=503, content=response.model_dump(mode="json"))
    return response


@router.get("/api/v1/diagnostics", response_model=DiagnosticsResponse)
async def get_diagnostics(app: Application = Depends(get_homesec_app)) -> DiagnosticsResponse:
    """Detailed component diagnostics."""
    if app.bootstrap_mode:
        return DiagnosticsResponse(
            status="bootstrap",
            uptime_seconds=app.uptime_seconds,
            postgres=ComponentStatus(status="error", error="bootstrap_mode"),
            storage=ComponentStatus(status="error", error="bootstrap_mode"),
            cameras={},
        )

    pipeline_running = app.pipeline_running

    async def _check(ping: Any) -> ComponentStatus:
        start = time.perf_counter()
        try:
            ok = await ping()
            latency_ms = (time.perf_counter() - start) * 1000
        except Exception as exc:  # pragma: no cover - defensive
            return ComponentStatus(status="error", error=str(exc))

        if ok:
            return ComponentStatus(status="ok", latency_ms=latency_ms)
        return ComponentStatus(status="error", error="unavailable", latency_ms=latency_ms)

    postgres_status = await _check(app.repository.ping)
    storage_status = await _check(app.storage.ping)

    cameras: dict[str, CameraStatus] = {}
    for camera in app.config.cameras:
        source = app.get_source(camera.name)
        if camera.enabled and source is not None:
            healthy = source.is_healthy()
            last_heartbeat = source.last_heartbeat()
        else:
            healthy = False
            last_heartbeat = None

        cameras[camera.name] = CameraStatus(
            healthy=healthy,
            enabled=camera.enabled,
            last_heartbeat=last_heartbeat,
        )

    if not pipeline_running:
        status = "unhealthy"
    elif (
        postgres_status.status == "error"
        or storage_status.status == "error"
        or any(cam.enabled and not cam.healthy for cam in cameras.values())
    ):
        status = "degraded"
    else:
        status = "healthy"

    return DiagnosticsResponse(
        status=status,
        uptime_seconds=app.uptime_seconds,
        postgres=postgres_status,
        storage=storage_status,
        cameras=cameras,
    )
