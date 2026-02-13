"""Camera CRUD endpoints."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from homesec.api.dependencies import get_homesec_app
from homesec.models.config import CameraConfig

if TYPE_CHECKING:
    from homesec.app import Application

router = APIRouter(tags=["cameras"])


class CameraCreate(BaseModel):
    name: str
    enabled: bool = True
    source_backend: str
    source_config: dict[str, object]


class CameraUpdate(BaseModel):
    enabled: bool | None = None
    source_config: dict[str, object] | None = None


class CameraResponse(BaseModel):
    name: str
    enabled: bool
    source_backend: str
    healthy: bool
    last_heartbeat: float | None
    source_config: dict[str, object]


class ConfigChangeResponse(BaseModel):
    restart_required: bool = True
    camera: CameraResponse | None = None


def _source_config_to_dict(camera: CameraConfig) -> dict[str, object]:
    config = camera.source.config
    if hasattr(config, "model_dump"):
        return config.model_dump(mode="json")
    return dict(config)


def _camera_response(app: Application, camera: CameraConfig) -> CameraResponse:
    source = app.get_source(camera.name)
    if camera.enabled and source is not None:
        healthy = source.is_healthy()
        last_heartbeat = source.last_heartbeat()
    else:
        healthy = False
        last_heartbeat = None

    return CameraResponse(
        name=camera.name,
        enabled=camera.enabled,
        source_backend=camera.source.backend,
        healthy=healthy,
        last_heartbeat=last_heartbeat,
        source_config=_source_config_to_dict(camera),
    )


@router.get("/api/v1/cameras", response_model=list[CameraResponse])
async def list_cameras(app: Application = Depends(get_homesec_app)) -> list[CameraResponse]:
    """List all cameras."""
    if app.bootstrap_mode:
        return []
    config = await asyncio.to_thread(app.config_manager.get_config)
    return [_camera_response(app, camera) for camera in config.cameras]


@router.get("/api/v1/cameras/{name}", response_model=CameraResponse)
async def get_camera(name: str, app: Application = Depends(get_homesec_app)) -> CameraResponse:
    """Get a single camera."""
    if app.bootstrap_mode:
        raise HTTPException(status_code=404, detail="Camera not found")
    config = await asyncio.to_thread(app.config_manager.get_config)
    camera = next((cam for cam in config.cameras if cam.name == name), None)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return _camera_response(app, camera)


@router.post("/api/v1/cameras", response_model=ConfigChangeResponse, status_code=201)
async def create_camera(
    payload: CameraCreate, app: Application = Depends(get_homesec_app)
) -> ConfigChangeResponse:
    """Create a new camera."""
    if app.bootstrap_mode:
        raise HTTPException(
            status_code=409, detail="HomeSec is in bootstrap mode; configure full settings first"
        )
    try:
        result = await app.config_manager.add_camera(
            name=payload.name,
            enabled=payload.enabled,
            source_backend=payload.source_backend,
            source_config=payload.source_config,
        )
    except ValueError as exc:
        detail = str(exc)
        if "Camera not found" in detail:
            raise HTTPException(status_code=404, detail=detail) from exc
        raise HTTPException(status_code=400, detail=detail) from exc

    config = await asyncio.to_thread(app.config_manager.get_config)
    camera = next((cam for cam in config.cameras if cam.name == payload.name), None)
    return ConfigChangeResponse(
        restart_required=result.restart_required,
        camera=_camera_response(app, camera) if camera else None,
    )


@router.put("/api/v1/cameras/{name}", response_model=ConfigChangeResponse)
async def update_camera(
    name: str,
    payload: CameraUpdate,
    app: Application = Depends(get_homesec_app),
) -> ConfigChangeResponse:
    """Update a camera."""
    if app.bootstrap_mode:
        raise HTTPException(
            status_code=409, detail="HomeSec is in bootstrap mode; configure full settings first"
        )
    try:
        result = await app.config_manager.update_camera(
            camera_name=name,
            enabled=payload.enabled,
            source_config=payload.source_config,
        )
    except ValueError as exc:
        detail = str(exc)
        if "Camera not found" in detail:
            raise HTTPException(status_code=404, detail=detail) from exc
        raise HTTPException(status_code=400, detail=detail) from exc

    config = await asyncio.to_thread(app.config_manager.get_config)
    camera = next((cam for cam in config.cameras if cam.name == name), None)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    return ConfigChangeResponse(
        restart_required=result.restart_required,
        camera=_camera_response(app, camera),
    )


@router.delete("/api/v1/cameras/{name}", response_model=ConfigChangeResponse)
async def delete_camera(
    name: str, app: Application = Depends(get_homesec_app)
) -> ConfigChangeResponse:
    """Delete a camera."""
    if app.bootstrap_mode:
        raise HTTPException(
            status_code=409, detail="HomeSec is in bootstrap mode; configure full settings first"
        )
    try:
        result = await app.config_manager.remove_camera(camera_name=name)
    except ValueError as exc:
        detail = str(exc)
        if "Camera not found" in detail:
            raise HTTPException(status_code=404, detail=detail) from exc
        raise HTTPException(status_code=400, detail=detail) from exc

    return ConfigChangeResponse(restart_required=result.restart_required, camera=None)
