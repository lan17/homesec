"""Clip browsing endpoints."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from homesec.api.dependencies import get_homesec_app

if TYPE_CHECKING:
    from homesec.app import Application
from homesec.models.clip import ClipStateData
from homesec.models.enums import ClipStatus

router = APIRouter(tags=["clips"])
logger = logging.getLogger(__name__)


class ClipResponse(BaseModel):
    id: str
    camera: str
    status: str
    created_at: datetime
    activity_type: str | None = None
    risk_level: str | None = None
    summary: str | None = None
    detected_objects: list[str] = Field(default_factory=list)
    storage_uri: str | None = None
    view_url: str | None = None
    alerted: bool = False


class ClipListResponse(BaseModel):
    clips: list[ClipResponse]
    total: int
    page: int
    page_size: int


def _status_value(status: ClipStatus | str) -> str:
    if isinstance(status, ClipStatus):
        return status.value
    return str(status)


def _clip_response(state: ClipStateData) -> ClipResponse:
    analysis = state.analysis_result
    detected = state.filter_result.detected_classes if state.filter_result else []
    alerted = state.alert_decision.notify if state.alert_decision else False
    created_at = state.created_at or datetime.now(timezone.utc)
    clip_id = state.clip_id or ""

    return ClipResponse(
        id=clip_id,
        camera=state.camera_name,
        status=_status_value(state.status),
        created_at=created_at,
        activity_type=analysis.activity_type if analysis else None,
        risk_level=str(analysis.risk_level) if analysis else None,
        summary=analysis.summary if analysis else None,
        detected_objects=detected,
        storage_uri=state.storage_uri,
        view_url=state.view_url,
        alerted=alerted,
    )


@router.get("/api/v1/clips", response_model=ClipListResponse)
async def list_clips(
    camera: str | None = None,
    status: ClipStatus | None = None,
    alerted: bool | None = None,
    risk_level: str | None = None,
    activity_type: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    app: Application = Depends(get_homesec_app),
) -> ClipListResponse:
    """List clips with filtering and pagination."""
    offset = (page - 1) * page_size
    clips, total = await app.repository.list_clips(
        camera=camera,
        status=status,
        alerted=alerted,
        risk_level=risk_level,
        activity_type=activity_type,
        since=since,
        until=until,
        offset=offset,
        limit=page_size,
    )

    return ClipListResponse(
        clips=[_clip_response(state) for state in clips],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/api/v1/clips/{clip_id}", response_model=ClipResponse)
async def get_clip(clip_id: str, app: Application = Depends(get_homesec_app)) -> ClipResponse:
    """Get a single clip."""
    state = await app.repository.get_clip(clip_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Clip not found")
    return _clip_response(state)


@router.delete("/api/v1/clips/{clip_id}", response_model=ClipResponse)
async def delete_clip(clip_id: str, app: Application = Depends(get_homesec_app)) -> ClipResponse:
    """Delete a clip and its storage object."""
    state = await app.repository.get_clip(clip_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Clip not found: {clip_id}")

    try:
        if state.storage_uri:
            await app.storage.delete(state.storage_uri)
    except Exception as exc:
        logger.error(
            "Storage delete failed for clip %s: %s",
            clip_id,
            exc,
            exc_info=exc,
        )
        raise HTTPException(status_code=500, detail="Storage deletion failed") from exc

    try:
        deleted = await app.repository.delete_clip(clip_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if deleted.storage_uri is None:
        deleted.storage_uri = state.storage_uri

    return _clip_response(deleted)
