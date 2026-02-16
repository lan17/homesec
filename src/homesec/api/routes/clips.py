"""Clip browsing endpoints."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from urllib.parse import quote, urlencode

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field

from homesec.api.dependencies import get_homesec_app
from homesec.api.errors import APIError, APIErrorCode
from homesec.api.media_tokens import issue_clip_media_token
from homesec.api.pagination import CursorDecodeError, decode_clip_cursor, encode_clip_cursor

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
    limit: int
    next_cursor: str | None
    has_more: bool


class ClipMediaTokenResponse(BaseModel):
    media_url: str
    tokenized: bool
    expires_at: datetime | None


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
    clip_status: ClipStatus | None = Query(default=None, alias="status"),
    alerted: bool | None = None,
    detected: bool | None = None,
    risk_level: str | None = None,
    activity_type: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = Query(50, ge=1, le=100),
    cursor: str | None = None,
    app: Application = Depends(get_homesec_app),
) -> ClipListResponse:
    """List clips with filtering and keyset pagination."""

    since_utc = _normalize_aware_datetime("since", since)
    until_utc = _normalize_aware_datetime("until", until)
    if since_utc is not None and until_utc is not None and since_utc > until_utc:
        raise APIError(
            "since must be less than or equal to until",
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=APIErrorCode.CLIPS_TIME_RANGE_INVALID,
        )

    list_cursor = None
    if cursor is not None:
        try:
            list_cursor = decode_clip_cursor(cursor)
        except CursorDecodeError as exc:
            raise APIError(
                "Invalid cursor token",
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=APIErrorCode.CLIPS_CURSOR_INVALID,
            ) from exc

    page = await app.repository.list_clips(
        camera=camera,
        status=clip_status,
        alerted=alerted,
        detected=detected,
        risk_level=risk_level,
        activity_type=activity_type,
        since=since_utc,
        until=until_utc,
        cursor=list_cursor,
        limit=limit,
    )

    next_cursor = encode_clip_cursor(page.next_cursor) if page.next_cursor is not None else None

    return ClipListResponse(
        clips=[_clip_response(state) for state in page.clips],
        limit=limit,
        next_cursor=next_cursor,
        has_more=page.has_more,
    )


@router.get("/api/v1/clips/{clip_id}", response_model=ClipResponse)
async def get_clip(clip_id: str, app: Application = Depends(get_homesec_app)) -> ClipResponse:
    """Get a single clip."""
    state = await app.repository.get_clip(clip_id)
    if state is None:
        raise APIError(
            "Clip not found",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=APIErrorCode.CLIP_NOT_FOUND,
        )
    return _clip_response(state)


@router.post("/api/v1/clips/{clip_id}/media-token", response_model=ClipMediaTokenResponse)
async def create_clip_media_token(
    clip_id: str,
    app: Application = Depends(get_homesec_app),
) -> ClipMediaTokenResponse:
    """Create media playback URL.

    When auth is disabled, returns direct media URL with tokenized=False and expires_at=None.
    When auth is enabled, returns tokenized URL with tokenized=True and non-null expires_at.
    """
    state = await app.repository.get_clip(clip_id)
    if state is None:
        raise APIError(
            "Clip not found",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=APIErrorCode.CLIP_NOT_FOUND,
        )
    if state.storage_uri is None:
        raise APIError(
            "Clip media unavailable",
            status_code=status.HTTP_409_CONFLICT,
            error_code=APIErrorCode.CLIP_MEDIA_UNAVAILABLE,
        )

    media_path = _build_media_path(clip_id)
    if not app.config.server.auth_enabled:
        return ClipMediaTokenResponse(media_url=media_path, tokenized=False, expires_at=None)

    api_key = app.config.server.get_api_key()
    # API key is guaranteed here by router-level verify_api_key dependency when auth is enabled.
    assert api_key is not None

    token, expires_at = issue_clip_media_token(api_key=api_key, clip_id=clip_id)
    return ClipMediaTokenResponse(
        media_url=_build_media_path(clip_id, token=token),
        tokenized=True,
        expires_at=expires_at,
    )


@router.delete("/api/v1/clips/{clip_id}", response_model=ClipResponse)
async def delete_clip(clip_id: str, app: Application = Depends(get_homesec_app)) -> ClipResponse:
    """Delete a clip and its storage object."""
    state = await app.repository.get_clip(clip_id)
    if state is None:
        raise APIError(
            f"Clip not found: {clip_id}",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=APIErrorCode.CLIP_NOT_FOUND,
        )

    try:
        deleted = await app.repository.delete_clip(clip_id)
    except ValueError as exc:
        logger.warning("Clip delete mark failed (not found): clip_id=%s", clip_id)
        raise APIError(
            str(exc),
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=APIErrorCode.CLIP_NOT_FOUND,
        ) from exc
    except Exception as exc:
        logger.error("Clip delete mark failed: clip_id=%s error=%s", clip_id, exc, exc_info=True)
        raise APIError(
            "Clip delete mark failed",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=APIErrorCode.CLIP_DELETE_MARK_FAILED,
        ) from exc

    storage_uri = state.storage_uri or deleted.storage_uri
    if deleted.storage_uri is None:
        deleted.storage_uri = state.storage_uri

    try:
        if storage_uri:
            await app.storage.delete(storage_uri)
    except Exception as exc:
        logger.error(
            "Storage delete failed after DB mark-deleted for clip %s: %s",
            clip_id,
            exc,
            exc_info=True,
        )
        raise APIError(
            "Storage deletion failed",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=APIErrorCode.CLIP_STORAGE_DELETE_FAILED,
        ) from exc

    return _clip_response(deleted)


def _normalize_aware_datetime(field_name: str, value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise APIError(
            f"{field_name} must include timezone information",
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=APIErrorCode.CLIPS_TIMESTAMP_TZ_REQUIRED,
        )
    return value.astimezone(timezone.utc)


def _build_media_path(clip_id: str, *, token: str | None = None) -> str:
    path = f"/api/v1/clips/{quote(clip_id, safe='')}/media"
    if token is None:
        return path
    return f"{path}?{urlencode({'token': token})}"
