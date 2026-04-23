"""Preview control-plane and playback endpoints."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote, urlencode

from fastapi import APIRouter, Depends, status
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from starlette.background import BackgroundTask

from homesec.api.dependencies import get_homesec_app, verify_preview_access
from homesec.api.errors import APIError, APIErrorCode
from homesec.api.preview_tokens import issue_camera_preview_token
from homesec.preview_paths import (
    is_preview_segment_name,
    preview_playlist_path,
    preview_segment_path,
)
from homesec.runtime.errors import PreviewCameraNotFoundError, PreviewRuntimeUnavailableError
from homesec.runtime.models import CameraPreviewStartRefusal, CameraPreviewStatus, PreviewState

if TYPE_CHECKING:
    from homesec.app import Application
from homesec.runtime.models import PreviewRefusalReason

logger = logging.getLogger(__name__)
_PREVIEW_CACHE_HEADERS = {"Cache-Control": "no-store"}
_ACTIVE_PLAYBACK_STATES = frozenset(
    {
        PreviewState.STARTING,
        PreviewState.READY,
        PreviewState.DEGRADED,
    }
)

control_router = APIRouter(tags=["preview"])
playback_router = APIRouter(tags=["preview"])
router = control_router


class PreviewStatusResponse(BaseModel):
    camera_name: str
    enabled: bool
    state: PreviewState
    viewer_count: int | None = None
    degraded_reason: str | None = None
    last_error: str | None = None
    idle_shutdown_at: float | None = None


class PreviewSessionResponse(BaseModel):
    camera_name: str
    state: PreviewState
    viewer_count: int | None = None
    token: str | None = None
    token_expires_at: datetime | None = None
    playlist_url: str
    idle_timeout_s: float
    warning: str | None = None


class PreviewStopResponse(BaseModel):
    accepted: bool
    state: PreviewState


def _status_response(status: CameraPreviewStatus) -> PreviewStatusResponse:
    return PreviewStatusResponse(
        camera_name=status.camera_name,
        enabled=status.enabled,
        state=status.state,
        viewer_count=status.viewer_count,
        degraded_reason=status.degraded_reason,
        last_error=status.last_error,
        idle_shutdown_at=status.idle_shutdown_at,
    )


def _playlist_url(camera_name: str, *, token: str | None = None) -> str:
    path = f"/api/v1/preview/cameras/{quote(camera_name, safe='')}/playlist.m3u8"
    if token is None:
        return path
    return f"{path}?{urlencode({'token': token})}"


def _warning(status: CameraPreviewStatus) -> str | None:
    if status.state is not PreviewState.DEGRADED:
        return None
    if status.degraded_reason:
        return status.degraded_reason
    if status.last_error:
        return status.last_error
    return "Preview is degraded"


def _refusal_error_code(reason: PreviewRefusalReason) -> APIErrorCode:
    match reason:
        case PreviewRefusalReason.RECORDING_PRIORITY:
            return APIErrorCode.PREVIEW_RECORDING_PRIORITY
        case PreviewRefusalReason.SESSION_BUDGET_EXHAUSTED:
            return APIErrorCode.PREVIEW_SESSION_BUDGET_EXHAUSTED
        case PreviewRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE:
            return APIErrorCode.PREVIEW_TEMPORARILY_UNAVAILABLE


def _raise_camera_not_found(exc: PreviewCameraNotFoundError) -> None:
    raise APIError(
        str(exc),
        status_code=status.HTTP_404_NOT_FOUND,
        error_code=APIErrorCode.PREVIEW_CAMERA_NOT_FOUND,
    ) from exc


def _raise_runtime_unavailable(exc: PreviewRuntimeUnavailableError) -> None:
    raise APIError(
        str(exc),
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        error_code=APIErrorCode.PREVIEW_RUNTIME_UNAVAILABLE,
    ) from exc


def _preview_storage_dir(app: Application) -> Path:
    return Path(app.config.preview.config.storage_dir)


def _read_playlist_text(playlist_path: Path) -> str:
    if not playlist_path.is_file():
        raise APIError(
            "Preview media unavailable",
            status_code=status.HTTP_409_CONFLICT,
            error_code=APIErrorCode.PREVIEW_MEDIA_UNAVAILABLE,
        )
    try:
        playlist_text = playlist_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning(
            "Failed to read preview playlist at %s: %s",
            playlist_path,
            exc,
            exc_info=True,
        )
        raise APIError(
            "Preview media unavailable",
            status_code=status.HTTP_409_CONFLICT,
            error_code=APIErrorCode.PREVIEW_MEDIA_UNAVAILABLE,
        ) from exc
    if not playlist_text.strip():
        raise APIError(
            "Preview media unavailable",
            status_code=status.HTTP_409_CONFLICT,
            error_code=APIErrorCode.PREVIEW_MEDIA_UNAVAILABLE,
        )
    return playlist_text


def _rewrite_playlist_for_token(playlist_text: str, token: str | None) -> str:
    if token is None:
        return playlist_text

    lines: list[str] = []
    for line in playlist_text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            separator = "&" if "?" in stripped else "?"
            lines.append(f"{stripped}{separator}{urlencode({'token': token})}")
        else:
            lines.append(line)

    rewritten = "\n".join(lines)
    if playlist_text.endswith(("\n", "\r")):
        rewritten += "\n"
    return rewritten


async def _ensure_known_camera(app: Application, camera_name: str) -> None:
    if any(camera.name == camera_name for camera in app.config.cameras):
        return
    _raise_camera_not_found(PreviewCameraNotFoundError(camera_name))


async def _ensure_preview_playback_enabled(app: Application, camera_name: str) -> None:
    await _ensure_known_camera(app, camera_name)

    camera = next((item for item in app.config.cameras if item.name == camera_name), None)
    if (
        camera is None
        or not app.config.preview.enabled
        or not getattr(camera, "enabled", True)
        or getattr(getattr(camera, "source", None), "backend", None) != "rtsp"
    ):
        raise APIError(
            "Preview is not enabled for this camera",
            status_code=status.HTTP_409_CONFLICT,
            error_code=APIErrorCode.PREVIEW_TEMPORARILY_UNAVAILABLE,
        )
    if not app.pipeline_running:
        raise APIError(
            "Preview runtime unavailable",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=APIErrorCode.PREVIEW_RUNTIME_UNAVAILABLE,
        )


async def _ensure_preview_media_active(app: Application, camera_name: str) -> None:
    try:
        preview_status = await app.get_camera_preview_status(camera_name)
    except PreviewCameraNotFoundError as exc:
        _raise_camera_not_found(exc)
    except PreviewRuntimeUnavailableError as exc:
        _raise_runtime_unavailable(exc)

    if not preview_status.enabled or preview_status.state not in _ACTIVE_PLAYBACK_STATES:
        raise APIError(
            "Preview media unavailable",
            status_code=status.HTTP_409_CONFLICT,
            error_code=APIErrorCode.PREVIEW_MEDIA_UNAVAILABLE,
        )


async def _note_preview_viewer_activity_best_effort(
    app: Application,
    camera_name: str,
    *,
    viewer_id: str | None = None,
) -> None:
    try:
        await app.note_camera_preview_viewer_activity(
            camera_name,
            viewer_id=viewer_id,
        )
    except PreviewCameraNotFoundError as exc:
        logger.warning(
            "Preview viewer activity camera disappeared for camera=%s: %s",
            camera_name,
            exc,
        )
    except PreviewRuntimeUnavailableError as exc:
        logger.warning(
            "Preview viewer activity update skipped for camera=%s: %s",
            camera_name,
            exc,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Unexpected preview viewer activity failure for camera=%s: %s",
            camera_name,
            exc,
            exc_info=True,
        )


@control_router.get("/api/v1/preview/cameras/{camera_name}", response_model=PreviewStatusResponse)
async def get_preview_status(
    camera_name: str,
    app: Application = Depends(get_homesec_app),
) -> PreviewStatusResponse:
    """Return preview status for a camera."""
    try:
        preview_status = await app.get_camera_preview_status(camera_name)
    except PreviewCameraNotFoundError as exc:
        _raise_camera_not_found(exc)
    except PreviewRuntimeUnavailableError as exc:
        _raise_runtime_unavailable(exc)

    return _status_response(preview_status)


@control_router.post("/api/v1/preview/cameras/{camera_name}", response_model=PreviewSessionResponse)
async def ensure_preview_active(
    camera_name: str,
    app: Application = Depends(get_homesec_app),
) -> PreviewSessionResponse:
    """Ensure preview is active for a camera and mint a fresh attach token."""
    try:
        outcome = await app.ensure_camera_preview_active(camera_name)
    except PreviewCameraNotFoundError as exc:
        _raise_camera_not_found(exc)
    except PreviewRuntimeUnavailableError as exc:
        _raise_runtime_unavailable(exc)

    if isinstance(outcome, CameraPreviewStartRefusal):
        raise APIError(
            outcome.message,
            status_code=status.HTTP_409_CONFLICT,
            error_code=_refusal_error_code(outcome.reason),
            extra={"reason": outcome.reason.value},
        )

    preview_config = app.config.preview
    token: str | None = None
    expires_at: datetime | None = None
    if app.server_config.auth_enabled:
        api_key = app.server_config.get_api_key()
        if api_key is None:
            raise APIError(
                "API key not configured",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error_code=APIErrorCode.API_KEY_NOT_CONFIGURED,
            )
        token, expires_at = issue_camera_preview_token(
            api_key=api_key,
            camera_name=camera_name,
            ttl_s=preview_config.token_ttl_s,
        )

    return PreviewSessionResponse(
        camera_name=camera_name,
        state=outcome.state,
        viewer_count=outcome.viewer_count,
        token=token,
        token_expires_at=expires_at,
        playlist_url=_playlist_url(camera_name, token=token),
        idle_timeout_s=preview_config.idle_timeout_s,
        warning=_warning(outcome),
    )


@control_router.delete(
    "/api/v1/preview/cameras/{camera_name}",
    response_model=PreviewStopResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def force_stop_preview(
    camera_name: str,
    app: Application = Depends(get_homesec_app),
) -> PreviewStopResponse:
    """Force-stop preview for a camera."""
    try:
        result = await app.force_stop_camera_preview(camera_name)
    except PreviewCameraNotFoundError as exc:
        _raise_camera_not_found(exc)
    except PreviewRuntimeUnavailableError as exc:
        _raise_runtime_unavailable(exc)

    return PreviewStopResponse(accepted=result.accepted, state=result.state)


@playback_router.get("/api/v1/preview/cameras/{camera_name}/playlist.m3u8")
async def get_preview_playlist(
    camera_name: str,
    preview_token: str | None = Depends(verify_preview_access),
    app: Application = Depends(get_homesec_app),
) -> Response:
    """Return the live HLS playlist for a camera preview session."""
    await _ensure_preview_playback_enabled(app, camera_name)
    await _ensure_preview_media_active(app, camera_name)
    playlist_text = _read_playlist_text(
        preview_playlist_path(_preview_storage_dir(app), camera_name)
    )
    return Response(
        content=_rewrite_playlist_for_token(playlist_text, preview_token),
        media_type="application/vnd.apple.mpegurl",
        headers=_PREVIEW_CACHE_HEADERS,
    )


@playback_router.get("/api/v1/preview/cameras/{camera_name}/{segment_name}")
async def get_preview_segment(
    camera_name: str,
    segment_name: str,
    preview_token: str | None = Depends(verify_preview_access),
    app: Application = Depends(get_homesec_app),
) -> FileResponse:
    """Return a live HLS transport-stream segment for a camera preview session."""
    if not is_preview_segment_name(segment_name):
        raise APIError(
            "Not Found",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=APIErrorCode.NOT_FOUND,
        )

    await _ensure_preview_playback_enabled(app, camera_name)
    await _ensure_preview_media_active(app, camera_name)
    try:
        segment_path = preview_segment_path(_preview_storage_dir(app), camera_name, segment_name)
    except ValueError as exc:
        raise APIError(
            "Not Found",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=APIErrorCode.NOT_FOUND,
        ) from exc

    if not segment_path.is_file():
        raise APIError(
            "Preview media unavailable",
            status_code=status.HTTP_409_CONFLICT,
            error_code=APIErrorCode.PREVIEW_MEDIA_UNAVAILABLE,
        )

    return FileResponse(
        path=segment_path,
        media_type="video/mp2t",
        headers=_PREVIEW_CACHE_HEADERS,
        background=BackgroundTask(
            _note_preview_viewer_activity_best_effort,
            app,
            camera_name,
            viewer_id=preview_token,
        ),
    )
