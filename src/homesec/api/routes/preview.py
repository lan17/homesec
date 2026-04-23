"""Preview control-plane endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from urllib.parse import quote, urlencode

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel

from homesec.api.dependencies import get_homesec_app
from homesec.api.errors import APIError, APIErrorCode
from homesec.api.preview_tokens import issue_camera_preview_token
from homesec.runtime.errors import PreviewCameraNotFoundError, PreviewRuntimeUnavailableError
from homesec.runtime.models import CameraPreviewStartRefusal, CameraPreviewStatus, PreviewState

if TYPE_CHECKING:
    from homesec.app import Application
from homesec.runtime.models import PreviewRefusalReason

router = APIRouter(tags=["preview"])


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


@router.get("/api/v1/preview/cameras/{camera_name}", response_model=PreviewStatusResponse)
async def get_preview_status(
    camera_name: str,
    app: Application = Depends(get_homesec_app),
) -> PreviewStatusResponse:
    """Return preview status for a camera."""
    try:
        preview_status = await app.get_camera_preview_status(camera_name)
    except PreviewCameraNotFoundError as exc:
        _raise_camera_not_found(exc)

    return _status_response(preview_status)


@router.post("/api/v1/preview/cameras/{camera_name}", response_model=PreviewSessionResponse)
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


@router.delete(
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
