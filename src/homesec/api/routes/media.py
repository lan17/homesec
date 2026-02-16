"""Clip media playback endpoint."""

from __future__ import annotations

import logging
import mimetypes
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, status
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from homesec.api.dependencies import get_homesec_app
from homesec.api.errors import APIError, APIErrorCode

if TYPE_CHECKING:
    from homesec.app import Application

router = APIRouter(tags=["clips"])
logger = logging.getLogger(__name__)


@router.get("/api/v1/clips/{clip_id}/media")
async def get_clip_media(clip_id: str, app: Application = Depends(get_homesec_app)) -> FileResponse:
    """Stream clip media through HomeSec for in-app playback."""
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

    temp_dir = Path(tempfile.mkdtemp(prefix="homesec-media-"))
    media_suffix = _infer_media_suffix(state.storage_uri)
    media_path = temp_dir / _build_media_filename(clip_id, media_suffix)

    try:
        await app.storage.get(state.storage_uri, media_path)
    except Exception as exc:
        _cleanup_media_temp_dir(temp_dir)
        logger.error(
            "Clip media fetch failed for clip_id=%s storage_uri=%s: %s",
            clip_id,
            state.storage_uri,
            exc,
            exc_info=True,
        )
        raise APIError(
            "Clip media fetch failed",
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_code=APIErrorCode.CLIP_MEDIA_FETCH_FAILED,
        ) from exc

    return FileResponse(
        path=media_path,
        media_type=_guess_media_type(media_path),
        headers={"Content-Disposition": f'inline; filename="{media_path.name}"'},
        background=BackgroundTask(_cleanup_media_temp_dir, temp_dir),
    )


def _cleanup_media_temp_dir(temp_dir: Path) -> None:
    try:
        shutil.rmtree(temp_dir)
    except FileNotFoundError:
        return
    except Exception as exc:
        logger.warning("Failed to remove temp media dir %s: %s", temp_dir, exc, exc_info=True)


def _infer_media_suffix(storage_uri: str) -> str:
    _, _, uri_path = storage_uri.partition(":")
    normalized = uri_path.split("?", maxsplit=1)[0].split("#", maxsplit=1)[0]
    suffix = Path(normalized).suffix
    if suffix:
        return suffix
    return ".mp4"


def _build_media_filename(clip_id: str, suffix: str) -> str:
    safe_clip_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in clip_id)
    normalized_suffix = suffix if suffix.startswith(".") else ".mp4"
    return f"{safe_clip_id or 'clip'}{normalized_suffix}"


def _guess_media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"
