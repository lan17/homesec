"""Shared preview artifact path helpers."""

from __future__ import annotations

import re
from pathlib import Path

_SEGMENT_NAME_RE = re.compile(r"^segment_\d+\.ts$")


def sanitize_preview_camera_name(name: str) -> str:
    """Return the canonical preview directory slug for a camera name."""
    raw = str(name).strip()
    if not raw:
        return "camera"
    out: list[str] = []
    for ch in raw:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "camera"


def preview_camera_dir(storage_dir: Path, camera_name: str) -> Path:
    """Return the camera-scoped preview directory inside the configured scratch root."""
    return Path(storage_dir) / "homesec" / sanitize_preview_camera_name(camera_name)


def preview_playlist_path(storage_dir: Path, camera_name: str) -> Path:
    """Return the camera-scoped HLS playlist path."""
    return preview_camera_dir(storage_dir, camera_name) / "playlist.m3u8"


def preview_ffmpeg_log_path(storage_dir: Path, camera_name: str) -> Path:
    """Return the camera-scoped ffmpeg stderr log path."""
    return preview_camera_dir(storage_dir, camera_name) / "preview_ffmpeg.log"


def preview_segment_filename_pattern(storage_dir: Path, camera_name: str) -> Path:
    """Return the ffmpeg segment filename pattern for a camera."""
    return preview_camera_dir(storage_dir, camera_name) / "segment_%06d.ts"


def preview_segment_path(storage_dir: Path, camera_name: str, segment_name: str) -> Path:
    """Return the path for a validated segment file."""
    if not is_preview_segment_name(segment_name):
        raise ValueError(f"Invalid preview segment name: {segment_name}")
    return preview_camera_dir(storage_dir, camera_name) / segment_name


def is_preview_segment_name(segment_name: str) -> bool:
    """Return True when the segment name matches the v1 HLS layout."""
    return bool(_SEGMENT_NAME_RE.fullmatch(segment_name))
