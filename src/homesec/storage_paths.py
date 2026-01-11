"""Helpers for building storage destination paths."""

from __future__ import annotations

from pathlib import PurePosixPath

from homesec.models.clip import Clip
from homesec.models.config import StoragePathsConfig


def _sanitize_segment(value: str) -> str:
    cleaned = value.strip().replace("/", "_").replace("\\", "_")
    cleaned = "_".join(part for part in cleaned.split() if part)
    return cleaned or "unknown"


def _normalize_dest_path(path: PurePosixPath) -> str:
    if path.is_absolute():
        raise ValueError(f"dest_path must be relative, got {path}")
    for part in path.parts:
        if part in ("", ".", ".."):
            raise ValueError(f"dest_path contains invalid segment: {path}")
    return str(path)


def build_clip_path(clip: Clip, paths_cfg: StoragePathsConfig) -> str:
    """Build destination path for a clip using configured defaults."""
    camera = _sanitize_segment(clip.camera_name)
    raw_name = clip.local_path.name or f"{clip.clip_id}{clip.local_path.suffix or '.mp4'}"
    filename = _sanitize_segment(raw_name)
    path = PurePosixPath(paths_cfg.clips_dir) / camera / filename
    return _normalize_dest_path(path)


def build_backup_path(name: str, paths_cfg: StoragePathsConfig) -> str:
    """Build destination path for a backup file."""
    filename = _sanitize_segment(name)
    path = PurePosixPath(paths_cfg.backups_dir) / filename
    return _normalize_dest_path(path)


def build_artifact_path(name: str, paths_cfg: StoragePathsConfig) -> str:
    """Build destination path for an artifact file."""
    filename = _sanitize_segment(name)
    path = PurePosixPath(paths_cfg.artifacts_dir) / filename
    return _normalize_dest_path(path)
