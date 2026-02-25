"""Retention wiring helpers for runtime assembly."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from homesec.interfaces import ClipSource
from homesec.models.config import RetentionConfig
from homesec.repository import ClipRepository
from homesec.retention.pruner import LocalRetentionPruner

logger = logging.getLogger(__name__)

_SOURCE_PATH_ATTRIBUTES: Final[tuple[str, ...]] = ("watch_dir", "root_dir", "output_dir")


def resolve_max_local_size_bytes(retention: RetentionConfig) -> int:
    """Resolve configured local retention byte cap."""
    return retention.max_local_size_bytes


def discover_local_clip_dirs(sources: list[ClipSource]) -> list[Path]:
    """Discover source-local clip directories for retention scanning."""
    discovered: list[Path] = []
    seen: set[Path] = set()

    for source in sources:
        found_for_source = False
        for attribute in _SOURCE_PATH_ATTRIBUTES:
            raw_path = getattr(source, attribute, None)
            if raw_path is None:
                continue
            path = _coerce_path(raw_path)
            if path is None:
                continue
            normalized = path.expanduser()
            if normalized in seen:
                found_for_source = True
                continue
            seen.add(normalized)
            discovered.append(normalized)
            found_for_source = True

        if not found_for_source:
            logger.warning(
                "Retention source dir discovery skipped: source=%s",
                type(source).__name__,
            )

    discovered.sort()
    return discovered


def build_local_retention_pruner(
    *,
    repository: ClipRepository,
    retention: RetentionConfig,
    sources: list[ClipSource],
) -> LocalRetentionPruner:
    """Construct LocalRetentionPruner from runtime state."""
    local_clip_dirs = discover_local_clip_dirs(sources)
    if not local_clip_dirs:
        logger.warning("Retention pruner has no local clip directories; prune passes will no-op")
    max_local_size_bytes = resolve_max_local_size_bytes(retention)
    logger.info(
        "Retention pruner configured: max_local_size_bytes=%d local_clip_dirs=%s",
        max_local_size_bytes,
        local_clip_dirs,
    )
    return LocalRetentionPruner(
        repository=repository,
        local_clip_dirs=local_clip_dirs,
        max_local_size_bytes=max_local_size_bytes,
    )


def _coerce_path(raw_path: object) -> Path | None:
    if isinstance(raw_path, Path):
        return raw_path
    if isinstance(raw_path, str):
        return Path(raw_path)
    return None
