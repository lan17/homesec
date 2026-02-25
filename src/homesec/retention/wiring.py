"""Retention wiring helpers for runtime assembly."""

from __future__ import annotations

import logging

from homesec.models.config import RetentionConfig
from homesec.repository import ClipRepository
from homesec.retention.pruner import LocalRetentionPruner

logger = logging.getLogger(__name__)


def build_local_retention_pruner(
    *,
    repository: ClipRepository,
    retention: RetentionConfig,
) -> LocalRetentionPruner:
    """Construct LocalRetentionPruner from runtime state.

    Local clip directories are discovered dynamically from clip arrivals.
    """
    logger.info(
        "Retention pruner configured: max_local_size_bytes=%d local_clip_dirs=[] "
        "policy=learn_from_clip_local_path",
        retention.max_local_size_bytes,
    )
    return LocalRetentionPruner(
        repository=repository,
        local_clip_dirs=[],
        max_local_size_bytes=retention.max_local_size_bytes,
    )
