"""Storage backend plugins and registry."""

from __future__ import annotations

import logging
from typing import cast

from homesec.interfaces import StorageBackend
from homesec.models.config import StorageConfig
from homesec.plugins.registry import PluginType, load_plugin

logger = logging.getLogger(__name__)


def load_storage_plugin(config: StorageConfig) -> StorageBackend:
    """Load and instantiate a storage backend plugin.

    Args:
        config: Storage configuration with backend name and backend-specific settings

    Returns:
        Instantiated storage backend

    Raises:
        ValueError: If backend is unknown or backend-specific config is missing
    """
    # Extract backend-specific config using attribute access
    specific_config = getattr(config, config.backend.lower(), None)
    if specific_config is None:
        raise ValueError(
            f"Missing '{config.backend.lower()}' config in storage section. "
            f"Add 'storage.{config.backend.lower()}' to your config."
        )

    return cast(
        StorageBackend,
        load_plugin(
            PluginType.STORAGE,
            config.backend,
            specific_config,
        ),
    )


__all__ = ["load_storage_plugin"]
