"""Storage backend plugins and registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, TypeVar

from pydantic import BaseModel

from homesec.interfaces import StorageBackend
from homesec.models.config import StorageConfig

logger = logging.getLogger(__name__)

# Type alias for clarity
StorageFactory = Callable[[BaseModel], StorageBackend]


@dataclass(frozen=True)
class StoragePlugin:
    """Metadata for a storage backend plugin."""

    name: str
    config_model: type[BaseModel]
    factory: StorageFactory


STORAGE_REGISTRY: dict[str, StoragePlugin] = {}


def register_storage(plugin: StoragePlugin) -> None:
    """Register a storage plugin with collision detection.

    Args:
        plugin: Storage plugin to register

    Raises:
        ValueError: If a plugin with the same name is already registered
    """
    if plugin.name in STORAGE_REGISTRY:
        raise ValueError(
            f"Storage plugin '{plugin.name}' is already registered. "
            f"Plugin names must be unique across all storage plugins."
        )
    STORAGE_REGISTRY[plugin.name] = plugin


T = TypeVar("T", bound=Callable[[], StoragePlugin])


def storage_plugin(name: str) -> Callable[[T], T]:
    """Decorator to register a storage backend plugin.

    Usage:
        @storage_plugin(name="my_storage")
        def my_storage_plugin() -> StoragePlugin:
            return StoragePlugin(...)

    Args:
        name: Plugin name (for validation only - must match plugin.name)

    Returns:
        Decorator function that registers the plugin
    """

    def decorator(factory_fn: T) -> T:
        plugin = factory_fn()
        register_storage(plugin)
        return factory_fn

    return decorator


def create_storage(config: StorageConfig) -> StorageBackend:
    """Create storage backend from config using plugin registry.

    Args:
        config: Storage configuration with backend name and backend-specific settings

    Returns:
        Instantiated storage backend

    Raises:
        RuntimeError: If backend is unknown or backend-specific config is missing
    """
    backend_name = config.backend.lower()

    if backend_name not in STORAGE_REGISTRY:
        available = ", ".join(sorted(STORAGE_REGISTRY.keys()))
        raise RuntimeError(
            f"Unknown storage backend: '{backend_name}'. Available: {available}"
        )

    plugin = STORAGE_REGISTRY[backend_name]

    # Extract backend-specific config using attribute access
    # e.g., config.dropbox, config.local
    specific_config = getattr(config, backend_name, None)
    if specific_config is None:
        raise RuntimeError(
            f"Missing '{backend_name}' config in storage section. "
            f"Add 'storage.{backend_name}' to your config."
        )

    return plugin.factory(specific_config)


__all__ = [
    "StoragePlugin",
    "StorageFactory",
    "STORAGE_REGISTRY",
    "register_storage",
    "storage_plugin",
    "create_storage",
]
