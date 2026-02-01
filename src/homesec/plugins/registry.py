"""Unified Plugin Registry for HomeSec.

This module provides the core infrastructure for the Class-Based Plugin Architecture.
It handles plugin registration, discovery, and strict configuration validation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar, cast

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PluginType(str, Enum):
    """Categorization of plugin types."""

    SOURCE = "source"
    FILTER = "filter"
    ANALYZER = "analyzer"
    STORAGE = "storage"
    NOTIFIER = "notifier"
    ALERT_POLICY = "alert_policy"


ConfigT = TypeVar("ConfigT", bound=BaseModel)
PluginInterfaceT = TypeVar("PluginInterfaceT", bound=object, covariant=True)


class PluginProtocol(Protocol[ConfigT, PluginInterfaceT]):
    """Protocol defining the structure of a valid HomeSec plugin class."""

    config_cls: type[ConfigT]

    @classmethod
    def create(cls, config: ConfigT) -> PluginInterfaceT:
        """Factory method to create the plugin instance."""
        ...


class PluginRegistry(Generic[ConfigT, PluginInterfaceT]):
    """Generic registry for a specific type of plugin."""

    def __init__(self, plugin_type: PluginType) -> None:
        self.plugin_type = plugin_type
        self._plugins: dict[str, type[PluginProtocol[ConfigT, PluginInterfaceT]]] = {}

    def register(
        self, name: str, plugin_cls: type[PluginProtocol[ConfigT, PluginInterfaceT]]
    ) -> None:
        """Register a plugin class."""
        if name in self._plugins:
            raise ValueError(f"{self.plugin_type} plugin '{name}' is already registered.")

        self._plugins[name] = plugin_cls
        logger.debug("Registered %s plugin: %s", self.plugin_type, name)

    def load(
        self, name: str, config_dict: dict[str, Any], **runtime_context: Any
    ) -> PluginInterfaceT:
        """Load and instantiate a plugin.

        Args:
            name: The name of the plugin to load.
            config_dict: Raw configuration dictionary (from YAML/JSON).
            **runtime_context: Key-value pairs to inject into the config *before* validation.
                               (e.g. camera_name="front_door")

        Returns:
            An instantiated and configured plugin object.

        Raises:
            ValueError: If the plugin name is unknown.
            ValidationError: If configuration is invalid.
        """
        if name not in self._plugins:
            available = ", ".join(sorted(self._plugins.keys()))
            raise ValueError(f"Unknown {self.plugin_type} plugin: '{name}'. Available: {available}")

        plugin_cls = self._plugins[name]

        # 1. Inject runtime context into config (if the config model supports those fields)
        # We merge it into the raw dict so Pydantic can validate it.
        # This allows injecting "camera_name" into SourceConfig, etc.
        merged_config = config_dict.copy()
        merged_config.update(runtime_context)

        # 2. Validate configuration
        validated_config = plugin_cls.config_cls.model_validate(merged_config)

        # 3. Create instance
        return plugin_cls.create(validated_config)

    def validate(self, name: str, config_dict: dict[str, Any], **runtime_context: Any) -> BaseModel:
        """Validate configuration for a plugin without instantiating it."""
        if name not in self._plugins:
            available = ", ".join(sorted(self._plugins.keys()))
            raise ValueError(f"Unknown {self.plugin_type} plugin: '{name}'. Available: {available}")

        plugin_cls = self._plugins[name]

        merged_config = config_dict.copy()
        merged_config.update(runtime_context)

        return plugin_cls.config_cls.model_validate(merged_config)

    def get_all(self) -> dict[str, type[PluginProtocol[ConfigT, PluginInterfaceT]]]:
        """Return all registered plugins."""
        return self._plugins.copy()


# Global Registry Storage
# We keep separate registries per type for strict typing
_REGISTRIES: dict[PluginType, PluginRegistry[Any, Any]] = {t: PluginRegistry(t) for t in PluginType}


def plugin(plugin_type: PluginType, name: str) -> Callable[[type], type]:
    """Decorator to register a class as a plugin.

    Args:
        plugin_type: The category of plugin (SOURCE, FILTER, etc.)
        name: The unique name for this plugin (e.g., "rtsp", "yolo")
    """

    def decorator(cls: type) -> type:
        # Runtime verification could happen here, or we trust static analysis/Protocol
        if not hasattr(cls, "config_cls"):
            raise TypeError(f"Plugin class {cls.__name__} must define 'config_cls'")
        if not hasattr(cls, "create"):
            raise TypeError(f"Plugin class {cls.__name__} must define 'create' classmethod")

        registry = _REGISTRIES[plugin_type]
        # We cast because the decorator is generic but _REGISTRIES holds specific types
        registry.register(name, cls)

        # Attach metadata for inspection if needed
        cast(Any, cls).__plugin_name__ = name
        cast(Any, cls).__plugin_type__ = plugin_type

        return cls

    return decorator


def load_plugin(
    plugin_type: PluginType, name: str, config: dict[str, Any] | BaseModel, **runtime_context: Any
) -> Any:
    """Public API to load any plugin.

    Args:
        plugin_type: The enum type of plugin to load.
        name: The plugin name (e.g. "rtsp").
        config: The raw config dict OR an already-validated BaseModel.
        runtime_context: Dependencies to inject (camera_name, etc.)
    """
    registry = _REGISTRIES[plugin_type]

    # Handle case where config is already a BaseModel (e.g. nested in AppConfig)
    if isinstance(config, BaseModel):
        config_dict = config.model_dump()
    else:
        config_dict = config

    return registry.load(name, config_dict, **runtime_context)


def validate_plugin(
    plugin_type: PluginType, name: str, config: dict[str, Any] | BaseModel, **runtime_context: Any
) -> BaseModel:
    """Validate plugin configuration without instantiating it."""
    registry = _REGISTRIES[plugin_type]

    if isinstance(config, BaseModel):
        config_dict = config.model_dump()
    else:
        config_dict = config

    return registry.validate(name, config_dict, **runtime_context)


def get_plugin_names(plugin_type: PluginType) -> list[str]:
    """Get list of registered plugin names for a given type."""
    return sorted(_REGISTRIES[plugin_type].get_all().keys())
