"""Filter plugins and registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from homesec.interfaces import ObjectFilter

if TYPE_CHECKING:
    from homesec.models.filter import FilterConfig

logger = logging.getLogger(__name__)

# Type alias for clarity
FilterFactory = Callable[["FilterConfig"], ObjectFilter]


@dataclass(frozen=True)
class FilterPlugin:
    """Metadata for a filter plugin."""

    name: str
    config_model: type[BaseModel]
    factory: FilterFactory


FILTER_REGISTRY: dict[str, FilterPlugin] = {}


def register_filter(plugin: FilterPlugin) -> None:
    """Register a filter plugin with collision detection.

    Args:
        plugin: Filter plugin to register

    Raises:
        ValueError: If a plugin with the same name is already registered
    """
    if plugin.name in FILTER_REGISTRY:
        raise ValueError(
            f"Filter plugin '{plugin.name}' is already registered. "
            f"Plugin names must be unique across all filter plugins."
        )
    FILTER_REGISTRY[plugin.name] = plugin


T = TypeVar("T", bound=Callable[[], FilterPlugin])


def filter_plugin(name: str) -> Callable[[T], T]:
    """Decorator to register a filter plugin.

    Usage:
        @filter_plugin(name="my_filter")
        def my_filter_plugin() -> FilterPlugin:
            return FilterPlugin(...)

    Args:
        name: Plugin name (for validation only - must match plugin.name)

    Returns:
        Decorator function that registers the plugin
    """

    def decorator(factory_fn: T) -> T:
        plugin = factory_fn()
        register_filter(plugin)
        return factory_fn

    return decorator


def load_filter_plugin(config: FilterConfig) -> ObjectFilter:
    """Load filter plugin by name from config.

    Validates the config dict against the plugin's config_model and creates
    a FilterConfig with the validated settings object.

    Args:
        config: Filter configuration with plugin name and raw config dict

    Returns:
        Instantiated filter plugin

    Raises:
        ValueError: If plugin name is unknown or config validation fails
    """
    plugin_name = config.plugin.lower()

    if plugin_name not in FILTER_REGISTRY:
        available = ", ".join(sorted(FILTER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown filter plugin: '{plugin_name}'. Available: {available}"
        )

    plugin = FILTER_REGISTRY[plugin_name]

    # Validate config.config dict against plugin's config_model
    validated_settings = plugin.config_model.model_validate(config.config)

    # Create new FilterConfig with validated settings object
    from homesec.models.filter import FilterConfig as FilterConfigModel

    validated_config = FilterConfigModel(
        plugin=config.plugin,
        max_workers=config.max_workers,
        config=validated_settings,
    )

    return plugin.factory(validated_config)


__all__ = [
    "FilterPlugin",
    "FilterFactory",
    "FILTER_REGISTRY",
    "register_filter",
    "filter_plugin",
    "load_filter_plugin",
]
