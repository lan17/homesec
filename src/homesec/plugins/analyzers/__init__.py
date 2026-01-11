"""VLM analyzer plugins and registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from homesec.interfaces import VLMAnalyzer

if TYPE_CHECKING:
    from homesec.models.vlm import VLMConfig

logger = logging.getLogger(__name__)

# Type alias for clarity
VLMFactory = Callable[["VLMConfig"], VLMAnalyzer]


@dataclass(frozen=True)
class VLMPlugin:
    """Metadata for a VLM analyzer plugin."""

    name: str
    config_model: type[BaseModel]
    factory: VLMFactory


VLM_REGISTRY: dict[str, VLMPlugin] = {}


def register_vlm(plugin: VLMPlugin) -> None:
    """Register a VLM plugin with collision detection.

    Args:
        plugin: VLM plugin to register

    Raises:
        ValueError: If a plugin with the same name is already registered
    """
    if plugin.name in VLM_REGISTRY:
        raise ValueError(
            f"VLM plugin '{plugin.name}' is already registered. "
            f"Plugin names must be unique across all VLM plugins."
        )
    VLM_REGISTRY[plugin.name] = plugin


T = TypeVar("T", bound=Callable[[], VLMPlugin])


def vlm_plugin(name: str) -> Callable[[T], T]:
    """Decorator to register a VLM analyzer plugin.

    Usage:
        @vlm_plugin(name="my_vlm")
        def my_vlm_plugin() -> VLMPlugin:
            return VLMPlugin(...)

    Args:
        name: Plugin name (for validation only - must match plugin.name)

    Returns:
        Decorator function that registers the plugin
    """

    def decorator(factory_fn: T) -> T:
        plugin = factory_fn()
        register_vlm(plugin)
        return factory_fn

    return decorator


def load_vlm_plugin(config: VLMConfig) -> VLMAnalyzer:
    """Load VLM plugin by name from config.

    Validates the llm dict against the plugin's config_model and creates
    a VLMConfig with the validated settings object.

    Args:
        config: VLM configuration with backend name and raw llm dict

    Returns:
        Instantiated VLM plugin

    Raises:
        ValueError: If plugin name is unknown or config validation fails
    """
    plugin_name = config.backend.lower()

    if plugin_name not in VLM_REGISTRY:
        available = ", ".join(sorted(VLM_REGISTRY.keys()))
        raise ValueError(
            f"Unknown VLM plugin: '{plugin_name}'. Available: {available}"
        )

    plugin = VLM_REGISTRY[plugin_name]

    # Validate config.llm dict against plugin's config_model
    validated_llm_settings = plugin.config_model.model_validate(config.llm)

    # Create new VLMConfig with validated llm settings object
    from homesec.models.vlm import VLMConfig as VLMConfigModel

    validated_config = VLMConfigModel(
        backend=config.backend,
        trigger_classes=config.trigger_classes,
        max_workers=config.max_workers,
        llm=validated_llm_settings,
        preprocessing=config.preprocessing,
    )

    return plugin.factory(validated_config)


__all__ = [
    "VLMPlugin",
    "VLMFactory",
    "VLM_REGISTRY",
    "register_vlm",
    "vlm_plugin",
    "load_vlm_plugin",
]
