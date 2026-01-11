"""Notifier plugins and registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, TypeVar

from pydantic import BaseModel

from homesec.interfaces import Notifier
from homesec.plugins.notifiers.multiplex import MultiplexNotifier, NotifierEntry

logger = logging.getLogger(__name__)


NotifierFactory = Callable[[BaseModel], Notifier]


@dataclass(frozen=True)
class NotifierPlugin:
    name: str
    config_model: type[BaseModel]
    factory: NotifierFactory


NOTIFIER_REGISTRY: dict[str, NotifierPlugin] = {}


def register_notifier(plugin: NotifierPlugin) -> None:
    """Register a notifier plugin with collision detection.

    Args:
        plugin: Notifier plugin to register

    Raises:
        ValueError: If a plugin with the same name is already registered
    """
    if plugin.name in NOTIFIER_REGISTRY:
        raise ValueError(
            f"Notifier plugin '{plugin.name}' is already registered. "
            f"Plugin names must be unique across all notifier plugins."
        )
    NOTIFIER_REGISTRY[plugin.name] = plugin


T = TypeVar("T", bound=Callable[[], NotifierPlugin])


def notifier_plugin(name: str) -> Callable[[T], T]:
    """Decorator to register a notifier plugin.

    Usage:
        @notifier_plugin(name="my_notifier")
        def my_notifier_plugin() -> NotifierPlugin:
            return NotifierPlugin(...)

    Args:
        name: Plugin name (for validation only - must match plugin.name)

    Returns:
        Decorator function that registers the plugin
    """

    def decorator(factory_fn: T) -> T:
        plugin = factory_fn()
        register_notifier(plugin)
        return factory_fn

    return decorator


__all__ = [
    "MultiplexNotifier",
    "NotifierEntry",
    "NotifierPlugin",
    "NOTIFIER_REGISTRY",
    "register_notifier",
    "notifier_plugin",
]
