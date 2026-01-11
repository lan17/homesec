"""Alert policy plugins and registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, TypeVar

from pydantic import BaseModel

from homesec.interfaces import AlertPolicy
from homesec.models.config import AlertPolicyOverrides

logger = logging.getLogger(__name__)


AlertPolicyFactory = Callable[
    [BaseModel, dict[str, AlertPolicyOverrides], list[str]], AlertPolicy
]


@dataclass(frozen=True)
class AlertPolicyPlugin:
    name: str
    config_model: type[BaseModel]
    factory: AlertPolicyFactory


ALERT_POLICY_REGISTRY: dict[str, AlertPolicyPlugin] = {}


def register_alert_policy(plugin: AlertPolicyPlugin) -> None:
    """Register an alert policy plugin with collision detection.

    Args:
        plugin: Alert policy plugin to register

    Raises:
        ValueError: If a plugin with the same name is already registered
    """
    if plugin.name in ALERT_POLICY_REGISTRY:
        raise ValueError(
            f"Alert policy plugin '{plugin.name}' is already registered. "
            f"Plugin names must be unique across all alert policy plugins."
        )
    ALERT_POLICY_REGISTRY[plugin.name] = plugin


T = TypeVar("T", bound=Callable[[], AlertPolicyPlugin])


def alert_policy_plugin(name: str) -> Callable[[T], T]:
    """Decorator to register an alert policy plugin.

    Usage:
        @alert_policy_plugin(name="my_policy")
        def my_policy_plugin() -> AlertPolicyPlugin:
            return AlertPolicyPlugin(...)

    Args:
        name: Plugin name (for validation only - must match plugin.name)

    Returns:
        Decorator function that registers the plugin
    """

    def decorator(factory_fn: T) -> T:
        plugin = factory_fn()
        register_alert_policy(plugin)
        return factory_fn

    return decorator


__all__ = [
    "AlertPolicyPlugin",
    "ALERT_POLICY_REGISTRY",
    "register_alert_policy",
    "alert_policy_plugin",
]
