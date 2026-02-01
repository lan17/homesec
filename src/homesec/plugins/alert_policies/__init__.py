"""Alert policy plugins and registry."""

from __future__ import annotations

import logging
from typing import Any, cast

from homesec.interfaces import AlertPolicy
from homesec.plugins.alert_policies.noop import NoopAlertPolicySettings
from homesec.plugins.registry import PluginType, load_plugin

logger = logging.getLogger(__name__)


def load_alert_policy(
    config: Any,  # AlertPolicyConfig but trying to avoid circular import if possible
    trigger_classes: list[str],
) -> AlertPolicy:
    """Load and instantiate an alert policy plugin.

    Args:
        config: Alert policy configuration (AlertPolicyConfig)
        trigger_classes: List of object classes that trigger analysis

    Returns:
        Configured AlertPolicy instance
    """
    # Handle disabled -> noop fallback
    if not config.enabled:
        return cast(
            AlertPolicy,
            load_plugin(
                PluginType.ALERT_POLICY,
                "noop",
                NoopAlertPolicySettings(),
            ),
        )

    runtime_context: dict[str, Any] = {}
    if config.backend == "default":
        runtime_context["trigger_classes"] = list(trigger_classes)

    return cast(
        AlertPolicy,
        load_plugin(
            PluginType.ALERT_POLICY,
            config.backend,
            config.config,
            **runtime_context,
        ),
    )


__all__ = ["load_alert_policy"]
