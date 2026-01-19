"""Alert policy plugins and registry."""

from __future__ import annotations

import logging
from typing import Any, cast

from homesec.interfaces import AlertPolicy
from homesec.models.config import AlertPolicyOverrides
from homesec.plugins.alert_policies.noop import NoopAlertPolicySettings
from homesec.plugins.registry import PluginType, load_plugin

logger = logging.getLogger(__name__)


def load_alert_policy(
    config: Any,  # AlertPolicyConfig but trying to avoid circular import if possible
    per_camera_overrides: dict[str, AlertPolicyOverrides],
    trigger_classes: list[str],
) -> AlertPolicy:
    """Load and instantiate an alert policy plugin.

    Args:
        config: Alert policy configuration (AlertPolicyConfig)
        per_camera_overrides: Map of camera name to override settings
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

    return cast(
        AlertPolicy,
        load_plugin(
            PluginType.ALERT_POLICY,
            config.backend,
            config.config,
            overrides=per_camera_overrides,
            trigger_classes=trigger_classes,
        ),
    )


__all__ = ["load_alert_policy"]
