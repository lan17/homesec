"""Notifier plugins and registry."""

from __future__ import annotations

import logging
from typing import Any, cast

from pydantic import BaseModel

from homesec.interfaces import Notifier
from homesec.plugins.registry import PluginType, load_plugin

logger = logging.getLogger(__name__)


def load_notifier_plugin(
    backend: str,
    config: dict[str, object] | BaseModel,
    **runtime_context: Any,
) -> Notifier:
    """Load and instantiate a notifier plugin.

    Args:
        backend: Notifier backend name (e.g., "mqtt", "sendgrid_email")
        config: Raw config dict or already-validated BaseModel

    Returns:
        Configured Notifier instance

    Raises:
        ValueError: If backend not found in registry
        ValidationError: If config validation fails
    """
    return cast(
        Notifier,
        load_plugin(
            PluginType.NOTIFIER,
            backend,
            config,
            **runtime_context,
        ),
    )


__all__ = ["load_notifier_plugin"]
