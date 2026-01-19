"""Filter plugins and registry."""

from __future__ import annotations

import logging
from typing import cast

from homesec.interfaces import ObjectFilter
from homesec.models.filter import FilterConfig
from homesec.plugins.registry import PluginType, load_plugin

logger = logging.getLogger(__name__)


def load_filter(config: FilterConfig) -> ObjectFilter:
    """Load and instantiate a filter plugin.

    Args:
        config: Filter configuration

    Returns:
        Configured ObjectFilter instance

    Raises:
        ValueError: If plugin not found in registry
        ValidationError: If config validation fails
    """
    return cast(
        ObjectFilter,
        load_plugin(
            PluginType.FILTER,
            config.plugin,
            config.config,
            max_workers=config.max_workers,
        ),
    )


__all__ = ["load_filter"]
