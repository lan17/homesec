"""Analyzer plugins and registry."""

from __future__ import annotations

import logging
from typing import cast

from homesec.interfaces import VLMAnalyzer
from homesec.models.vlm import VLMConfig
from homesec.plugins.registry import PluginType, load_plugin

logger = logging.getLogger(__name__)


def load_analyzer(config: VLMConfig) -> VLMAnalyzer:
    """Load and instantiate a VLM analyzer plugin.

    Args:
        config: VLM configuration

    Returns:
        Configured VLMAnalyzer instance

    Raises:
        ValueError: If backend not found in registry
        ValidationError: If config validation fails
    """
    return cast(
        VLMAnalyzer,
        load_plugin(
            PluginType.ANALYZER,
            config.backend,
            config.config,
        ),
    )


__all__ = ["load_analyzer"]
