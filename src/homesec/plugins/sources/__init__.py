"""Source plugins and registry."""

from __future__ import annotations

import logging
from typing import cast

from pydantic import BaseModel

from homesec.interfaces import ClipSource
from homesec.plugins.registry import PluginType, load_plugin

logger = logging.getLogger(__name__)


def load_source_plugin(
    source_backend: str,
    config: dict[str, object] | BaseModel,
    camera_name: str,
) -> ClipSource:
    """Load and instantiate a source plugin.

    Args:
        source_backend: Name of the source plugin (e.g., "rtsp", "local_folder")
        config: Raw config dict or validated BaseModel
        camera_name: Name of the camera (runtime context)

    Returns:
        Instantiated ClipSource

    Raises:
        ValueError: If source_backend is unknown or config validation fails
    """
    return cast(
        ClipSource,
        load_plugin(
            PluginType.SOURCE,
            source_backend,
            config,
            camera_name=camera_name,
        ),
    )


__all__ = ["load_source_plugin"]
