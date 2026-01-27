"""RTSP source plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homesec.interfaces import ClipSource
from homesec.models.source import RTSPSourceConfig
from homesec.plugins.registry import PluginType, plugin
from homesec.sources.rtsp.core import RTSPSource as RTSPSourceImpl

if TYPE_CHECKING:
    pass


@plugin(plugin_type=PluginType.SOURCE, name="rtsp")
class RTSPPluginSource(RTSPSourceImpl):
    """RTSP source plugin wrapper."""

    config_cls = RTSPSourceConfig

    @classmethod
    def create(cls, config: RTSPSourceConfig) -> ClipSource:
        # RTSPSourceImpl expects config and camera_name
        if config.camera_name is None:
            raise ValueError("camera_name is required for RTSPSource")
        return cls(config=config, camera_name=config.camera_name)
