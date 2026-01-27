"""FTP source plugin."""

from __future__ import annotations

from homesec.interfaces import ClipSource
from homesec.models.source.ftp import FtpSourceConfig
from homesec.plugins.registry import PluginType, plugin

# Import the actual implementation from sources module
from homesec.sources.ftp import FtpSource as FtpSourceImpl


@plugin(plugin_type=PluginType.SOURCE, name="ftp")
class FtpPluginSource(FtpSourceImpl):
    """FTP source plugin wrapper."""

    config_cls = FtpSourceConfig

    @classmethod
    def create(cls, config: FtpSourceConfig) -> ClipSource:
        # FtpSourceImpl expects config and camera_name
        # config.camera_name is populated by registry
        if config.camera_name is None:
            raise ValueError("camera_name is required for FtpSource")
        return cls(config=config, camera_name=config.camera_name)
