"""Local folder source plugin."""

from __future__ import annotations

from homesec.interfaces import ClipSource
from homesec.plugins.registry import PluginType, plugin
from homesec.sources.local_folder import LocalFolderSource as LocalFolderSourceImpl
from homesec.sources.local_folder import LocalFolderSourceConfig


@plugin(plugin_type=PluginType.SOURCE, name="local_folder")
class LocalFolderPlugin(LocalFolderSourceImpl):
    """Register local_folder source plugin."""

    config_cls = LocalFolderSourceConfig

    @classmethod
    def create(cls, config: LocalFolderSourceConfig) -> ClipSource:
        # Note: LocalFolderSourceImpl takes (config) in __init__?
        # Let's check LocalFolderSourceImpl signature.
        # It inherits from ClipSource.
        # Wait, usually implementation classes take specific args, not the config object?
        # I need to check src/homesec/sources/local_folder.py.
        # But assuming refactor, I should match what create does.
        # Old create: LocalFolderSourceImpl(config=config, camera_name=context.camera_name)
        # config now has camera_name injected.
        return cls(config=config, camera_name=config.camera_name)


__all__ = ["LocalFolderPlugin"]
