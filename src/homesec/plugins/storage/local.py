"""Local filesystem storage backend."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path, PurePosixPath

from homesec.interfaces import StorageBackend
from homesec.models.config import LocalStorageConfig
from homesec.models.storage import StorageUploadResult
from homesec.plugins.registry import PluginType, plugin

logger = logging.getLogger(__name__)


@plugin(plugin_type=PluginType.STORAGE, name="local")
class LocalStorage(StorageBackend):
    """Local storage backend for development and tests."""

    config_cls = LocalStorageConfig

    @classmethod
    def create(cls, config: LocalStorageConfig) -> StorageBackend:
        return cls(config)

    def __init__(self, config: LocalStorageConfig) -> None:
        self.root = Path(config.root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._shutdown_called = False

    async def put_file(self, local_path: Path, dest_path: str) -> StorageUploadResult:
        self._ensure_open()
        dest = self._full_dest_path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, local_path, dest)
        storage_uri = f"local:{dest}"
        view_url = f"file://{dest}"
        return StorageUploadResult(storage_uri=storage_uri, view_url=view_url)

    async def get_view_url(self, storage_uri: str) -> str | None:
        if not storage_uri.startswith("local:"):
            return None
        return f"file://{storage_uri[6:]}"

    async def get(self, storage_uri: str, local_path: Path) -> None:
        self._ensure_open()
        src = self._parse_storage_uri(storage_uri)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, src, local_path)

    async def exists(self, storage_uri: str) -> bool:
        self._ensure_open()
        try:
            path = self._parse_storage_uri(storage_uri)
        except ValueError:
            return False
        return await asyncio.to_thread(path.exists)

    async def delete(self, storage_uri: str) -> None:
        self._ensure_open()
        path = self._parse_storage_uri(storage_uri)
        await asyncio.to_thread(path.unlink, True)

    async def ping(self) -> bool:
        return self.root.exists() and self.root.is_dir()

    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout
        self._shutdown_called = True

    def _ensure_open(self) -> None:
        if self._shutdown_called:
            raise RuntimeError("Storage has been shut down")

    def _parse_storage_uri(self, storage_uri: str) -> Path:
        if not storage_uri.startswith("local:"):
            raise ValueError(f"Invalid storage_uri: {storage_uri}")
        return Path(storage_uri[6:])

    def _full_dest_path(self, dest_path: str) -> Path:
        cleaned = str(dest_path).lstrip("/")
        if not cleaned or "\\" in cleaned:
            raise ValueError(f"Invalid dest_path: {dest_path}")
        path = PurePosixPath(cleaned)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Invalid dest_path: {dest_path}")
        return self.root.joinpath(*path.parts)
