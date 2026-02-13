"""Configuration persistence manager for HomeSec."""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import yaml
from pydantic import BaseModel

from homesec.config.loader import (
    ConfigError,
    load_config,
    load_config_from_dict,
    load_config_or_bootstrap,
)
from homesec.models.bootstrap import BootstrapConfig
from homesec.models.config import CameraConfig, CameraSourceConfig, Config, NotifierConfig


class ConfigUpdateResult(BaseModel):
    """Result of a config update operation."""

    restart_required: bool = True


class ConfigManager:
    """Manages configuration persistence (single file, last-write-wins).

    On mutations, backs up current config to {path}.bak before overwriting.
    """

    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path

    def get_config(self) -> Config:
        """Get the current configuration."""
        return load_config(self._config_path)

    def get_config_or_bootstrap(self) -> Config | BootstrapConfig:
        """Get the current configuration or bootstrap config."""
        return load_config_or_bootstrap(self._config_path)

    async def add_camera(
        self,
        name: str,
        enabled: bool,
        source_backend: str,
        source_config: dict[str, object],
    ) -> ConfigUpdateResult:
        """Add a new camera to the config."""
        config = await asyncio.to_thread(self.get_config_or_bootstrap)
        if isinstance(config, BootstrapConfig):
            raise ValueError("HomeSec is in bootstrap mode; configure full settings first")

        if any(camera.name == name for camera in config.cameras):
            raise ValueError(f"Camera already exists: {name}")

        config.cameras.append(
            CameraConfig(
                name=name,
                enabled=enabled,
                source=CameraSourceConfig(backend=source_backend, config=source_config),
            )
        )

        validated = await self._validate_config(config)
        await self._save_config(validated)
        return ConfigUpdateResult()

    async def update_camera(
        self,
        camera_name: str,
        enabled: bool | None,
        source_config: dict[str, object] | None,
    ) -> ConfigUpdateResult:
        """Update an existing camera in the config."""
        config = await asyncio.to_thread(self.get_config_or_bootstrap)
        if isinstance(config, BootstrapConfig):
            raise ValueError("HomeSec is in bootstrap mode; configure full settings first")

        camera = next((cam for cam in config.cameras if cam.name == camera_name), None)
        if camera is None:
            raise ValueError(f"Camera not found: {camera_name}")

        if enabled is not None:
            camera.enabled = enabled
        if source_config is not None:
            camera.source = CameraSourceConfig(
                backend=camera.source.backend,
                config=source_config,
            )

        validated = await self._validate_config(config)
        await self._save_config(validated)
        return ConfigUpdateResult()

    async def remove_camera(
        self,
        camera_name: str,
    ) -> ConfigUpdateResult:
        """Remove a camera from the config."""
        config = await asyncio.to_thread(self.get_config_or_bootstrap)
        if isinstance(config, BootstrapConfig):
            raise ValueError("HomeSec is in bootstrap mode; configure full settings first")

        updated = [camera for camera in config.cameras if camera.name != camera_name]
        if len(updated) == len(config.cameras):
            raise ValueError(f"Camera not found: {camera_name}")

        config.cameras = updated

        validated = await self._validate_config(config)
        await self._save_config(validated)
        return ConfigUpdateResult()

    async def upsert_notifier(
        self,
        backend: str,
        enabled: bool | None = True,
        config: dict[str, object] | None = None,
    ) -> ConfigUpdateResult:
        """Create or update a notifier configuration entry."""
        current = await asyncio.to_thread(self.get_config_or_bootstrap)
        backend_key = backend.lower()
        notifier = next(
            (entry for entry in current.notifiers if entry.backend.lower() == backend_key),
            None,
        )

        if notifier is None:
            current.notifiers.append(
                NotifierConfig(
                    backend=backend,
                    enabled=enabled if enabled is not None else True,
                    config=config or {},
                )
            )
        else:
            if enabled is not None:
                notifier.enabled = enabled
            if config is not None:
                existing = _config_to_dict(notifier.config)
                for key, value in config.items():
                    if key not in existing or existing[key] in (None, ""):
                        existing[key] = value
                notifier.config = existing

        if isinstance(current, BootstrapConfig):
            await self._save_config(current)
            return ConfigUpdateResult()

        validated = await self._validate_config(current)
        await self._save_config(validated)
        return ConfigUpdateResult()

    async def _validate_config(self, config: Config) -> Config:
        """Validate configuration via the standard loader path."""
        payload = config.model_dump(mode="json")
        try:
            return await asyncio.to_thread(load_config_from_dict, payload)
        except ConfigError as exc:
            raise ValueError(str(exc)) from exc

    async def _save_config(self, config: BaseModel) -> None:
        """Save config to disk with backup."""

        def _write() -> None:
            backup_path = Path(str(self._config_path) + ".bak")
            if self._config_path.exists():
                shutil.copy2(self._config_path, backup_path)

            payload = config.model_dump(mode="json")
            tmp_path = self._config_path.with_suffix(self._config_path.suffix + ".tmp")
            tmp_path.parent.mkdir(parents=True, exist_ok=True)

            with tmp_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(tmp_path, self._config_path)

        await asyncio.to_thread(_write)


def _config_to_dict(value: dict[str, object] | BaseModel) -> dict[str, object]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value)
