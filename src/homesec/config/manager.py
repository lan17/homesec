"""Configuration persistence manager for HomeSec."""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from homesec.config.errors import (
    CameraAlreadyExistsError,
    CameraConfigInvalidError,
    CameraConfigRedactedPlaceholderError,
    CameraNotFoundError,
    ConfigApplyInProgressError,
    ConfigBackendChangeUnsupportedError,
    ConfigPatchInvalidError,
    ConfigSaveError,
    ConfigVersionConflictError,
)
from homesec.config.loader import ConfigError, config_signature, load_config, load_config_from_dict
from homesec.models.config import CameraConfig, CameraSourceConfig, Config
from homesec.models.enums import VLMRunMode

_SENSITIVE_CONFIG_FILE_MODE = 0o600
_REDACTED_PLACEHOLDER = "***redacted***"


class ConfigUpdateResult(BaseModel):
    """Result of a config update operation."""

    restart_required: bool = True


class _ConfigPatchModel(BaseModel):
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _reject_explicit_null(self) -> _ConfigPatchModel:
        for field in self.model_fields_set:
            if getattr(self, field) is None:
                raise ValueError(f"{field} must not be null; omit unchanged fields")
        return self


class PluginConfigPatch(_ConfigPatchModel):
    """Edit the existing backend's opaque configuration without replacing it."""

    backend: str | None = None
    config: dict[str, object] | None = None

    @field_validator("backend", mode="before")
    @classmethod
    def _normalize_backend(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value


class StoragePathsPatch(_ConfigPatchModel):
    clips_dir: str | None = None
    backups_dir: str | None = None
    artifacts_dir: str | None = None


class StorageConfigPatch(PluginConfigPatch):
    paths: StoragePathsPatch | None = None


class VLMPreprocessPatch(_ConfigPatchModel):
    max_frames: int | None = None
    max_size: int | None = None
    quality: int | None = None


class VLMConfigPatch(PluginConfigPatch):
    run_mode: VLMRunMode | None = None
    trigger_classes: list[str] | None = None
    preprocessing: VLMPreprocessPatch | None = None


class AlertPolicyConfigPatch(PluginConfigPatch):
    enabled: bool | None = None


class NotifierConfigPatch(_ConfigPatchModel):
    """Patch an existing ordered notifier entry, preserving all other entries."""

    index: int = Field(ge=0)
    enabled: bool | None = None
    config: dict[str, object] | None = None


class ConfigPatch(_ConfigPatchModel):
    """The supported save-only edits to the canonical configuration document."""

    expected_config_version: str = Field(min_length=1)
    storage: StorageConfigPatch | None = None
    filter: PluginConfigPatch | None = None
    vlm: VLMConfigPatch | None = None
    alert_policy: AlertPolicyConfigPatch | None = None
    notifiers: list[NotifierConfigPatch] | None = None


class ConfigManager:
    """Manages configuration persistence (single file, last-write-wins).

    On mutations, backs up current config to {path}.bak before overwriting.
    """

    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        self._lock: asyncio.Lock | None = None
        self._mutations_frozen = False

    def freeze_mutations(self) -> None:
        """Stop future writes after accepting a restart of this process."""
        self._mutations_frozen = True

    def _mutation_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def config_path(self) -> Path:
        """Path to the managed config file."""
        return self._config_path

    @staticmethod
    def _enforce_sensitive_file_mode(path: Path) -> None:
        """Enforce restrictive config file mode on POSIX platforms."""
        if os.name != "posix":
            return
        os.chmod(path, _SENSITIVE_CONFIG_FILE_MODE)

    def get_config(self) -> Config:
        """Get the current configuration."""
        return load_config(self._config_path)

    @staticmethod
    def _to_source_config_dict(config: dict[str, object] | BaseModel) -> dict[str, object]:
        if isinstance(config, BaseModel):
            payload = config.model_dump(mode="json")
            return cast(dict[str, object], payload)
        return dict(config)

    @classmethod
    def _contains_redacted_placeholder(cls, value: object) -> bool:
        if isinstance(value, dict):
            return any(cls._contains_redacted_placeholder(nested) for nested in value.values())
        if isinstance(value, list):
            return any(cls._contains_redacted_placeholder(item) for item in value)
        if isinstance(value, str):
            return _REDACTED_PLACEHOLDER in value
        return False

    @classmethod
    def _merge_config(
        cls,
        existing: dict[str, object],
        patch: dict[str, object],
    ) -> dict[str, object]:
        """Apply partial update semantics to configuration payloads.

        - omitted key => unchanged
        - value is null => clear key
        - nested dict values => recursive merge
        - all other values => replace
        """
        merged: dict[str, object] = dict(existing)
        for key, patch_value in patch.items():
            current_value = merged.get(key)
            if patch_value is None:
                merged.pop(key, None)
                continue
            if isinstance(current_value, dict) and isinstance(patch_value, dict):
                merged[key] = cls._merge_config(
                    cast(dict[str, object], current_value),
                    cast(dict[str, object], patch_value),
                )
                continue
            merged[key] = patch_value
        return merged

    async def add_camera(
        self,
        name: str,
        enabled: bool,
        source_backend: str,
        source_config: dict[str, object],
    ) -> ConfigUpdateResult:
        """Add a new camera to the config."""
        async with self._mutation_lock():
            config = await asyncio.to_thread(self.get_config)

            if any(camera.name == name for camera in config.cameras):
                raise CameraAlreadyExistsError(f"Camera already exists: {name}")

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
        source_backend: str | None,
        source_config: dict[str, object] | None,
    ) -> ConfigUpdateResult:
        """Update an existing camera in the config."""
        async with self._mutation_lock():
            config = await asyncio.to_thread(self.get_config)

            camera = next((cam for cam in config.cameras if cam.name == camera_name), None)
            if camera is None:
                raise CameraNotFoundError(f"Camera not found: {camera_name}")

            if enabled is not None:
                camera.enabled = enabled

            should_update_source = source_backend is not None or source_config is not None
            if should_update_source:
                if source_config is not None and self._contains_redacted_placeholder(source_config):
                    raise CameraConfigRedactedPlaceholderError(
                        "source_config patch contains redacted placeholders; "
                        "omit unchanged fields or provide replacement values"
                    )

                current_source_config = self._to_source_config_dict(camera.source.config)
                next_backend = (
                    source_backend if source_backend is not None else camera.source.backend
                )
                base_source_config = (
                    {}
                    if source_backend is not None and source_backend != camera.source.backend
                    else current_source_config
                )
                next_source_config = (
                    self._merge_config(base_source_config, source_config)
                    if source_config is not None
                    else base_source_config
                )

                try:
                    camera.source = CameraSourceConfig(
                        backend=next_backend,
                        config=next_source_config,
                    )
                except (ValidationError, ValueError, TypeError) as exc:
                    raise CameraConfigInvalidError(
                        f"Invalid source configuration for camera '{camera_name}'",
                        cause=exc,
                    ) from exc

            validated = await self._validate_config(config)
            await self._save_config(validated)
            return ConfigUpdateResult()

    async def remove_camera(
        self,
        camera_name: str,
    ) -> ConfigUpdateResult:
        """Remove a camera from the config."""
        async with self._mutation_lock():
            config = await asyncio.to_thread(self.get_config)

            updated = [camera for camera in config.cameras if camera.name != camera_name]
            if len(updated) == len(config.cameras):
                raise CameraNotFoundError(f"Camera not found: {camera_name}")

            config.cameras = updated

            validated = await self._validate_config(config)
            await self._save_config(validated)
            return ConfigUpdateResult()

    async def replace_config(self, config: Config) -> ConfigUpdateResult:
        """Replace full config atomically after validation."""
        async with self._mutation_lock():
            payload = config.model_dump(mode="json")
            validated = await asyncio.to_thread(load_config_from_dict, payload)
            await self._save_config(validated)
            return ConfigUpdateResult()

    @asynccontextmanager
    async def config_snapshot(self, expected_config_version: str) -> AsyncIterator[Config]:
        """Hold a version-checked saved snapshot through a mutation or apply acceptance."""
        async with self._mutation_lock():
            config = await asyncio.to_thread(self.get_config)
            if config_signature(config) != expected_config_version:
                raise ConfigVersionConflictError(
                    "Configuration changed since it was loaded; refresh before saving"
                )
            yield config

    async def patch_config(self, patch: ConfigPatch) -> Config:
        """Validate and persist a partial edit without applying it to the runtime."""
        async with self.config_snapshot(patch.expected_config_version) as config:
            patch_payload = patch.model_dump(mode="json", exclude_unset=True)
            if self._contains_redacted_placeholder(patch_payload):
                raise ConfigPatchInvalidError(
                    "Patch contains redacted placeholders; omit unchanged secret fields"
                )

            payload = config.model_dump(mode="json")
            for section in ("storage", "filter", "vlm", "alert_policy"):
                section_patch = patch_payload.get(section)
                if not isinstance(section_patch, dict):
                    continue
                current = cast(dict[str, object], payload[section])
                if "backend" in section_patch and section_patch["backend"] != current["backend"]:
                    raise ConfigBackendChangeUnsupportedError(
                        f"Changing the {section} backend is not supported by settings edits"
                    )
                payload[section] = self._merge_config(current, section_patch)

            seen_indexes: set[int] = set()
            for notifier_patch in patch.notifiers or []:
                index = notifier_patch.index
                if index in seen_indexes or index >= len(config.notifiers):
                    raise ConfigPatchInvalidError(
                        "Notifier indexes must be unique and reference existing entries"
                    )
                seen_indexes.add(index)
                current_notifier = payload["notifiers"][index]
                entry_patch = notifier_patch.model_dump(mode="json", exclude_unset=True)
                entry_patch.pop("index")
                payload["notifiers"][index] = self._merge_config(current_notifier, entry_patch)

            try:
                validated = await asyncio.to_thread(load_config_from_dict, payload)
            except ConfigError as exc:
                raise ConfigPatchInvalidError(
                    "Configuration patch failed validation; check the submitted settings",
                    cause=exc,
                ) from exc
            try:
                await self._save_config(validated)
            except OSError as exc:
                raise ConfigSaveError(
                    "Unable to save configuration; check that its directory is writable",
                    cause=exc,
                ) from exc
            return validated

    async def _validate_config(self, config: Config) -> Config:
        """Validate configuration via the standard loader path."""
        payload = config.model_dump(mode="json")
        try:
            return await asyncio.to_thread(load_config_from_dict, payload)
        except ConfigError as exc:
            raise CameraConfigInvalidError(str(exc), cause=exc) from exc

    async def _save_config(self, config: Config) -> None:
        """Save config to disk with backup."""
        if self._mutations_frozen:
            raise ConfigApplyInProgressError(
                "HomeSec is restarting; retry saving after it restarts"
            )

        def _write() -> None:
            backup_path = Path(str(self._config_path) + ".bak")
            if self._config_path.exists():
                shutil.copy2(self._config_path, backup_path)
                self._enforce_sensitive_file_mode(backup_path)

            payload = config.model_dump(mode="json")
            tmp_path = self._config_path.with_suffix(self._config_path.suffix + ".tmp")
            tmp_path.parent.mkdir(parents=True, exist_ok=True)

            fd = os.open(
                tmp_path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                _SENSITIVE_CONFIG_FILE_MODE,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)
                handle.flush()
                os.fsync(handle.fileno())
            self._enforce_sensitive_file_mode(tmp_path)

            os.replace(tmp_path, self._config_path)
            self._enforce_sensitive_file_mode(self._config_path)

        await asyncio.to_thread(_write)
