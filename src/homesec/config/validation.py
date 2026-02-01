"""Custom configuration validation helpers."""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from homesec.config.loader import ConfigError
from homesec.models.config import Config
from homesec.plugins.registry import PluginType, validate_plugin


def validate_camera_references(config: Config, camera_names: list[str] | None = None) -> None:
    """Validate that per-camera config keys reference valid camera names.

    Args:
        config: Config instance to validate
        camera_names: Optional list of valid camera names from cameras config

    Raises:
        ConfigError: If per-camera keys reference unknown cameras
    """
    if camera_names is None:
        camera_names = [camera.name for camera in config.cameras]

    camera_set = set(camera_names)
    errors = []

    overrides: dict[str, object] = {}
    if config.alert_policy.backend == "default":
        raw = config.alert_policy.config
        if isinstance(raw, BaseModel):
            overrides = getattr(raw, "overrides", {}) or {}
        elif isinstance(raw, dict):
            overrides = raw.get("overrides", {}) or {}

    for camera in overrides:
        if camera not in camera_set:
            errors.append(f"alert_policy.overrides references unknown camera: {camera}")

    if errors:
        raise ConfigError("Invalid camera references:\n  " + "\n  ".join(errors))


def validate_plugin_names(
    config: Config,
    valid_filters: list[str],
    valid_vlms: list[str],
    valid_storage: list[str] | None = None,
    valid_notifiers: list[str] | None = None,
    valid_alert_policies: list[str] | None = None,
    valid_sources: list[str] | None = None,
) -> None:
    """Validate that plugin names are recognized.

    Args:
        config: Config instance to validate
        valid_filters: List of valid filter plugin names
        valid_vlms: List of valid VLM plugin names
        valid_storage: Optional list of valid storage backends
        valid_notifiers: Optional list of valid notifier backends
        valid_alert_policies: Optional list of valid alert policy backends
        valid_sources: Optional list of valid source types

    Raises:
        ConfigError: If plugin names are not recognized
    """
    errors = []

    valid_filters_lower = {name.lower() for name in valid_filters}
    if config.filter.backend.lower() not in valid_filters_lower:
        errors.append(
            f"Unknown filter plugin: {config.filter.backend} (valid: {sorted(valid_filters_lower)})"
        )

    valid_vlms_lower = {name.lower() for name in valid_vlms}
    if config.vlm.backend.lower() not in valid_vlms_lower:
        errors.append(
            f"Unknown VLM plugin: {config.vlm.backend} (valid: {sorted(valid_vlms_lower)})"
        )

    if valid_storage is not None:
        valid_storage_lower = {name.lower() for name in valid_storage}
        if config.storage.backend.lower() not in valid_storage_lower:
            errors.append(
                f"Unknown storage backend: {config.storage.backend} "
                f"(valid: {sorted(valid_storage_lower)})"
            )

    if valid_notifiers is not None:
        valid_notifiers_lower = {name.lower() for name in valid_notifiers}
        for notifier in config.notifiers:
            if notifier.backend.lower() not in valid_notifiers_lower:
                errors.append(
                    f"Unknown notifier backend: {notifier.backend} "
                    f"(valid: {sorted(valid_notifiers_lower)})"
                )

    if valid_alert_policies is not None:
        valid_alert_policies_lower = {name.lower() for name in valid_alert_policies}
        if config.alert_policy.backend.lower() not in valid_alert_policies_lower:
            errors.append(
                "Unknown alert policy backend: "
                f"{config.alert_policy.backend} (valid: {sorted(valid_alert_policies_lower)})"
            )

    if valid_sources is not None:
        valid_sources_lower = {name.lower() for name in valid_sources}
        for camera in config.cameras:
            if camera.source.backend.lower() not in valid_sources_lower:
                errors.append(
                    f"Unknown source type for camera '{camera.name}': "
                    f"{camera.source.backend} (valid: {sorted(valid_sources_lower)})"
                )

    if errors:
        raise ConfigError("Invalid plugin configuration:\n  " + "\n  ".join(errors))


def validate_plugin_configs(config: Config) -> None:
    """Validate plugin configs against registered plugin config models."""
    errors: list[str] = []

    def _add_error(prefix: str, err: Exception) -> None:
        if isinstance(err, ValidationError):
            errors.append(f"{prefix}: {err}")
        else:
            errors.append(f"{prefix}: {err}")

    try:
        validate_plugin(
            PluginType.STORAGE,
            config.storage.backend,
            config.storage.config,
        )
    except Exception as exc:
        _add_error(f"storage[{config.storage.backend}]", exc)

    try:
        validate_plugin(
            PluginType.FILTER,
            config.filter.backend,
            config.filter.config,
        )
    except Exception as exc:
        _add_error(f"filter[{config.filter.backend}]", exc)

    try:
        validate_plugin(
            PluginType.ANALYZER,
            config.vlm.backend,
            config.vlm.config,
        )
    except Exception as exc:
        _add_error(f"vlm[{config.vlm.backend}]", exc)

    if config.alert_policy.enabled:
        try:
            runtime_context = {}
            if config.alert_policy.backend == "default":
                runtime_context["trigger_classes"] = list(config.vlm.trigger_classes)
            validate_plugin(
                PluginType.ALERT_POLICY,
                config.alert_policy.backend,
                config.alert_policy.config,
                **runtime_context,
            )
        except Exception as exc:
            _add_error(f"alert_policy[{config.alert_policy.backend}]", exc)

    for index, notifier in enumerate(config.notifiers):
        try:
            validate_plugin(
                PluginType.NOTIFIER,
                notifier.backend,
                notifier.config,
            )
        except Exception as exc:
            _add_error(f"notifier[{index}:{notifier.backend}]", exc)

    for camera in config.cameras:
        try:
            validate_plugin(
                PluginType.SOURCE,
                camera.source.backend,
                camera.source.config,
                camera_name=camera.name,
            )
        except Exception as exc:
            _add_error(f"source[{camera.name}:{camera.source.backend}]", exc)

    if errors:
        raise ConfigError("Invalid plugin config:\n  " + "\n  ".join(errors))
