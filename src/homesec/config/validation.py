"""Custom configuration validation helpers."""

from __future__ import annotations

from homesec.config.loader import ConfigError
from homesec.models.config import Config


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

    for camera in config.per_camera_alert:
        if camera not in camera_set:
            errors.append(f"per_camera_alert references unknown camera: {camera}")

    if errors:
        raise ConfigError("Invalid camera references:\n  " + "\n  ".join(errors))


def validate_plugin_names(
    config: Config,
    valid_filters: list[str],
    valid_vlms: list[str],
    valid_storage: list[str] | None = None,
    valid_notifiers: list[str] | None = None,
    valid_alert_policies: list[str] | None = None,
) -> None:
    """Validate that plugin names are recognized.

    Args:
        config: Config instance to validate
        valid_filters: List of valid filter plugin names
        valid_vlms: List of valid VLM plugin names
        valid_storage: Optional list of valid storage backends
        valid_notifiers: Optional list of valid notifier backends

    Raises:
        ConfigError: If plugin names are not recognized
    """
    errors = []

    if config.filter.plugin not in valid_filters:
        errors.append(f"Unknown filter plugin: {config.filter.plugin} (valid: {valid_filters})")

    if config.vlm.backend not in valid_vlms:
        errors.append(f"Unknown VLM plugin: {config.vlm.backend} (valid: {valid_vlms})")

    if valid_storage is not None and config.storage.backend not in valid_storage:
        errors.append(
            f"Unknown storage backend: {config.storage.backend} (valid: {valid_storage})"
        )

    if valid_notifiers is not None:
        for notifier in config.notifiers:
            if notifier.backend not in valid_notifiers:
                errors.append(
                    "Unknown notifier backend: "
                    f"{notifier.backend} (valid: {valid_notifiers})"
                )

    if valid_alert_policies is not None:
        if config.alert_policy.backend not in valid_alert_policies:
            errors.append(
                "Unknown alert policy backend: "
                f"{config.alert_policy.backend} (valid: {valid_alert_policies})"
            )

    if errors:
        raise ConfigError("Invalid plugin configuration:\n  " + "\n  ".join(errors))
