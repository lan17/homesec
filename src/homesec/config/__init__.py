"""Configuration loading and validation."""

from homesec.config.loader import (
    ConfigError,
    load_config,
    load_config_from_dict,
    load_config_or_bootstrap,
    resolve_env_var,
)
from homesec.config.validation import (
    validate_camera_references,
    validate_config,
    validate_plugin_configs,
    validate_plugin_names,
)

__all__ = [
    "ConfigError",
    "load_config",
    "load_config_from_dict",
    "load_config_or_bootstrap",
    "resolve_env_var",
    "validate_camera_references",
    "validate_config",
    "validate_plugin_configs",
    "validate_plugin_names",
]
