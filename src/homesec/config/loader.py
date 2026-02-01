"""Configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from homesec.models.config import Config


class ConfigError(Exception):
    """Configuration loading or validation error."""

    def __init__(self, message: str, path: Path | None = None) -> None:
        super().__init__(message)
        self.path = path


def load_config(path: Path) -> Config:
    """Load and validate configuration from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        Validated Config instance

    Raises:
        ConfigError: If file not found, YAML invalid, or validation fails
    """
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}", path=path)

    try:
        with path.open() as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}", path=path) from e

    if raw is None:
        raise ConfigError(f"Config file is empty: {path}", path=path)

    if not isinstance(raw, dict):
        raise ConfigError(f"Config must be a YAML mapping, got {type(raw).__name__}", path=path)

    try:
        config = Config.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(format_validation_error(e, path), path=path) from e

    # Discover plugins and validate plugin-specific config
    from homesec.config.validation import validate_config
    from homesec.plugins import discover_all_plugins

    discover_all_plugins()
    validate_config(config)
    return config


def load_config_from_dict(data: dict[str, Any]) -> Config:
    """Load and validate configuration from a dict (useful for testing).

    Args:
        data: Configuration dictionary

    Returns:
        Validated Config instance

    Raises:
        ConfigError: If validation fails
    """
    try:
        config = Config.model_validate(data)
    except ValidationError as e:
        raise ConfigError(format_validation_error(e, path=None)) from e

    from homesec.config.validation import validate_config
    from homesec.plugins import discover_all_plugins

    discover_all_plugins()
    validate_config(config)
    return config


def resolve_env_var(env_var_name: str, required: bool = True) -> str | None:
    """Resolve environment variable by name.

    Args:
        env_var_name: Name of the environment variable
        required: If True, raise if not found

    Returns:
        Environment variable value, or None if not required and not found

    Raises:
        ConfigError: If required and not found
    """
    value = os.environ.get(env_var_name)
    if value is None and required:
        raise ConfigError(f"Required environment variable not set: {env_var_name}")
    return value


def format_validation_error(e: ValidationError, path: Path | None = None) -> str:
    """Format Pydantic validation error for human readability.

    Args:
        e: Pydantic ValidationError
        path: Optional config file path for context

    Returns:
        Human-readable error message
    """
    prefix = f"Config validation failed ({path}):" if path else "Config validation failed:"
    errors = []
    for err in e.errors():
        loc = " -> ".join(str(x) for x in err["loc"])
        msg = err["msg"]
        errors.append(f"  {loc}: {msg}")
    return prefix + "\n" + "\n".join(errors)
