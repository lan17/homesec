"""Configuration loading and validation."""

from __future__ import annotations

import logging
import os
import stat
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from homesec.models.config import Config

logger = logging.getLogger(__name__)
_SENSITIVE_MODE_MASK = 0o077


class ConfigErrorCode(str, Enum):
    """Stable config error codes for runtime and API mapping."""

    FILE_NOT_FOUND = "CONFIG_FILE_NOT_FOUND"
    YAML_INVALID = "CONFIG_YAML_INVALID"
    EMPTY_FILE = "CONFIG_EMPTY_FILE"
    ROOT_NOT_MAPPING = "CONFIG_ROOT_NOT_MAPPING"
    VALIDATION_FAILED = "CONFIG_VALIDATION_FAILED"
    CAMERA_REFERENCES_INVALID = "CONFIG_CAMERA_REFERENCES_INVALID"
    PLUGIN_NAMES_INVALID = "CONFIG_PLUGIN_NAMES_INVALID"
    PLUGIN_CONFIG_INVALID = "CONFIG_PLUGIN_CONFIG_INVALID"
    ENV_VAR_MISSING = "CONFIG_ENV_VAR_MISSING"
    UNKNOWN = "CONFIG_UNKNOWN"


class ConfigError(Exception):
    """Configuration loading or validation error."""

    def __init__(
        self,
        message: str,
        *,
        code: ConfigErrorCode = ConfigErrorCode.UNKNOWN,
        path: Path | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.path = path
        self.__cause__ = cause


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
        raise ConfigError(
            f"Config file not found: {path}",
            code=ConfigErrorCode.FILE_NOT_FOUND,
            path=path,
        )
    _warn_if_permissive_config_mode(path)

    try:
        with path.open() as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(
            f"Invalid YAML in {path}: {e}",
            code=ConfigErrorCode.YAML_INVALID,
            path=path,
            cause=e,
        ) from e

    if raw is None:
        raise ConfigError(
            f"Config file is empty: {path}",
            code=ConfigErrorCode.EMPTY_FILE,
            path=path,
        )

    if not isinstance(raw, dict):
        raise ConfigError(
            f"Config must be a YAML mapping, got {type(raw).__name__}",
            code=ConfigErrorCode.ROOT_NOT_MAPPING,
            path=path,
        )

    try:
        config = Config.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(
            format_validation_error(e, path),
            code=ConfigErrorCode.VALIDATION_FAILED,
            path=path,
            cause=e,
        ) from e

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
        raise ConfigError(
            format_validation_error(e, path=None),
            code=ConfigErrorCode.VALIDATION_FAILED,
            cause=e,
        ) from e

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
        raise ConfigError(
            f"Required environment variable not set: {env_var_name}",
            code=ConfigErrorCode.ENV_VAR_MISSING,
        )
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


def _warn_if_permissive_config_mode(path: Path) -> None:
    """Warn when config file mode exposes secrets to group/other users."""
    if os.name != "posix":
        return
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return
    if mode & _SENSITIVE_MODE_MASK:
        logger.warning(
            "Config file permissions are too permissive for secret-bearing config: path=%s mode=%04o expected=0600",
            path,
            mode,
        )
