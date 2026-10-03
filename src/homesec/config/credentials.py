"""Write-only credentials referenced by plugin-owned configuration fields."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Literal, get_args
from uuid import uuid4

from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator

from homesec.config.errors import CredentialStoreError
from homesec.models.config import Config
from homesec.plugins.registry import PluginType, validate_plugin

_MANAGED_PREFIX = "HOMESEC_SECRET_"
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600


class CredentialStatus(BaseModel):
    """Credential availability without disclosing a stored value."""

    configured: bool
    source: Literal["managed", "environment"]


class _CredentialsDocument(BaseModel):
    model_config = {"extra": "forbid", "hide_input_in_errors": True}

    version: Literal[1] = 1
    values: dict[str, SecretStr] = Field(default_factory=dict)

    @field_validator("values")
    @classmethod
    def validate_values(cls, values: dict[str, SecretStr]) -> dict[str, SecretStr]:
        if any(
            not credential_value_is_environment_compatible(value.get_secret_value())
            for value in values.values()
        ):
            raise ValueError("Credential values must be compatible with the process environment")
        return values


def is_managed_reference(reference: str | None) -> bool:
    """Whether an environment reference belongs to HomeSec's private namespace."""
    return (
        reference is not None
        and re.fullmatch(r"HOMESEC_SECRET_[0-9a-f]{32}", reference) is not None
    )


def credential_value_is_environment_compatible(value: str) -> bool:
    """Validate environment installation without reflecting a credential in an error."""
    if "\0" in value:
        return False
    try:
        os.fsencode(value)
    except UnicodeEncodeError:
        return False
    return True


def new_managed_reference() -> str:
    """Allocate a fresh reference so replacing or clearing a value changes YAML revision."""
    return f"{_MANAGED_PREFIX}{uuid4().hex}"


def _nested_model(annotation: object) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    models = [
        member
        for member in get_args(annotation)
        if isinstance(member, type) and issubclass(member, BaseModel)
    ]
    return models[0] if len(models) == 1 else None


def _credential_fields(
    model: type[BaseModel], value: BaseModel | None, prefix: str
) -> Iterator[tuple[str, str | None]]:
    for name, field in model.model_fields.items():
        path = f"{prefix}.{name}"
        current = getattr(value, name) if value is not None else field.default
        metadata = field.json_schema_extra
        if isinstance(metadata, dict) and metadata.get("homesec_credential") is True:
            yield path, current if isinstance(current, str) else None
            continue
        nested = (
            type(current) if isinstance(current, BaseModel) else _nested_model(field.annotation)
        )
        if nested is not None:
            yield from _credential_fields(
                nested, current if isinstance(current, BaseModel) else None, path
            )


def credential_references(config: Config) -> dict[str, str | None]:
    """Discover allowed credential slots through registered plugin schemas."""
    plugins = [
        ("storage.config", PluginType.STORAGE, config.storage.backend, config.storage.config),
        ("vlm.config", PluginType.ANALYZER, config.vlm.backend, config.vlm.config),
        *[
            (f"notifiers.{index}.config", PluginType.NOTIFIER, notifier.backend, notifier.config)
            for index, notifier in enumerate(config.notifiers)
        ],
    ]
    result: dict[str, str | None] = {}
    for prefix, plugin_type, backend, payload in plugins:
        validated = validate_plugin(plugin_type, backend, payload)
        result.update(_credential_fields(type(validated), validated, prefix))
    return result


def managed_credential_references(config: BaseModel) -> frozenset[str]:
    """Owned references used by a full config or validated plugin config."""
    references = (
        credential_references(config).values()
        if isinstance(config, Config)
        else dict(_credential_fields(type(config), config, "config")).values()
    )
    return frozenset(
        reference
        for reference in references
        if reference is not None and is_managed_reference(reference)
    )


def _require_environment_compatible_references(references: Iterable[str | None]) -> None:
    if any(
        reference is not None and not credential_value_is_environment_compatible(reference)
        for reference in references
    ):
        raise CredentialStoreError(
            "Credential environment references could not be read; check the configured references"
        )


def managed_credentials_path(config_path: Path) -> Path:
    """Keep private data under its own restrictive directory beside the YAML."""
    return config_path.parent / ".homesec" / "credentials.json"


def _require_private_owner(path: Path, *, directory: bool) -> None:
    info = path.lstat()
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_type(info.st_mode):
        raise OSError("Managed credential path is not a regular private file or directory")
    if os.name == "posix" and (info.st_uid != os.geteuid() or info.st_mode & 0o077):
        raise OSError("Managed credential path has unsafe ownership or permissions")


def _read_values(config_path: Path) -> dict[str, str]:
    path = managed_credentials_path(config_path)
    try:
        if not path.parent.exists() and not path.parent.is_symlink():
            return {}
        _require_private_owner(path.parent, directory=True)
        if not path.exists() and not path.is_symlink():
            return {}
        _require_private_owner(path, directory=False)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            document = _CredentialsDocument.model_validate(json.load(handle))
        if any(not is_managed_reference(key) for key in document.values):
            raise ValueError("Managed credential document contains an unowned reference")
        return {reference: value.get_secret_value() for reference, value in document.values.items()}
    except (OSError, ValueError, ValidationError) as exc:
        raise CredentialStoreError(
            "Managed credentials could not be read; check the private credential file and permissions",
            cause=exc,
        ) from exc


def load_managed_credentials(config_path: Path, config: Config) -> dict[str, str]:
    """Read only owned values referenced by this configuration for fresh startup."""
    references = managed_credential_references(config)
    if not references:
        return {}
    values = _read_values(config_path)
    return {reference: values[reference] for reference in references if reference in values}


def credential_status(config_path: Path, config: Config) -> dict[str, CredentialStatus]:
    """Report saved credential presence; managed values never come from host env."""
    references = credential_references(config)
    _require_environment_compatible_references(references.values())
    values = (
        _read_values(config_path) if any(map(is_managed_reference, references.values())) else {}
    )
    return {
        path: CredentialStatus(
            configured=bool(values.get(reference))
            if reference is not None and is_managed_reference(reference)
            else bool(os.environ.get(reference))
            if reference is not None
            else False,
            source="managed" if is_managed_reference(reference) else "environment",
        )
        for path, reference in references.items()
    }


def save_managed_credentials(config_path: Path, additions: Mapping[str, str]) -> None:
    """Atomically add fresh owned references before their YAML is committed.

    Keep previous references so failed saves, the YAML backup, and an older active
    configuration remain usable. Credentials are never overwritten in place.
    """
    if any(not credential_value_is_environment_compatible(value) for value in additions.values()):
        raise CredentialStoreError(
            "Credential values must be compatible with the process environment"
        )
    values = _read_values(config_path)
    if any(not is_managed_reference(key) or key in values for key in additions):
        raise CredentialStoreError("Managed credential references must be fresh and owned")
    values.update(additions)
    path = managed_credentials_path(config_path)
    temp_path: Path | None = None
    try:
        path.parent.mkdir(mode=_PRIVATE_DIRECTORY_MODE, exist_ok=True)
        _require_private_owner(path.parent, directory=True)
        descriptor, temp_name = tempfile.mkstemp(
            prefix="credentials-", suffix=".tmp", dir=path.parent
        )
        temp_path = Path(temp_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "values": values}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        if os.name == "posix":
            os.chmod(path, _PRIVATE_FILE_MODE)
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except OSError as exc:
        raise CredentialStoreError(
            "Managed credentials could not be saved; check the private credential directory is writable",
            cause=exc,
        ) from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
