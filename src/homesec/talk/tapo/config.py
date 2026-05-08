"""Configuration helpers for the Tapo local talk backend."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator

from homesec.talk.backends import TalkBackendConfigError, TalkBackendContext

_SAFE_ENV_VAR_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_HEX_PATTERN = re.compile(r"^[A-F0-9]+$")


class TapoLocalTalkConfig(BaseModel):
    """Config for a local TP-Link Tapo camera talk endpoint."""

    model_config = {"extra": "forbid"}

    host: str | None = None
    port: int = Field(default=8800, ge=1, le=65535)

    username: str = "admin"
    username_env: str | None = None

    password_sha256_env: str | None = None
    password_md5_env: str | None = None

    connect_timeout_s: float | None = Field(default=None, ge=0.1, le=30.0)
    io_timeout_s: float | None = Field(default=None, ge=0.1, le=30.0)

    mode: Literal["aec"] = "aec"
    audio_codec: Literal["PCMA/8000"] = "PCMA/8000"

    @field_validator("host", "username", mode="before")
    @classmethod
    def _strip_optional_string(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator(
        "username_env",
        "password_sha256_env",
        "password_md5_env",
    )
    @classmethod
    def _validate_env_var_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not _SAFE_ENV_VAR_NAME_PATTERN.fullmatch(stripped):
            raise ValueError("Tapo local env var names must be safe shell identifiers")
        return stripped

    @model_validator(mode="after")
    def _validate_non_empty_values(self) -> TapoLocalTalkConfig:
        if self.host == "":
            raise ValueError("Tapo local host cannot be empty")
        if self.username == "":
            raise ValueError("Tapo local username cannot be empty")
        return self


@dataclass(frozen=True, slots=True)
class TapoCredential:
    """Resolved credential material for Tapo local Digest authentication."""

    username: str
    password_hash: str = field(repr=False)
    hash_kind: Literal["sha256", "md5"]


def resolve_tapo_host(config: TapoLocalTalkConfig, context: TalkBackendContext) -> str:
    """Resolve the Tapo LAN host from backend config or source URI context."""
    if config.host:
        return config.host

    if context.source_host:
        return context.source_host

    for uri in (context.resolved_source_uri, context.source_uri):
        host = _host_from_uri(uri)
        if host:
            return host

    raise TalkBackendConfigError("Tapo local backend requires host or source URI host")


def resolve_tapo_credential(
    config: TapoLocalTalkConfig,
    context: TalkBackendContext,
    *,
    preferred_hash_kind: Literal["sha256", "md5"] = "sha256",
) -> TapoCredential:
    """Resolve Tapo credential material from env vars or source URI credentials."""
    username = _resolve_username(config, context)
    env_order = _credential_env_order(config, preferred_hash_kind=preferred_hash_kind)
    if env_order:
        missing_env_names: list[str] = []
        for hash_kind, env_name in env_order:
            raw_value = context.env_value(env_name)
            if raw_value is None or raw_value.strip() == "":
                missing_env_names.append(env_name)
                continue
            normalized = raw_value.strip().upper()
            _validate_hash_shape(normalized, env_name=env_name, hash_kind=hash_kind)
            return TapoCredential(
                username=username,
                password_hash=normalized,
                hash_kind=hash_kind,
            )

        if missing_env_names:
            raise TalkBackendConfigError(
                f"Required Tapo local environment variable is not set: {missing_env_names[0]}"
            )

    source_password = _source_uri_password(context)
    if source_password:
        return TapoCredential(
            username=username,
            password_hash=_hash_source_password(
                source_password,
                hash_kind=preferred_hash_kind,
            ),
            hash_kind=preferred_hash_kind,
        )

    raise TalkBackendConfigError(
        "Tapo local backend requires password hash env var or RTSP URL password"
    )


def _resolve_username(config: TapoLocalTalkConfig, context: TalkBackendContext) -> str:
    if config.username_env is None:
        return config.username
    username = context.env_value(config.username_env)
    if username is None or username.strip() == "":
        raise TalkBackendConfigError(
            f"Required Tapo local environment variable is not set: {config.username_env}"
        )
    return username.strip()


def _credential_env_order(
    config: TapoLocalTalkConfig,
    *,
    preferred_hash_kind: Literal["sha256", "md5"],
) -> list[tuple[Literal["sha256", "md5"], str]]:
    configured: dict[Literal["sha256", "md5"], str] = {}
    if config.password_sha256_env is not None:
        configured["sha256"] = config.password_sha256_env
    if config.password_md5_env is not None:
        configured["md5"] = config.password_md5_env

    ordered: list[tuple[Literal["sha256", "md5"], str]] = []
    if preferred_hash_kind in configured:
        ordered.append((preferred_hash_kind, configured[preferred_hash_kind]))
    for hash_kind in ("sha256", "md5"):
        if hash_kind in configured and hash_kind != preferred_hash_kind:
            ordered.append((hash_kind, configured[hash_kind]))
    return ordered


def _validate_hash_shape(
    value: str,
    *,
    env_name: str,
    hash_kind: Literal["sha256", "md5"],
) -> None:
    expected_length = 64 if hash_kind == "sha256" else 32
    if len(value) != expected_length or _HEX_PATTERN.fullmatch(value) is None:
        raise TalkBackendConfigError(f"Tapo local credential hash in {env_name} is invalid")


def _host_from_uri(uri: str | None) -> str | None:
    if not uri:
        return None
    parsed = urlsplit(uri)
    return parsed.hostname


def _source_uri_password(context: TalkBackendContext) -> str | None:
    for uri in (context.resolved_source_uri, context.source_uri):
        if not uri:
            continue
        password = urlsplit(uri).password
        if password:
            return unquote(password)
    return None


def _hash_source_password(
    password: str,
    *,
    hash_kind: Literal["sha256", "md5"],
) -> str:
    if hash_kind == "sha256":
        return hashlib.sha256(password.encode("utf-8")).hexdigest().upper()
    return hashlib.md5(password.encode("utf-8"), usedforsecurity=False).hexdigest().upper()
