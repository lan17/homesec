"""Configuration models with per-camera override support."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from homesec.models.filter import FilterConfig
from homesec.models.vlm import VLMConfig


class AlertPolicyConfig(BaseModel):
    """Alert policy plugin configuration."""

    backend: str = "default"
    enabled: bool = True
    config: dict[str, Any] | BaseModel = Field(default_factory=dict)

    @field_validator("backend", mode="before")
    @classmethod
    def _normalize_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value


class StoragePathsConfig(BaseModel):
    """Logical storage paths for different artifact types."""

    clips_dir: str = "clips"
    backups_dir: str = "backups"
    artifacts_dir: str = "artifacts"


class StorageConfig(BaseModel):
    """Storage backend configuration."""

    model_config = {"extra": "forbid"}

    backend: str = "dropbox"
    config: dict[str, Any] | BaseModel = Field(default_factory=dict)
    paths: StoragePathsConfig = Field(default_factory=StoragePathsConfig)

    @field_validator("backend", mode="before")
    @classmethod
    def _normalize_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value


class StateStoreConfig(BaseModel):
    """State store configuration."""

    dsn_env: str | None = None
    dsn: str | None = None

    @model_validator(mode="after")
    def _validate_backend(self) -> StateStoreConfig:
        if not (self.dsn_env or self.dsn):
            raise ValueError("state_store.dsn_env or state_store.dsn required for postgres")
        return self


class NotifierConfig(BaseModel):
    """Notifier configuration entry."""

    backend: str
    enabled: bool = True
    config: dict[str, Any] | BaseModel = Field(default_factory=dict)

    @field_validator("backend", mode="before")
    @classmethod
    def _normalize_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value


class RetentionConfig(BaseModel):
    """Retention configuration for local storage."""

    max_local_size: str | None = None


class ConcurrencyConfig(BaseModel):
    """Concurrency limits for pipeline stages."""

    max_clips_in_flight: int = Field(default=4, ge=1)
    upload_workers: int = Field(default=4, ge=1)
    filter_workers: int = Field(default=4, ge=1)
    vlm_workers: int = Field(default=2, ge=1)


class RetryConfig(BaseModel):
    """Retry configuration for transient failures."""

    max_attempts: int = Field(default=3, ge=1)
    backoff_s: float = Field(default=1.0, ge=0.0)


class FastAPIServerConfig(BaseModel):
    """Configuration for the FastAPI server."""

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    auth_enabled: bool = False
    api_key_env: str | None = None

    def get_api_key(self) -> str | None:
        """Resolve API key from environment variable."""
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)

    @model_validator(mode="after")
    def _validate_auth(self) -> FastAPIServerConfig:
        if self.auth_enabled and not self.api_key_env:
            raise ValueError("server.api_key_env is required when server.auth_enabled is true")
        return self


class HealthConfig(BaseModel):
    """Health endpoint configuration."""

    host: str = "0.0.0.0"
    port: int = 8080
    mqtt_is_critical: bool = False


class CameraSourceConfig(BaseModel):
    """Camera source configuration wrapper."""

    model_config = {"extra": "forbid"}

    backend: str
    config: dict[str, Any] | BaseModel = Field(default_factory=dict)

    @field_validator("backend", mode="before")
    @classmethod
    def _normalize_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value


class CameraConfig(BaseModel):
    """Camera configuration and clip source selection."""

    name: str
    enabled: bool = True
    source: CameraSourceConfig


class Config(BaseModel):
    """Main configuration with per-camera override support."""

    version: int = 1
    cameras: list[CameraConfig]
    storage: StorageConfig
    state_store: StateStoreConfig = Field(default_factory=StateStoreConfig)
    notifiers: list[NotifierConfig]
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    server: FastAPIServerConfig = Field(default_factory=FastAPIServerConfig)
    filter: FilterConfig
    vlm: VLMConfig
    alert_policy: AlertPolicyConfig

    @model_validator(mode="after")
    def _validate_notifiers(self) -> Config:
        if not self.notifiers:
            raise ValueError("notifiers must include at least one notifier")
        return self
