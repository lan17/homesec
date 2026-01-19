"""Configuration models with per-camera override support."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

from homesec.models.enums import RiskLevel, RiskLevelField
from homesec.models.filter import FilterConfig
from homesec.models.source import FtpSourceConfig, LocalFolderSourceConfig, RTSPSourceConfig
from homesec.models.vlm import VLMConfig


class AlertPolicyOverrides(BaseModel):
    """Per-camera alert policy overrides (only non-None fields override base)."""

    min_risk_level: RiskLevelField | None = None
    notify_on_activity_types: list[str] | None = None
    notify_on_motion: bool | None = None


class DefaultAlertPolicySettings(BaseModel):
    """Default alert policy settings."""

    min_risk_level: RiskLevelField = RiskLevel.MEDIUM
    notify_on_activity_types: list[str] = Field(default_factory=list)
    notify_on_motion: bool = False
    overrides: dict[str, AlertPolicyOverrides] = Field(default_factory=dict, exclude=True)
    trigger_classes: list[str] = Field(default_factory=list, exclude=True)


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

    @model_validator(mode="after")
    def _validate_alert_policy(self) -> AlertPolicyConfig:
        # Validate and reassign config for built-in backends
        if self.backend == "default" and isinstance(self.config, dict):
            validated = DefaultAlertPolicySettings.model_validate(self.config)
            object.__setattr__(self, "config", validated)
        return self


class DropboxStorageConfig(BaseModel):
    """Dropbox storage configuration."""

    root: str
    token_env: str = "DROPBOX_TOKEN"
    app_key_env: str = "DROPBOX_APP_KEY"
    app_secret_env: str = "DROPBOX_APP_SECRET"
    refresh_token_env: str = "DROPBOX_REFRESH_TOKEN"
    web_url_prefix: str = "https://www.dropbox.com/home"


class LocalStorageConfig(BaseModel):
    """Local storage configuration."""

    root: str = "./storage"


class StoragePathsConfig(BaseModel):
    """Logical storage paths for different artifact types."""

    clips_dir: str = "clips"
    backups_dir: str = "backups"
    artifacts_dir: str = "artifacts"


class StorageConfig(BaseModel):
    """Storage backend configuration.

    Note: Backend names are validated against the registry at runtime via
    validate_plugin_names(). This allows third-party storage plugins via entry points.
    """

    model_config = {"extra": "allow"}  # Allow third-party backend configs

    backend: str = "dropbox"
    dropbox: DropboxStorageConfig | None = None
    local: LocalStorageConfig | None = None
    paths: StoragePathsConfig = Field(default_factory=StoragePathsConfig)

    @field_validator("backend", mode="before")
    @classmethod
    def _normalize_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value

    @model_validator(mode="after")
    def _validate_builtin_backends(self) -> StorageConfig:
        """Validate that built-in backends have their required config.

        Third-party backends are validated later in load_storage_plugin().
        """
        match self.backend:
            case "dropbox":
                if self.dropbox is None:
                    raise ValueError(
                        "storage.dropbox is required when backend=dropbox. "
                        "Add 'storage.dropbox' section to your config."
                    )
            case "local":
                if self.local is None:
                    raise ValueError(
                        "storage.local is required when backend=local. "
                        "Add 'storage.local' section to your config."
                    )
            case _:
                # Third-party backend - validation happens in load_storage_plugin()
                pass
        return self


class StateStoreConfig(BaseModel):
    """State store configuration."""

    dsn_env: str | None = None
    dsn: str | None = None

    @model_validator(mode="after")
    def _validate_backend(self) -> StateStoreConfig:
        if not (self.dsn_env or self.dsn):
            raise ValueError("state_store.dsn_env or state_store.dsn required for postgres")
        return self


class MQTTAuthConfig(BaseModel):
    """MQTT auth configuration using env var names."""

    username_env: str | None = None
    password_env: str | None = None


class MQTTConfig(BaseModel):
    """MQTT notifier configuration."""

    host: str
    port: int = 1883
    auth: MQTTAuthConfig | None = None
    topic_template: str = "homecam/alerts/{camera_name}"
    qos: int = 1
    retain: bool = False
    connection_timeout: float = 10.0


class SendGridEmailConfig(BaseModel):
    """SendGrid email notifier configuration."""

    api_key_env: str = "SENDGRID_API_KEY"
    from_email: str
    from_name: str | None = None
    to_emails: list[str] = Field(min_length=1)
    cc_emails: list[str] = Field(default_factory=list)
    bcc_emails: list[str] = Field(default_factory=list)
    subject_template: str = "[HomeSec] {camera_name}: {activity_type} ({risk_level})"
    text_template: str = (
        "HomeSec alert\n"
        "Camera: {camera_name}\n"
        "Clip: {clip_id}\n"
        "Risk: {risk_level}\n"
        "Activity: {activity_type}\n"
        "Reason: {notify_reason}\n"
        "Summary: {summary}\n"
        "View: {view_url}\n"
        "Storage: {storage_uri}\n"
        "Time: {ts}\n"
        "Upload failed: {upload_failed}\n"
    )
    html_template: str = (
        "<html><body>"
        "<h2>HomeSec alert</h2>"
        "<p><strong>Camera:</strong> {camera_name}</p>"
        "<p><strong>Clip:</strong> {clip_id}</p>"
        "<p><strong>Risk:</strong> {risk_level}</p>"
        "<p><strong>Activity:</strong> {activity_type}</p>"
        "<p><strong>Reason:</strong> {notify_reason}</p>"
        "<p><strong>Summary:</strong> {summary}</p>"
        '<p><strong>View:</strong> <a href="{view_url}">{view_url}</a></p>'
        "<p><strong>Storage:</strong> {storage_uri}</p>"
        "<p><strong>Time:</strong> {ts}</p>"
        "<p><strong>Upload failed:</strong> {upload_failed}</p>"
        "<h3>Structured analysis</h3>"
        "{analysis_html}"
        "</body></html>"
    )
    request_timeout_s: float = 10.0
    api_base: str = "https://api.sendgrid.com/v3"

    @model_validator(mode="after")
    def _validate_templates(self) -> SendGridEmailConfig:
        if not self.text_template and not self.html_template:
            raise ValueError("sendgrid_email requires at least one of text_template/html_template")
        return self


class NotifierConfig(BaseModel):
    """Notifier configuration entry."""

    backend: str
    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)

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


class HealthConfig(BaseModel):
    """Health endpoint configuration."""

    host: str = "0.0.0.0"
    port: int = 8080
    mqtt_is_critical: bool = False


class CameraSourceConfig(BaseModel):
    """Camera source configuration wrapper.

    The `type` field is a string to allow external plugins to register new source types.
    Built-in types ("rtsp", "local_folder", "ftp") have their configs validated here.
    External plugin configs are validated by the plugin at load time.
    """

    model_config = {"extra": "forbid"}

    type: str
    config: RTSPSourceConfig | LocalFolderSourceConfig | FtpSourceConfig | dict[str, Any]

    @model_validator(mode="before")
    @classmethod
    def _parse_source_config(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        source_type = data.get("type")
        raw_config = data.get("config")
        if isinstance(raw_config, (RTSPSourceConfig, LocalFolderSourceConfig, FtpSourceConfig)):
            return data

        config_data = raw_config or {}
        updated = dict(data)
        if isinstance(source_type, str):
            source_type = source_type.lower()
            updated["type"] = source_type
        match source_type:
            case "rtsp":
                updated["config"] = RTSPSourceConfig.model_validate(config_data)
            case "local_folder":
                updated["config"] = LocalFolderSourceConfig.model_validate(config_data)
            case "ftp":
                updated["config"] = FtpSourceConfig.model_validate(config_data)
            case _:
                # External plugin source type - keep config as dict
                # Plugin will validate at load time
                updated["config"] = config_data if isinstance(config_data, dict) else {}
        return updated

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value

    @model_validator(mode="after")
    def _validate_source_config(self) -> CameraSourceConfig:
        match self.type:
            case "rtsp":
                if not isinstance(self.config, RTSPSourceConfig):
                    raise ValueError("camera.source.config must be RTSPSourceConfig for type=rtsp")
            case "local_folder":
                if not isinstance(self.config, LocalFolderSourceConfig):
                    raise ValueError(
                        "camera.source.config must be LocalFolderSourceConfig for type=local_folder"
                    )
            case "ftp":
                if not isinstance(self.config, FtpSourceConfig):
                    raise ValueError("camera.source.config must be FtpSourceConfig for type=ftp")
            case _:
                # External plugin source type - validation happens at plugin load time
                pass
        return self


class CameraConfig(BaseModel):
    """Camera configuration and clip source selection."""

    name: str
    source: CameraSourceConfig


TConfig = TypeVar("TConfig", bound=BaseModel)


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
    filter: FilterConfig
    vlm: VLMConfig
    alert_policy: AlertPolicyConfig

    # Per-camera overrides (alert policy only)
    per_camera_alert: dict[str, AlertPolicyOverrides] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_notifiers(self) -> Config:
        if not self.notifiers:
            raise ValueError("notifiers must include at least one notifier")
        return self

    @model_validator(mode="after")
    def _validate_builtin_plugin_configs(self) -> Config:
        """Validate built-in plugin configs for early error detection.

        Third-party plugin configs are validated later during plugin loading.
        """
        # Validate built-in filter configs
        if self.filter.plugin == "yolo":
            from homesec.models.filter import YoloFilterSettings

            if isinstance(self.filter.config, dict):
                validated_filter_config = YoloFilterSettings.model_validate(self.filter.config)
                # Replace dict with validated object for built-in plugins
                object.__setattr__(self.filter, "config", validated_filter_config)

        # Validate built-in VLM configs
        if self.vlm.backend == "openai":
            from homesec.models.vlm import OpenAILLMConfig

            if isinstance(self.vlm.llm, dict):
                validated_llm_config = OpenAILLMConfig.model_validate(self.vlm.llm)
                # Replace dict with validated object for built-in plugins
                object.__setattr__(self.vlm, "llm", validated_llm_config)

        return self

    def _merge_overrides(
        self, base: TConfig, overrides: BaseModel, model_type: type[TConfig]
    ) -> TConfig:
        merged = {
            **base.model_dump(),
            **overrides.model_dump(exclude_none=True),
        }
        return model_type.model_validate(merged)

    def get_default_alert_policy(self, camera_name: str) -> DefaultAlertPolicySettings:
        """Get merged default alert policy settings for a specific camera."""
        if self.alert_policy.backend != "default":
            raise ValueError(
                f"default alert policy requested but backend is {self.alert_policy.backend}"
            )
        base = DefaultAlertPolicySettings.model_validate(self.alert_policy.config)
        if camera_name not in self.per_camera_alert:
            return base

        overrides = self.per_camera_alert[camera_name]
        return self._merge_overrides(base, overrides, DefaultAlertPolicySettings)
