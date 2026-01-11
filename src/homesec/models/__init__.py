"""HomeSec data models."""

from homesec.models.alert import Alert, AlertDecision
from homesec.models.clip import Clip, ClipStateData, _resolve_forward_refs
from homesec.models.config import (
    AlertPolicyConfig,
    AlertPolicyOverrides,
    DefaultAlertPolicySettings,
    CameraConfig,
    CameraSourceConfig,
    ConcurrencyConfig,
    Config,
    DropboxStorageConfig,
    HealthConfig,
    LocalStorageConfig,
    MQTTAuthConfig,
    MQTTConfig,
    NotifierConfig,
    RetentionConfig,
    RetryConfig,
    SendGridEmailConfig,
    StateStoreConfig,
    StorageConfig,
    StoragePathsConfig,
)
from homesec.models.filter import FilterConfig, FilterOverrides, FilterResult, YoloFilterSettings
from homesec.models.source import FtpSourceConfig, LocalFolderSourceConfig, RTSPSourceConfig
from homesec.models.vlm import (
    AnalysisResult,
    EntityTimeline,
    OpenAILLMConfig,
    RiskLevel,
    SequenceAnalysis,
    VLMConfig,
    VLMPreprocessConfig,
)

# Resolve forward references in ClipStateData
_resolve_forward_refs()

__all__ = [
    "Alert",
    "AlertDecision",
    "AlertPolicyConfig",
    "AlertPolicyOverrides",
    "DefaultAlertPolicySettings",
    "AnalysisResult",
    "CameraConfig",
    "CameraSourceConfig",
    "Clip",
    "ClipStateData",
    "ConcurrencyConfig",
    "Config",
    "DropboxStorageConfig",
    "EntityTimeline",
    "FilterConfig",
    "FilterOverrides",
    "FtpSourceConfig",
    "FilterResult",
    "HealthConfig",
    "LocalStorageConfig",
    "LocalFolderSourceConfig",
    "MQTTAuthConfig",
    "MQTTConfig",
    "NotifierConfig",
    "OpenAILLMConfig",
    "RTSPSourceConfig",
    "SendGridEmailConfig",
    "RetentionConfig",
    "RetryConfig",
    "RiskLevel",
    "SequenceAnalysis",
    "StateStoreConfig",
    "StorageConfig",
    "StoragePathsConfig",
    "VLMConfig",
    "VLMPreprocessConfig",
    "YoloFilterSettings",
]
