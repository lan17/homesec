"""HomeSec data models."""

from homesec.models.alert import Alert, AlertDecision
from homesec.models.clip import Clip, ClipStateData, _resolve_forward_refs
from homesec.models.config import (
    AlertPolicyConfig,
    AlertPolicyOverrides,
    CameraConfig,
    CameraSourceConfig,
    ConcurrencyConfig,
    Config,
    DefaultAlertPolicySettings,
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
from homesec.models.enums import RiskLevel, RiskLevelField
from homesec.models.filter import FilterConfig, FilterOverrides, FilterResult, YoloFilterSettings
from homesec.models.source import FtpSourceConfig, LocalFolderSourceConfig, RTSPSourceConfig
from homesec.models.vlm import (
    AnalysisResult,
    EntityTimeline,
    OpenAILLMConfig,
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
    "AnalysisResult",
    "CameraConfig",
    "CameraSourceConfig",
    "Clip",
    "ClipStateData",
    "ConcurrencyConfig",
    "Config",
    "DefaultAlertPolicySettings",
    "DropboxStorageConfig",
    "EntityTimeline",
    "FilterConfig",
    "FilterOverrides",
    "FilterResult",
    "FtpSourceConfig",
    "HealthConfig",
    "LocalFolderSourceConfig",
    "LocalStorageConfig",
    "MQTTAuthConfig",
    "MQTTConfig",
    "NotifierConfig",
    "OpenAILLMConfig",
    "RTSPSourceConfig",
    "RetentionConfig",
    "RetryConfig",
    "RiskLevel",
    "RiskLevelField",
    "SendGridEmailConfig",
    "SequenceAnalysis",
    "StateStoreConfig",
    "StorageConfig",
    "StoragePathsConfig",
    "VLMConfig",
    "VLMPreprocessConfig",
    "YoloFilterSettings",
]
