"""Setup/onboarding API models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from homesec.models.config import (
    AlertPolicyConfig,
    CameraConfig,
    FastAPIServerConfig,
    NotifierConfig,
    StateStoreConfig,
    StorageConfig,
)
from homesec.models.filter import FilterConfig
from homesec.models.vlm import VLMConfig

SetupState = Literal["fresh", "partial", "complete"]


class SetupStatusResponse(BaseModel):
    """Current setup completion state used by onboarding flows."""

    state: SetupState
    has_cameras: bool
    pipeline_running: bool
    auth_configured: bool


class PreflightCheckResponse(BaseModel):
    """Single preflight check outcome."""

    name: str
    passed: bool
    message: str
    latency_ms: float | None = None


class PreflightResponse(BaseModel):
    """Aggregated preflight response."""

    checks: list[PreflightCheckResponse]
    all_passed: bool


class FinalizeRequest(BaseModel):
    """Finalize setup by writing the assembled config and requesting restart."""

    cameras: list[CameraConfig] | None = None
    storage: StorageConfig | None = None
    state_store: StateStoreConfig | None = None
    notifiers: list[NotifierConfig] | None = None
    filter: FilterConfig | None = None
    vlm: VLMConfig | None = None
    alert_policy: AlertPolicyConfig | None = None
    server: FastAPIServerConfig | None = None


class FinalizeResponse(BaseModel):
    """Finalize setup response for onboarding UI."""

    success: bool
    config_path: str
    restart_requested: bool
    defaults_applied: list[str]
    errors: list[str]


TestConnectionTarget = Literal["camera", "storage", "notifier", "analyzer"]


class TestConnectionRequest(BaseModel):
    """Payload for generic setup test-connection endpoint."""

    type: TestConnectionTarget
    backend: str
    config: dict[str, Any]


class TestConnectionResponse(BaseModel):
    """Connection test outcome for setup/UI flows."""

    success: bool
    message: str
    latency_ms: float | None = None
    details: dict[str, Any] | None = None
