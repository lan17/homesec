"""Runtime control-plane models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from homesec.notifiers.multiplex import NotifierEntry
from homesec.pipeline import ClipPipeline

if TYPE_CHECKING:
    from homesec.interfaces import AlertPolicy, ClipSource, Notifier, ObjectFilter, VLMAnalyzer
    from homesec.models.config import Config


class ManagedRuntime(Protocol):
    """Runtime handle contract consumed by RuntimeManager."""

    generation: int
    config: Config
    config_signature: str


class RuntimeState(StrEnum):
    """Lifecycle state for the active runtime."""

    IDLE = "idle"
    RELOADING = "reloading"
    FAILED = "failed"


class PreviewState(StrEnum):
    """Camera-scoped preview lifecycle state."""

    IDLE = "idle"
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    ERROR = "error"


class PreviewRefusalReason(StrEnum):
    """Machine-readable preview start refusal reason."""

    RECORDING_PRIORITY = "recording_priority"
    SESSION_BUDGET_EXHAUSTED = "session_budget_exhausted"
    PREVIEW_TEMPORARILY_UNAVAILABLE = "preview_temporarily_unavailable"


@dataclass(slots=True)
class RuntimeBundle:
    """Restartable runtime unit managed by RuntimeManager."""

    generation: int
    config: Config
    config_signature: str
    notifier: Notifier
    notifier_entries: list[NotifierEntry]
    filter_plugin: ObjectFilter
    vlm_plugin: VLMAnalyzer
    alert_policy: AlertPolicy
    pipeline: ClipPipeline
    sources: list[ClipSource]
    sources_by_camera: dict[str, ClipSource]


@dataclass(slots=True)
class RuntimeCameraStatus:
    """Health status snapshot for a runtime camera source."""

    healthy: bool
    last_heartbeat: float | None


@dataclass(frozen=True, slots=True)
class CameraPreviewStatus:
    """Runtime-owned preview status for a single camera."""

    camera_name: str
    enabled: bool
    state: PreviewState
    viewer_count: int | None = None
    degraded_reason: str | None = None
    last_error: str | None = None
    idle_shutdown_at: float | None = None


@dataclass(frozen=True, slots=True)
class CameraPreviewStartRefusal:
    """Machine-readable refusal when preview cannot be attached or started."""

    camera_name: str
    reason: PreviewRefusalReason
    message: str


@dataclass(frozen=True, slots=True)
class CameraPreviewStopResult:
    """Runtime acknowledgement for a force-stop preview request."""

    camera_name: str
    accepted: bool
    state: PreviewState


@dataclass(slots=True)
class RuntimeStatusSnapshot:
    """Snapshot of runtime-manager status for API consumption."""

    state: RuntimeState
    generation: int
    reload_in_progress: bool
    active_config_version: str | None
    last_reload_at: datetime | None
    last_reload_error: str | None


@dataclass(frozen=True, slots=True)
class RuntimeReloadRequest:
    """Result of requesting a runtime reload."""

    accepted: bool
    message: str
    target_generation: int


@dataclass(frozen=True, slots=True)
class RuntimeReloadResult:
    """Final result of an attempted runtime reload."""

    success: bool
    generation: int
    error: str | None = None


def config_signature(config: Config) -> str:
    """Return a short, stable signature for a config payload."""
    payload = config.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def preview_error_status(
    camera_name: str,
    *,
    enabled: bool,
    message: str,
) -> CameraPreviewStatus:
    """Return an error preview status with bounded user-facing detail."""
    return CameraPreviewStatus(
        camera_name=camera_name,
        enabled=enabled,
        state=PreviewState.ERROR,
        last_error=message,
    )
