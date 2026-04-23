"""IPC message schemas for runtime subprocess supervision."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field

from homesec.runtime.models import PreviewRefusalReason, PreviewState


class WorkerEventType(StrEnum):
    """Worker-to-parent event type."""

    STARTED = "started"
    HEARTBEAT = "heartbeat"
    STOPPED = "stopped"
    ERROR = "error"


class WorkerCameraStatusPayload(BaseModel):
    """Serialized camera status emitted by worker."""

    healthy: bool
    last_heartbeat: float | None = None


class WorkerPreviewStatusPayload(BaseModel):
    """Serialized preview status emitted by worker."""

    enabled: bool
    state: PreviewState
    viewer_count: int | None = None
    degraded_reason: str | None = None
    last_error: str | None = None
    idle_shutdown_at: float | None = None


class WorkerEvent(BaseModel):
    """Structured event emitted by runtime worker process."""

    event: WorkerEventType
    generation: int
    correlation_id: str
    pid: int
    sent_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cameras: dict[str, WorkerCameraStatusPayload] = Field(default_factory=dict)
    previews: dict[str, WorkerPreviewStatusPayload] = Field(default_factory=dict)
    message: str | None = None


class WorkerCommandType(StrEnum):
    """Parent-to-worker control command type."""

    PREVIEW_STATUS = "preview_status"
    PREVIEW_ENSURE_ACTIVE = "preview_ensure_active"
    PREVIEW_FORCE_STOP = "preview_force_stop"


class WorkerCommandErrorCode(StrEnum):
    """Machine-readable worker command failure code."""

    CAMERA_NOT_FOUND = "camera_not_found"


class WorkerPreviewRefusalPayload(BaseModel):
    """Serialized preview start refusal."""

    reason: PreviewRefusalReason
    message: str


class WorkerPreviewStopPayload(BaseModel):
    """Serialized preview stop acknowledgement."""

    accepted: bool
    state: PreviewState


class WorkerCommand(BaseModel):
    """Structured preview control command sent to the runtime worker."""

    command: WorkerCommandType
    command_id: str
    generation: int
    correlation_id: str
    camera_name: str


class WorkerCommandResult(BaseModel):
    """Structured response to a preview control command."""

    command: WorkerCommandType
    command_id: str
    generation: int
    correlation_id: str
    camera_name: str
    status: WorkerPreviewStatusPayload | None = None
    refusal: WorkerPreviewRefusalPayload | None = None
    stop_result: WorkerPreviewStopPayload | None = None
    error_code: WorkerCommandErrorCode | None = None
    error_message: str | None = None
