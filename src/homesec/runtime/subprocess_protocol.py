"""IPC message schemas for runtime subprocess supervision."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from homesec.models.talk import (
    CameraTalkStatus,
    TalkCapabilityState,
    TalkInputFormat,
    TalkRefusalReason,
    TalkSessionPrepareResult,
    TalkState,
)
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


class WorkerTalkStatusPayload(BaseModel):
    """Serialized talk status emitted by worker."""

    enabled: bool
    policy_enabled: bool = False
    capability: TalkCapabilityState = TalkCapabilityState.UNKNOWN
    state: TalkState
    active_session_id: str | None = None
    supported_codecs: list[str] = Field(default_factory=list)
    offered_codecs: list[str] = Field(default_factory=list)
    selected_codec: str | None = None
    backend: str | None = None
    backend_reason: str | None = None
    last_error: str | None = None

    @model_validator(mode="after")
    def _derive_compatibility_defaults(self) -> WorkerTalkStatusPayload:
        """Default new fields when reading pre-capability worker payloads."""
        if "policy_enabled" not in self.model_fields_set:
            self.policy_enabled = self.enabled
        if "capability" not in self.model_fields_set:
            self.capability = _capability_from_talk_state(self.state)
        return self

    @classmethod
    def from_status(cls, status: CameraTalkStatus) -> WorkerTalkStatusPayload:
        """Build a worker payload from the shared API/runtime status model."""
        return cls(
            enabled=status.enabled,
            policy_enabled=status.policy_enabled,
            capability=status.capability,
            state=status.state,
            active_session_id=status.active_session_id,
            supported_codecs=list(status.supported_codecs),
            offered_codecs=list(status.offered_codecs),
            selected_codec=status.selected_codec,
            backend=status.backend,
            backend_reason=status.backend_reason,
            last_error=status.last_error,
        )


class WorkerTalkRefusalPayload(BaseModel):
    """Serialized talk refusal."""

    reason: TalkRefusalReason
    message: str


class WorkerTalkPreparePayload(BaseModel):
    """Serialized talk prepare acknowledgement."""

    session_id: str
    input: TalkInputFormat

    @classmethod
    def from_result(cls, result: TalkSessionPrepareResult) -> WorkerTalkPreparePayload:
        if result.session_id is None:
            raise ValueError("accepted talk prepare result requires session_id")
        return cls(session_id=result.session_id, input=result.input)


class WorkerTalkStopPayload(BaseModel):
    """Serialized talk stop acknowledgement."""

    accepted: bool
    state: TalkState


class WorkerCommandType(StrEnum):
    """Parent-to-worker control command type."""

    PREVIEW_STATUS = "preview_status"
    PREVIEW_ENSURE_ACTIVE = "preview_ensure_active"
    PREVIEW_FORCE_STOP = "preview_force_stop"
    PREVIEW_NOTE_VIEWER_ACTIVITY = "preview_note_viewer_activity"
    TALK_STATUS = "talk_status"
    TALK_PREPARE_SESSION = "talk_prepare_session"
    TALK_STREAM_OPEN = "talk_stream_open"
    TALK_STOP_SESSION = "talk_stop_session"


def _capability_from_talk_state(state: TalkState) -> TalkCapabilityState:
    match state:
        case TalkState.DISABLED:
            return TalkCapabilityState.DISABLED
        case TalkState.UNSUPPORTED:
            return TalkCapabilityState.UNSUPPORTED
        case TalkState.IDLE | TalkState.STARTING | TalkState.ACTIVE | TalkState.STOPPING:
            return TalkCapabilityState.SUPPORTED
        case TalkState.ERROR:
            return TalkCapabilityState.ERROR
        case TalkState.TEMPORARILY_UNAVAILABLE:
            return TalkCapabilityState.UNKNOWN


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
    viewer_id: str | None = None
    session_id: str | None = None
    talk_input: TalkInputFormat | None = None


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
    talk_status: WorkerTalkStatusPayload | None = None
    talk_refusal: WorkerTalkRefusalPayload | None = None
    talk_prepare: WorkerTalkPreparePayload | None = None
    talk_stop_result: WorkerTalkStopPayload | None = None
    error_code: WorkerCommandErrorCode | None = None
    error_message: str | None = None
