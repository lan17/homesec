"""Generic talk/session models shared across runtime and API boundaries."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from homesec.talk.backend_ids import sanitize_talk_backend_id, sanitize_talk_backend_reason


class TalkState(StrEnum):
    """Current talk capability/session state for a camera."""

    DISABLED = "disabled"
    UNSUPPORTED = "unsupported"
    IDLE = "idle"
    STARTING = "starting"
    ACTIVE = "active"
    STOPPING = "stopping"
    ERROR = "error"
    TEMPORARILY_UNAVAILABLE = "temporarily_unavailable"


class TalkCapabilityState(StrEnum):
    """Discovered camera talk capability independent from session state."""

    DISABLED = "disabled"
    UNKNOWN = "unknown"
    PROBING = "probing"
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNSUPPORTED_CODEC = "unsupported_codec"
    ERROR = "error"


class TalkRefusalReason(StrEnum):
    """Canonical reasons a talk session request may be refused."""

    CAMERA_NOT_FOUND = "camera_not_found"
    TALK_DISABLED = "talk_disabled"
    SOURCE_NOT_TALK_CAPABLE = "source_not_talk_capable"
    SESSION_ALREADY_ACTIVE = "session_already_active"
    SESSION_BUDGET_EXHAUSTED = "session_budget_exhausted"
    UNSUPPORTED_CAMERA = "unsupported_camera"
    UNSUPPORTED_CODEC = "unsupported_codec"
    CAMERA_BACKCHANNEL_FAILED = "camera_backchannel_failed"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    INVALID_AUDIO_FRAME = "invalid_audio_frame"
    BACKPRESSURE = "backpressure"


class TalkInputFormat(BaseModel):
    """Browser input frame format contract for push-to-talk."""

    model_config = {"extra": "forbid"}

    codec: Literal["pcm_s16le"] = "pcm_s16le"
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    channels: int = Field(default=1, ge=1, le=1)
    frame_ms: int = Field(default=20, ge=10, le=60)

    @model_validator(mode="after")
    def _require_integral_frame_samples(self) -> TalkInputFormat:
        if (self.sample_rate * self.frame_ms) % 1000 != 0:
            raise ValueError("sample_rate and frame_ms must produce an integral PCM frame size")
        return self

    @property
    def expected_bytes_per_frame(self) -> int:
        """Expected byte size for one PCM S16LE frame."""
        return (self.sample_rate * self.frame_ms // 1000) * self.channels * 2


class TalkCapabilityProbeResult(BaseModel):
    """Result of probing a camera for push-to-talk backchannel support."""

    model_config = {"extra": "forbid"}

    capability: TalkCapabilityState
    offered_codecs: list[str] = Field(default_factory=list)
    selected_codec: str | None = None
    refusal_reason: TalkRefusalReason | None = None
    message: str | None = None


class CameraTalkStatus(BaseModel):
    """Current talk status returned by runtime/application APIs."""

    model_config = {"extra": "forbid"}

    camera_name: str
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

    @model_validator(mode="before")
    @classmethod
    def _sanitize_backend_diagnostics(cls, value: object) -> object:
        return sanitize_talk_backend_diagnostic_fields(value)

    @model_validator(mode="after")
    def _derive_compatibility_defaults(self) -> CameraTalkStatus:
        """Default new capability fields from the legacy status contract."""
        if "policy_enabled" not in self.model_fields_set:
            self.policy_enabled = self.enabled
        if "capability" not in self.model_fields_set:
            self.capability = _capability_from_state(self.state)
        return self


def sanitize_talk_backend_diagnostic_fields(value: object) -> object:
    """Normalize or drop unsafe backend IDs before public diagnostics."""
    if not isinstance(value, dict):
        return value

    raw_backend = value.get("backend")
    if raw_backend is None and "backend_reason" not in value:
        return value

    sanitized_backend = sanitize_talk_backend_id(raw_backend)
    sanitized_value = dict(value)
    sanitized_value["backend"] = sanitized_backend
    if sanitized_backend is None:
        sanitized_value["backend_reason"] = None
    elif "backend_reason" in sanitized_value:
        sanitized_value["backend_reason"] = sanitize_talk_backend_reason(
            sanitized_value["backend_reason"]
        )
    return sanitized_value


def _capability_from_state(state: TalkState) -> TalkCapabilityState:
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


class TalkSessionPrepareRequest(BaseModel):
    """Runtime/source request to reserve a talk session slot."""

    model_config = {"extra": "forbid"}

    session_id: str | None = None
    input: TalkInputFormat = Field(default_factory=TalkInputFormat)


class TalkSessionPrepareResult(BaseModel):
    """Result of reserving a talk session slot."""

    model_config = {"extra": "forbid"}

    accepted: bool
    session_id: str | None = None
    refusal_reason: TalkRefusalReason | None = None
    message: str | None = None
    input: TalkInputFormat = Field(default_factory=TalkInputFormat)


class TalkSessionOpenRequest(BaseModel):
    """Runtime/source request to open an already reserved talk session."""

    model_config = {"extra": "forbid"}

    session_id: str
    input: TalkInputFormat = Field(default_factory=TalkInputFormat)
