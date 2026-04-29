"""Generic talk/session models shared across runtime and API boundaries."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


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

    codec: str = "pcm_s16le"
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    channels: int = Field(default=1, ge=1, le=1)
    frame_ms: int = Field(default=20, ge=10, le=60)

    @property
    def expected_bytes_per_frame(self) -> int:
        """Expected byte size for one PCM S16LE frame."""
        return int(self.sample_rate * self.frame_ms / 1000) * self.channels * 2


class CameraTalkStatus(BaseModel):
    """Current talk status returned by runtime/application APIs."""

    model_config = {"extra": "forbid"}

    camera_name: str
    enabled: bool
    state: TalkState
    active_session_id: str | None = None
    supported_codecs: list[str] = Field(default_factory=list)
    selected_codec: str | None = None
    last_error: str | None = None
