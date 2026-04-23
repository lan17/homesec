from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class LivePublisherState(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    ERROR = "error"


class LivePublisherRefusalReason(StrEnum):
    RECORDING_PRIORITY = "recording_priority"
    SESSION_BUDGET_EXHAUSTED = "session_budget_exhausted"
    PREVIEW_TEMPORARILY_UNAVAILABLE = "preview_temporarily_unavailable"


@dataclass(frozen=True, slots=True)
class LivePublisherStatus:
    state: LivePublisherState
    viewer_count: int | None = None
    degraded_reason: str | None = None
    last_error: str | None = None
    idle_shutdown_at: float | None = None


@dataclass(frozen=True, slots=True)
class LivePublisherStartRefusal:
    reason: LivePublisherRefusalReason
    message: str


class LivePublisher(Protocol):
    def status(self) -> LivePublisherStatus: ...

    def ensure_active(self) -> LivePublisherStatus | LivePublisherStartRefusal: ...

    def request_stop(self) -> None: ...

    def note_viewer_activity(self, viewer_id: str | None = None) -> None: ...

    def sync_recording_active(self, recording_active: bool) -> None: ...

    def shutdown(self) -> None: ...


class NoopLivePublisher(LivePublisher):
    """Default preview seam used until a real publisher backend is wired in."""

    def __init__(self) -> None:
        self._status = LivePublisherStatus(state=LivePublisherState.IDLE)

    def status(self) -> LivePublisherStatus:
        return self._status

    def ensure_active(self) -> LivePublisherStatus | LivePublisherStartRefusal:
        return LivePublisherStartRefusal(
            reason=LivePublisherRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE,
            message="Preview publisher is not configured for this RTSP source",
        )

    def request_stop(self) -> None:
        self._status = LivePublisherStatus(state=LivePublisherState.IDLE)

    def note_viewer_activity(self, viewer_id: str | None = None) -> None:
        _ = viewer_id

    def sync_recording_active(self, recording_active: bool) -> None:
        _ = recording_active

    def shutdown(self) -> None:
        self._status = LivePublisherStatus(state=LivePublisherState.IDLE)
