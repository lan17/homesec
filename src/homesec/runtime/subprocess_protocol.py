"""IPC message schemas for runtime subprocess supervision."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


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


class WorkerEvent(BaseModel):
    """Structured event emitted by runtime worker process."""

    event: WorkerEventType
    generation: int
    correlation_id: str
    pid: int
    sent_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cameras: dict[str, WorkerCameraStatusPayload] = Field(default_factory=dict)
    message: str | None = None
