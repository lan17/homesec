"""Event models for clip lifecycle tracking."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from homesec.models.enums import EventType, RiskLevelField
from homesec.models.filter import FilterResult


class ClipEvent(BaseModel):
    """Base class for all clip lifecycle events."""

    id: int | None = None
    clip_id: str
    timestamp: datetime
    event_type: str


class ClipRecordedEvent(ClipEvent):
    """Clip was recorded and queued for processing."""

    event_type: Literal[EventType.CLIP_RECORDED] = EventType.CLIP_RECORDED
    camera_name: str
    duration_s: float
    source_backend: str


class ClipDeletedEvent(ClipEvent):
    """Clip was deleted by a maintenance workflow (e.g., cleanup CLI)."""

    event_type: Literal[EventType.CLIP_DELETED] = EventType.CLIP_DELETED
    camera_name: str
    reason: str
    run_id: str
    local_path: str
    storage_uri: str | None
    deleted_local: bool
    deleted_storage: bool


class ClipRecheckedEvent(ClipEvent):
    """Clip was re-analyzed by a maintenance workflow."""

    event_type: Literal[EventType.CLIP_RECHECKED] = EventType.CLIP_RECHECKED
    camera_name: str
    reason: str
    run_id: str
    prior_filter: FilterResult | None
    recheck_filter: FilterResult


class UploadStartedEvent(ClipEvent):
    """Upload to storage backend started."""

    event_type: Literal[EventType.UPLOAD_STARTED] = EventType.UPLOAD_STARTED
    dest_key: str
    attempt: int


class UploadCompletedEvent(ClipEvent):
    """Upload to storage backend completed successfully."""

    event_type: Literal[EventType.UPLOAD_COMPLETED] = EventType.UPLOAD_COMPLETED
    storage_uri: str
    view_url: str | None
    attempt: int
    duration_ms: int


class UploadFailedEvent(ClipEvent):
    """Upload to storage backend failed."""

    event_type: Literal[EventType.UPLOAD_FAILED] = EventType.UPLOAD_FAILED
    attempt: int
    error_message: str
    error_type: str
    will_retry: bool


class FilterStartedEvent(ClipEvent):
    """Object detection filter started."""

    event_type: Literal[EventType.FILTER_STARTED] = EventType.FILTER_STARTED
    attempt: int


class FilterCompletedEvent(ClipEvent):
    """Object detection filter completed."""

    event_type: Literal[EventType.FILTER_COMPLETED] = EventType.FILTER_COMPLETED
    detected_classes: list[str]
    confidence: float
    model: str
    sampled_frames: int
    attempt: int
    duration_ms: int


class FilterFailedEvent(ClipEvent):
    """Object detection filter failed."""

    event_type: Literal[EventType.FILTER_FAILED] = EventType.FILTER_FAILED
    attempt: int
    error_message: str
    error_type: str
    will_retry: bool


class VLMStartedEvent(ClipEvent):
    """VLM analysis started."""

    event_type: Literal[EventType.VLM_STARTED] = EventType.VLM_STARTED
    attempt: int


class VLMCompletedEvent(ClipEvent):
    """VLM analysis completed."""

    event_type: Literal[EventType.VLM_COMPLETED] = EventType.VLM_COMPLETED
    risk_level: RiskLevelField
    activity_type: str
    summary: str
    analysis: dict[str, Any]
    prompt_tokens: int | None
    completion_tokens: int | None
    attempt: int
    duration_ms: int


class VLMFailedEvent(ClipEvent):
    """VLM analysis failed."""

    event_type: Literal[EventType.VLM_FAILED] = EventType.VLM_FAILED
    attempt: int
    error_message: str
    error_type: str
    will_retry: bool


class VLMSkippedEvent(ClipEvent):
    """VLM analysis skipped (no trigger classes detected)."""

    event_type: Literal[EventType.VLM_SKIPPED] = EventType.VLM_SKIPPED
    reason: str


class AlertDecisionMadeEvent(ClipEvent):
    """Alert policy decision made."""

    event_type: Literal[EventType.ALERT_DECISION_MADE] = EventType.ALERT_DECISION_MADE
    should_notify: bool
    reason: str
    detected_classes: list[str] | None
    vlm_risk: RiskLevelField | None


class NotificationSentEvent(ClipEvent):
    """Notification sent successfully."""

    event_type: Literal[EventType.NOTIFICATION_SENT] = EventType.NOTIFICATION_SENT
    notifier_name: str
    dedupe_key: str
    attempt: int = 1


class NotificationFailedEvent(ClipEvent):
    """Notification send failed."""

    event_type: Literal[EventType.NOTIFICATION_FAILED] = EventType.NOTIFICATION_FAILED
    notifier_name: str
    error_message: str
    error_type: str
    attempt: int = 1
    will_retry: bool = False


ClipLifecycleEvent = Annotated[
    ClipRecordedEvent
    | ClipDeletedEvent
    | ClipRecheckedEvent
    | UploadStartedEvent
    | UploadCompletedEvent
    | UploadFailedEvent
    | FilterStartedEvent
    | FilterCompletedEvent
    | FilterFailedEvent
    | VLMStartedEvent
    | VLMCompletedEvent
    | VLMFailedEvent
    | VLMSkippedEvent
    | AlertDecisionMadeEvent
    | NotificationSentEvent
    | NotificationFailedEvent,
    Field(discriminator="event_type"),
]
