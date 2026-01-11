"""Event models for clip lifecycle tracking."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from homesec.models.filter import FilterResult


class ClipEvent(BaseModel):
    """Base class for all clip lifecycle events."""
    id: int | None = None
    clip_id: str
    timestamp: datetime
    event_type: str


class ClipRecordedEvent(ClipEvent):
    """Clip was recorded and queued for processing."""

    event_type: Literal["clip_recorded"] = "clip_recorded"
    camera_name: str
    duration_s: float
    source_type: str


class ClipDeletedEvent(ClipEvent):
    """Clip was deleted by a maintenance workflow (e.g., cleanup CLI)."""

    event_type: Literal["clip_deleted"] = "clip_deleted"
    camera_name: str
    reason: str
    run_id: str
    local_path: str
    storage_uri: str | None
    deleted_local: bool
    deleted_storage: bool


class ClipRecheckedEvent(ClipEvent):
    """Clip was re-analyzed by a maintenance workflow."""

    event_type: Literal["clip_rechecked"] = "clip_rechecked"
    camera_name: str
    reason: str
    run_id: str
    prior_filter: FilterResult | None
    recheck_filter: FilterResult


class UploadStartedEvent(ClipEvent):
    """Upload to storage backend started."""
    event_type: Literal["upload_started"] = "upload_started"
    dest_key: str
    attempt: int


class UploadCompletedEvent(ClipEvent):
    """Upload to storage backend completed successfully."""
    event_type: Literal["upload_completed"] = "upload_completed"
    storage_uri: str
    view_url: str | None
    attempt: int
    duration_ms: int


class UploadFailedEvent(ClipEvent):
    """Upload to storage backend failed."""
    event_type: Literal["upload_failed"] = "upload_failed"
    attempt: int
    error_message: str
    error_type: str
    will_retry: bool


class FilterStartedEvent(ClipEvent):
    """Object detection filter started."""
    event_type: Literal["filter_started"] = "filter_started"
    attempt: int


class FilterCompletedEvent(ClipEvent):
    """Object detection filter completed."""
    event_type: Literal["filter_completed"] = "filter_completed"
    detected_classes: list[str]
    confidence: float
    model: str
    sampled_frames: int
    attempt: int
    duration_ms: int


class FilterFailedEvent(ClipEvent):
    """Object detection filter failed."""
    event_type: Literal["filter_failed"] = "filter_failed"
    attempt: int
    error_message: str
    error_type: str
    will_retry: bool


class VLMStartedEvent(ClipEvent):
    """VLM analysis started."""
    event_type: Literal["vlm_started"] = "vlm_started"
    attempt: int


class VLMCompletedEvent(ClipEvent):
    """VLM analysis completed."""
    event_type: Literal["vlm_completed"] = "vlm_completed"
    risk_level: str
    activity_type: str
    summary: str
    analysis: dict[str, Any]
    prompt_tokens: int | None
    completion_tokens: int | None
    attempt: int
    duration_ms: int


class VLMFailedEvent(ClipEvent):
    """VLM analysis failed."""
    event_type: Literal["vlm_failed"] = "vlm_failed"
    attempt: int
    error_message: str
    error_type: str
    will_retry: bool


class VLMSkippedEvent(ClipEvent):
    """VLM analysis skipped (no trigger classes detected)."""
    event_type: Literal["vlm_skipped"] = "vlm_skipped"
    reason: str


class AlertDecisionMadeEvent(ClipEvent):
    """Alert policy decision made."""
    event_type: Literal["alert_decision_made"] = "alert_decision_made"
    should_notify: bool
    reason: str
    detected_classes: list[str] | None
    vlm_risk: str | None


class NotificationSentEvent(ClipEvent):
    """Notification sent successfully."""
    event_type: Literal["notification_sent"] = "notification_sent"
    notifier_name: str
    dedupe_key: str
    attempt: int = 1


class NotificationFailedEvent(ClipEvent):
    """Notification send failed."""
    event_type: Literal["notification_failed"] = "notification_failed"
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
