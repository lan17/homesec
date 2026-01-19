"""Alert decision and notification payload models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from homesec.models.enums import RiskLevelField
from homesec.models.vlm import SequenceAnalysis


class AlertDecision(BaseModel):
    """Decision whether to send an alert for a clip."""

    notify: bool
    notify_reason: str  # e.g., "risk_level=high" or "activity_type=delivery (per-camera)"


class Alert(BaseModel):
    """MQTT notification payload."""

    clip_id: str
    camera_name: str
    storage_uri: str | None
    view_url: str | None
    risk_level: RiskLevelField | None  # None if VLM skipped
    activity_type: str | None
    notify_reason: str
    summary: str | None
    analysis: SequenceAnalysis | None = None
    ts: datetime
    dedupe_key: str  # Same as clip_id for MVP
    upload_failed: bool  # True if storage_uri is None due to upload failure
    vlm_failed: bool = False  # True if VLM analysis failed but alert sent anyway
