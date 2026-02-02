"""Centralized enums for type safety and IDE support."""

from enum import IntEnum, StrEnum
from typing import Annotated, Any

from pydantic import BeforeValidator, PlainSerializer


def _validate_risk_level(value: Any) -> "RiskLevel":
    """Validate and convert input to RiskLevel.

    Accepts:
        - RiskLevel enum member
        - Integer (0-3)
        - String ("low", "medium", "high", "critical", case-insensitive)
    """
    if isinstance(value, RiskLevel):
        return value
    if isinstance(value, int):
        return RiskLevel(value)
    if isinstance(value, str):
        return RiskLevel.from_string(value)
    raise ValueError(f"Cannot convert {type(value).__name__} to RiskLevel")


def _serialize_risk_level(value: "RiskLevel") -> str:
    """Serialize RiskLevel to lowercase string for config compatibility."""
    return str(value)


class EventType(StrEnum):
    """All clip lifecycle event types.

    Used in event models and the event store for type-safe event handling.
    """

    CLIP_RECORDED = "clip_recorded"
    CLIP_DELETED = "clip_deleted"
    CLIP_RECHECKED = "clip_rechecked"
    UPLOAD_STARTED = "upload_started"
    UPLOAD_COMPLETED = "upload_completed"
    UPLOAD_FAILED = "upload_failed"
    FILTER_STARTED = "filter_started"
    FILTER_COMPLETED = "filter_completed"
    FILTER_FAILED = "filter_failed"
    VLM_STARTED = "vlm_started"
    VLM_COMPLETED = "vlm_completed"
    VLM_FAILED = "vlm_failed"
    VLM_SKIPPED = "vlm_skipped"
    ALERT_DECISION_MADE = "alert_decision_made"
    NOTIFICATION_SENT = "notification_sent"
    NOTIFICATION_FAILED = "notification_failed"


class ClipStatus(StrEnum):
    """Clip processing status values.

    Represents the high-level status of a clip in the processing pipeline.
    """

    QUEUED_LOCAL = "queued_local"
    UPLOADED = "uploaded"
    ANALYZED = "analyzed"
    DONE = "done"
    ERROR = "error"
    DELETED = "deleted"


class VLMRunMode(StrEnum):
    """Policy for when to run VLM analysis."""

    TRIGGER_ONLY = "trigger_only"
    ALWAYS = "always"
    NEVER = "never"


class RiskLevel(IntEnum):
    """VLM risk assessment levels.

    Uses IntEnum for natural ordering and comparison:
        RiskLevel.HIGH > RiskLevel.LOW  # True
        RiskLevel.MEDIUM >= RiskLevel.MEDIUM  # True

    String serialization is handled by Pydantic for config compatibility.
    """

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

    def __str__(self) -> str:
        """Return lowercase name for human-readable output."""
        return self.name.lower()

    @classmethod
    def from_string(cls, value: str) -> "RiskLevel":
        """Parse risk level from string (case-insensitive).

        Args:
            value: Risk level string (e.g., "low", "MEDIUM", "High")

        Returns:
            Corresponding RiskLevel enum member

        Raises:
            ValueError: If value is not a valid risk level
        """
        try:
            return cls[value.upper()]
        except KeyError:
            valid = ", ".join(member.name.lower() for member in cls)
            raise ValueError(f"Invalid risk level '{value}'. Valid values: {valid}") from None


# Pydantic-compatible type that accepts strings, ints, and RiskLevel
# Use this in Pydantic models instead of raw RiskLevel
RiskLevelField = Annotated[
    RiskLevel,
    BeforeValidator(_validate_risk_level),
    PlainSerializer(_serialize_risk_level, return_type=str),
]
