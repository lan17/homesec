"""VLM analysis data and config models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from homesec.models.enums import RiskLevelField, VLMRunMode

__all__ = ["AnalysisResult", "VLMConfig", "VLMPreprocessConfig"]


class AnalysisResult(BaseModel):
    """Structured result from VLM analysis of a video clip."""

    risk_level: RiskLevelField
    activity_type: str
    summary: str
    analysis: SequenceAnalysis | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class EntityTimeline(BaseModel):
    """Timeline of an entity across multiple frames."""

    model_config = {"extra": "forbid"}
    type: Literal["person", "vehicle", "animal", "package", "object", "unknown"]
    first_seen_timestamp: str
    last_seen_timestamp: str
    description: str
    movement: str
    location: str
    interaction: str


class SequenceAnalysis(BaseModel):
    """Structured analysis of a sequence of security camera frames."""

    model_config = {"extra": "forbid"}
    sequence_description: str
    max_risk_level: RiskLevelField
    primary_activity: Literal[
        "normal_delivery",
        "normal_visitor",
        "passerby",
        "suspicious_behavior",
        "dangerous_activity",
        "no_activity",
        "unknown",
    ]
    observations: list[str]
    entities_timeline: list[EntityTimeline]
    requires_review: bool
    frame_count: int
    video_start_time: str
    video_end_time: str


class VLMPreprocessConfig(BaseModel):
    """Preprocessing configuration for VLM frame extraction."""

    model_config = {"extra": "forbid"}
    max_frames: int = 10
    max_size: int = 1024
    quality: int = 85


class VLMConfig(BaseModel):
    """Base VLM configuration.

    Backend-specific config is stored in the 'config' field.
    - During YAML parsing: dict[str, Any] (preserves all third-party fields)
    - After plugin discovery: BaseModel subclass (validated against plugin.config_model)

    Note: Backend names are validated against the registry at runtime via
    validate_plugin_names(). This allows third-party VLM plugins via entry points.
    """

    model_config = {"extra": "forbid"}
    backend: str
    trigger_classes: list[str] = Field(default_factory=lambda: ["person"])
    run_mode: VLMRunMode = VLMRunMode.TRIGGER_ONLY
    config: dict[str, Any] | BaseModel  # Dict before validation, BaseModel after
    preprocessing: VLMPreprocessConfig = Field(default_factory=VLMPreprocessConfig)

    @field_validator("backend", mode="before")
    @classmethod
    def _normalize_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value

    @field_validator("run_mode", mode="before")
    @classmethod
    def _normalize_run_mode(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value
