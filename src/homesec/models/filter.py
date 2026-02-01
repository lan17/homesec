"""Object detection filter data and config models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class FilterResult(BaseModel):
    """Result from object detection filter on a video clip."""

    detected_classes: list[str]
    confidence: float
    model: str
    sampled_frames: int


class FilterOverrides(BaseModel):
    """Runtime overrides for filter settings (model path not allowed)."""

    model_config = {"extra": "forbid"}

    classes: list[str] | None = Field(default=None, min_length=1)
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    sample_fps: int | None = Field(default=None, ge=1)
    min_box_h_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    min_hits: int | None = Field(default=None, ge=1)


class FilterConfig(BaseModel):
    """Base filter configuration (plugin-agnostic).

    Plugin-specific config is stored in the 'config' field.
    - During YAML parsing: dict[str, Any] (preserves all third-party fields)
    - After plugin discovery: BaseModel subclass (validated against plugin.config_model)

    Note: Plugin names are validated against the registry at runtime via
    validate_plugin_names(). This allows third-party plugins via entry points.
    """

    model_config = {"extra": "forbid"}

    backend: str
    config: dict[str, Any] | BaseModel  # Dict before validation, BaseModel after

    @field_validator("backend", mode="before")
    @classmethod
    def _normalize_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value
