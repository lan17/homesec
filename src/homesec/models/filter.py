"""Object detection filter data and config models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FilterResult(BaseModel):
    """Result from object detection filter on a video clip."""

    detected_classes: list[str]
    confidence: float
    model: str
    sampled_frames: int


class YoloFilterSettings(BaseModel):
    """YOLO filter settings.

    model_path accepts a filename; bare names resolve under ./yolo_cache.
    """

    model_config = {"extra": "forbid"}

    model_path: str = "yolo11n.pt"
    classes: list[str] = Field(default_factory=lambda: ["person"], min_length=1)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    sample_fps: int = Field(default=2, ge=1)
    min_box_h_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    min_hits: int = Field(default=1, ge=1)


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
    - During YAML parsing: dict[str, object] (preserves all third-party fields)
    - After plugin discovery: BaseModel subclass (validated against plugin.config_model)

    Note: Plugin names are validated against the registry at runtime via
    validate_plugin_names(). This allows third-party plugins via entry points.
    """

    model_config = {"extra": "forbid"}

    plugin: str
    max_workers: int = Field(default=4, ge=1)
    config: dict[str, object] | BaseModel  # Dict before validation, BaseModel after
