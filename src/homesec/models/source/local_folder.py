"""Local folder source configuration model."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LocalFolderSourceConfig(BaseModel):
    """Local folder source configuration."""

    model_config = {"extra": "forbid"}

    camera_name: str | None = Field(
        default=None,
        description="Optional human-friendly camera name.",
    )
    watch_dir: str = Field(
        default="recordings",
        description="Directory to watch for new clips.",
    )
    poll_interval: float = Field(
        default=1.0,
        ge=0.0,
        description="Polling interval in seconds.",
    )
    stability_threshold_s: float = Field(
        default=3.0,
        ge=0.0,
        description="Seconds to wait for file size to stabilize before accepting a clip.",
    )
