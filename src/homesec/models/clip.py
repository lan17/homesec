"""Core clip-centric data models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from homesec.models.enums import ClipStatus

if TYPE_CHECKING:
    from homesec.models.alert import AlertDecision
    from homesec.models.filter import FilterResult
    from homesec.models.vlm import AnalysisResult


class Clip(BaseModel):
    """Represents a finalized video clip ready for processing."""

    clip_id: str
    camera_name: str
    local_path: Path
    start_ts: datetime
    end_ts: datetime
    duration_s: float
    source_backend: str  # "rtsp", "ftp", etc.


class ClipStateData(BaseModel):
    """Lightweight snapshot of current clip state (stored in clip_states.data JSONB)."""

    schema_version: int = 1
    camera_name: str

    # High-level status for queries
    status: ClipStatus

    # Pointers
    local_path: str
    storage_uri: str | None = None
    view_url: str | None = None

    # Latest results (denormalized for fast access)
    filter_result: FilterResult | None = None
    analysis_result: AnalysisResult | None = None
    alert_decision: AlertDecision | None = None

    @property
    def upload_completed(self) -> bool:
        """Check if upload stage completed."""
        return self.storage_uri is not None

    @property
    def filter_completed(self) -> bool:
        """Check if filter stage completed."""
        return self.filter_result is not None

    @property
    def vlm_completed(self) -> bool:
        """Check if VLM stage completed."""
        return self.analysis_result is not None


# Resolve forward references after imports are available
def _resolve_forward_refs() -> None:
    """Resolve forward references in ClipStateData."""
    # Explicitly import types to make them available for model_rebuild
    from homesec.models.alert import AlertDecision
    from homesec.models.filter import FilterResult
    from homesec.models.vlm import AnalysisResult

    ClipStateData.model_rebuild(
        _types_namespace={
            "FilterResult": FilterResult,
            "AnalysisResult": AnalysisResult,
            "AlertDecision": AlertDecision,
        }
    )
