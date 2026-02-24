"""Setup/onboarding API models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

SetupState = Literal["fresh", "partial", "complete"]


class SetupStatusResponse(BaseModel):
    """Current setup completion state used by onboarding flows."""

    state: SetupState
    has_cameras: bool
    pipeline_running: bool
    auth_configured: bool


class PreflightCheckResponse(BaseModel):
    """Single preflight check outcome."""

    name: str
    passed: bool
    message: str
    latency_ms: float | None = None


class PreflightResponse(BaseModel):
    """Aggregated preflight response."""

    checks: list[PreflightCheckResponse]
    all_passed: bool
