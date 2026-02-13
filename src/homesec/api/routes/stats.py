"""System statistics endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from homesec.api.dependencies import get_homesec_app

if TYPE_CHECKING:
    from homesec.app import Application

router = APIRouter(tags=["stats"])


class StatsResponse(BaseModel):
    clips_today: int
    alerts_today: int
    cameras_total: int
    cameras_online: int
    uptime_seconds: float


@router.get("/api/v1/stats", response_model=StatsResponse)
async def get_stats(app: Application = Depends(get_homesec_app)) -> StatsResponse:
    """Return system statistics."""
    if app.bootstrap_mode:
        return StatsResponse(
            clips_today=0,
            alerts_today=0,
            cameras_total=0,
            cameras_online=0,
            uptime_seconds=app.uptime_seconds,
        )

    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    clips_today = await app.repository.count_clips_since(today_start)
    alerts_today = await app.repository.count_alerts_since(today_start)
    cameras_total = len(app.config.cameras)
    cameras_online = sum(1 for source in app.sources if source.is_healthy())

    return StatsResponse(
        clips_today=clips_today,
        alerts_today=alerts_today,
        cameras_total=cameras_total,
        cameras_online=cameras_online,
        uptime_seconds=app.uptime_seconds,
    )
