"""Mock state store for testing."""

from __future__ import annotations

import asyncio
from datetime import datetime

from homesec.models.clip import ClipStateData
from homesec.models.enums import ClipStatus


class MockStateStore:
    """Mock implementation of StateStore interface for testing.

    Stores clip states in memory dict for test assertions.
    Supports configurable failure injection and delays.
    """

    def __init__(
        self,
        simulate_failure: bool = False,
        delay_s: float = 0.0,
    ) -> None:
        """Initialize mock state store.

        Args:
            simulate_failure: If True, upsert/get operations raise RuntimeError
            delay_s: Artificial delay before returning
        """
        self.simulate_failure = simulate_failure
        self.delay_s = delay_s
        self.states: dict[str, ClipStateData] = {}  # clip_id -> state
        self.upsert_count = 0
        self.get_count = 0

    async def upsert(self, clip_id: str, data: ClipStateData) -> None:
        """Insert or update clip state (mock implementation)."""
        self.upsert_count += 1

        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)

        if self.simulate_failure:
            raise RuntimeError("Simulated state store upsert failure")

        self.states[clip_id] = data

    async def get(self, clip_id: str) -> ClipStateData | None:
        """Retrieve clip state (mock implementation)."""
        self.get_count += 1

        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)

        if self.simulate_failure:
            raise RuntimeError("Simulated state store get failure")

        return self.states.get(clip_id)

    async def get_clip(self, clip_id: str) -> ClipStateData | None:
        """Retrieve clip state by ID (mock implementation)."""
        return await self.get(clip_id)

    async def list_clips(
        self,
        *,
        camera: str | None = None,
        status: ClipStatus | None = None,
        alerted: bool | None = None,
        risk_level: str | None = None,
        activity_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[ClipStateData], int]:
        items = list(self.states.values())

        if camera is not None:
            items = [item for item in items if item.camera_name == camera]
        if status is not None:
            items = [item for item in items if str(item.status) == status.value]
        if alerted is True:
            items = [item for item in items if item.alert_decision and item.alert_decision.notify]
        elif alerted is False:
            items = [
                item for item in items if not (item.alert_decision and item.alert_decision.notify)
            ]
        if risk_level is not None:
            items = [
                item
                for item in items
                if item.analysis_result and str(item.analysis_result.risk_level) == risk_level
            ]
        if activity_type is not None:
            items = [
                item
                for item in items
                if item.analysis_result and item.analysis_result.activity_type == activity_type
            ]
        if since is not None:
            items = [item for item in items if item.created_at and item.created_at >= since]
        if until is not None:
            items = [item for item in items if item.created_at and item.created_at <= until]

        total = len(items)
        sliced = items[int(offset) : int(offset) + int(limit)]
        return (sliced, total)

    async def mark_clip_deleted(self, clip_id: str) -> ClipStateData:
        state = self.states.get(clip_id)
        if state is None:
            raise ValueError(f"Clip not found: {clip_id}")
        state.status = ClipStatus.DELETED
        self.states[clip_id] = state
        return state

    async def count_clips_since(self, since: datetime) -> int:
        _ = since
        return len(self.states)

    async def count_alerts_since(self, since: datetime) -> int:
        _ = since
        return 0

    async def list_candidate_clips_for_cleanup(
        self,
        *,
        older_than_days: int | None,
        camera_name: str | None,
        batch_size: int,
        cursor: tuple[datetime, str] | None = None,
    ) -> list[tuple[str, ClipStateData, datetime]]:
        _ = older_than_days
        _ = camera_name
        _ = batch_size
        _ = cursor
        return []

    async def ping(self) -> bool:
        """Health check (mock implementation)."""
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)

        return not self.simulate_failure

    async def shutdown(self, timeout: float | None = None) -> None:
        """Cleanup resources (no-op for mock)."""
        _ = timeout
        return None
