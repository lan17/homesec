"""Mock event store for testing."""

from __future__ import annotations

import asyncio

from homesec.models.events import ClipLifecycleEvent


class MockEventStore:
    """Mock implementation of EventStore interface for testing.

    Stores events in memory and assigns incremental ids to mimic Postgres.
    Supports configurable failure injection and delays.
    """

    def __init__(
        self,
        simulate_failure: bool = False,
        delay_s: float = 0.0,
    ) -> None:
        self.simulate_failure = simulate_failure
        self.delay_s = delay_s
        self.events: list[ClipLifecycleEvent] = []
        self.append_count = 0
        self.get_count = 0
        self._next_id = 1

    async def append(self, event: ClipLifecycleEvent) -> None:
        """Append event (mock implementation)."""
        self.append_count += 1
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        if self.simulate_failure:
            raise RuntimeError("Simulated event store append failure")
        assigned = event.model_copy(update={"id": self._next_id})
        self._next_id += 1
        self.events.append(assigned)

    async def get_events(
        self,
        clip_id: str,
        after_id: int | None = None,
    ) -> list[ClipLifecycleEvent]:
        """Retrieve events (mock implementation)."""
        self.get_count += 1
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        if self.simulate_failure:
            raise RuntimeError("Simulated event store get failure")
        result = [event for event in self.events if event.clip_id == clip_id]
        if after_id is not None:
            result = [event for event in result if event.id is not None and event.id > after_id]
        return result
