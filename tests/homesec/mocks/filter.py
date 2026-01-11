"""Mock object detection filter for testing."""

from __future__ import annotations

import asyncio
from pathlib import Path

from homesec.errors import FilterError
from homesec.models.filter import FilterOverrides, FilterResult


class MockFilter:
    """Mock implementation of ObjectFilter interface for testing.
    
    Supports configurable failure injection and delays to test
    concurrency and error handling scenarios.
    """

    def __init__(
        self,
        simulate_failure: bool = False,
        delay_s: float = 0.0,
        result: FilterResult | None = None,
        plugin_name: str = "mock_filter",
    ) -> None:
        """Initialize mock filter.
        
        Args:
            simulate_failure: If True, detect() raises FilterError
            delay_s: Artificial delay before returning result
            result: FilterResult to return (uses default if None)
        """
        self.simulate_failure = simulate_failure
        self.delay_s = delay_s
        self.result = result or FilterResult(
            detected_classes=["person"],
            confidence=0.85,
            model="mock_filter",
            sampled_frames=30,
        )
        self.plugin_name = plugin_name
        self.detect_count = 0
        self.shutdown_called = False

    async def detect(
        self, video_path: Path, overrides: FilterOverrides | None = None
    ) -> FilterResult:
        """Detect objects in video clip (mock implementation)."""
        _ = overrides
        self.detect_count += 1

        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)

        if self.simulate_failure:
            raise FilterError(
                clip_id=video_path.stem,
                plugin_name=self.plugin_name,
                cause=RuntimeError("Simulated filter failure"),
            )

        return self.result

    async def shutdown(self, timeout: float | None = None) -> None:
        """Cleanup resources (no-op for mock)."""
        _ = timeout
        self.shutdown_called = True
