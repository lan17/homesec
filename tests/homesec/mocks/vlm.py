"""Mock VLM analyzer for testing."""

from __future__ import annotations

import asyncio
from pathlib import Path

from homesec.errors import VLMError
from homesec.models.filter import FilterResult
from homesec.models.vlm import AnalysisResult, VLMConfig


class MockVLM:
    """Mock implementation of VLMAnalyzer interface for testing.

    Supports configurable failure injection and delays to test
    concurrency and error handling scenarios.
    """

    def __init__(
        self,
        simulate_failure: bool = False,
        delay_s: float = 0.0,
        result: AnalysisResult | None = None,
    ) -> None:
        """Initialize mock VLM analyzer.

        Args:
            simulate_failure: If True, analyze() raises VLMError
            delay_s: Artificial delay before returning result
            result: AnalysisResult to return (uses default if None)
        """
        self.simulate_failure = simulate_failure
        self.delay_s = delay_s
        self.result = result or AnalysisResult(
            risk_level="low",
            activity_type="person_passing",
            summary="Person walked through frame",
        )
        self.analyze_count = 0
        self.shutdown_called = False

    async def analyze(
        self, video_path: Path, filter_result: FilterResult, config: VLMConfig
    ) -> AnalysisResult:
        """Analyze clip and produce structured assessment (mock implementation)."""
        self.analyze_count += 1

        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)

        if self.simulate_failure:
            raise VLMError(
                clip_id=video_path.stem,
                plugin_name=config.backend,
                cause=RuntimeError("Simulated VLM failure"),
            )

        return self.result

    async def shutdown(self, timeout: float | None = None) -> None:
        """Cleanup resources (no-op for mock)."""
        _ = timeout
        self.shutdown_called = True
