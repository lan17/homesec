"""Mock retention pruner for pipeline/runtime tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

from homesec.retention import RetentionPruneSummary


def _default_summary(*, reason: str) -> RetentionPruneSummary:
    return RetentionPruneSummary.empty(reason=reason)


class MockRetentionPruner:
    """Mock retention pruner with call tracking and failure simulation."""

    def __init__(
        self,
        *,
        delay_s: float = 0.0,
        simulate_failure: bool = False,
    ) -> None:
        self.delay_s = delay_s
        self.simulate_failure = simulate_failure
        self.reasons: list[str] = []
        self.clip_local_paths: list[Path | None] = []

    async def prune_once(
        self,
        *,
        reason: str,
        clip_local_path: Path | None = None,
    ) -> RetentionPruneSummary:
        self.reasons.append(reason)
        self.clip_local_paths.append(clip_local_path)
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        if self.simulate_failure:
            raise RuntimeError("Simulated retention prune failure")
        return _default_summary(reason=reason)
