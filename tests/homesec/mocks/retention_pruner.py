"""Mock retention pruner for pipeline/runtime tests."""

from __future__ import annotations

import asyncio

from homesec.retention import RetentionPruneSummary


def _default_summary(*, reason: str) -> RetentionPruneSummary:
    return RetentionPruneSummary(
        reason=reason,
        max_local_size_bytes=0,
        discovered_local_files=0,
        measured_local_files=0,
        unmeasured_local_files=0,
        measured_local_bytes=0,
        non_eligible_local_bytes=0,
        measurement_incomplete=False,
        blocked_over_limit=False,
        eligible_candidates=0,
        eligible_bytes_before=0,
        eligible_bytes_after=0,
        reclaimed_bytes=0,
        deleted_files=0,
        skipped_not_done=0,
        skipped_no_state=0,
        skipped_not_uploaded=0,
        skipped_path_mismatch=0,
        skipped_stat_error=0,
        skipped_missing_race=0,
        delete_errors=0,
    )


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

    async def prune_once(self, *, reason: str) -> RetentionPruneSummary:
        self.reasons.append(reason)
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        if self.simulate_failure:
            raise RuntimeError("Simulated retention prune failure")
        return _default_summary(reason=reason)
