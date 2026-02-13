"""Tests for application startup fail-fast behavior on source preflight failures."""

from __future__ import annotations

from typing import Any, cast

import pytest

from homesec.runtime.assembly import RuntimeAssembler
from homesec.runtime.models import RuntimeBundle


class _StubSource:
    def __init__(self, *, camera_name: str, fail_start: bool = False) -> None:
        self.camera_name = camera_name
        self._fail_start = fail_start
        self.shutdown_called = False

    def register_callback(self, callback: object) -> None:
        _ = callback

    async def start(self) -> None:
        if self._fail_start:
            raise RuntimeError("preflight failed")

    def is_healthy(self) -> bool:
        return True

    def last_heartbeat(self) -> float:
        return 0.0

    async def ping(self) -> bool:
        return True

    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout
        self.shutdown_called = True


class _NoopShutdown:
    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout


async def _noop_notifier_health(entries: list[object]) -> None:
    _ = entries


@pytest.mark.asyncio
async def test_runtime_bundle_startup_fails_fast_when_any_source_fails() -> None:
    """Runtime startup should shutdown started sources and fail when one source start fails."""
    # Given: One source that starts and one source that fails startup
    good_source = _StubSource(camera_name="front_door")
    bad_source = _StubSource(camera_name="garage", fail_start=True)
    assembler = RuntimeAssembler(
        storage=cast(Any, _NoopShutdown()),
        repository=cast(Any, object()),
        notifier_factory=lambda _config: (
            cast(Any, _NoopShutdown()),
            [],
        ),
        notifier_health_logger=cast(Any, _noop_notifier_health),
        alert_policy_factory=lambda _config: cast(Any, object()),
        source_factory=lambda _config: ([], {}),
    )
    runtime = RuntimeBundle(
        generation=1,
        config=cast(Any, object()),
        config_signature="cfgsig",
        notifier=cast(Any, object()),
        notifier_entries=[],
        filter_plugin=cast(Any, object()),
        vlm_plugin=cast(Any, object()),
        alert_policy=cast(Any, object()),
        pipeline=cast(Any, object()),
        sources=[good_source, bad_source],
        sources_by_camera={"front_door": good_source, "garage": bad_source},
    )

    # When: Starting runtime sources for preflight
    with pytest.raises(RuntimeError, match="Source startup preflight failed"):
        await assembler.start_bundle(runtime)

    # Then: Already-started sources are cleanly shut down
    assert good_source.shutdown_called
