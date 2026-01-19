"""Tests for clip source health semantics."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import pytest

from homesec.sources.base import AsyncClipSource, ThreadedClipSource


async def _wait_for(
    condition: Callable[[], bool], timeout_s: float = 1.0, interval_s: float = 0.01
) -> None:
    start = time.monotonic()
    while True:
        if condition():
            return
        if time.monotonic() - start > timeout_s:
            raise AssertionError("Condition not met before timeout")
        await asyncio.sleep(interval_s)


class _ImmediateThreadSource(ThreadedClipSource):
    def _run(self) -> None:
        return


class _ImmediateAsyncSource(AsyncClipSource):
    async def _run(self) -> None:
        return


@pytest.mark.asyncio
async def test_threaded_source_unhealthy_after_exit() -> None:
    """Threaded sources should be unhealthy after they exit."""
    # Given: A threaded source that exits immediately
    source = _ImmediateThreadSource()

    # Then: It is healthy before starting
    assert source.is_healthy()
    assert await source.ping()

    # When: The source is started and exits
    await source.start()
    await _wait_for(lambda: not source.is_healthy())

    # Then: Health checks report unhealthy
    assert not source.is_healthy()
    assert not await source.ping()


@pytest.mark.asyncio
async def test_async_source_unhealthy_after_exit() -> None:
    """Async sources should be unhealthy after they exit."""
    # Given: An async source that exits immediately
    source = _ImmediateAsyncSource()

    # Then: It is healthy before starting
    assert source.is_healthy()
    assert await source.ping()

    # When: The source is started and exits
    await source.start()
    await _wait_for(lambda: not source.is_healthy())

    # Then: Health checks report unhealthy
    assert not source.is_healthy()
    assert not await source.ping()
