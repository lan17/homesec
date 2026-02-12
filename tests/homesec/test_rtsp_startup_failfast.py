"""Tests for application startup fail-fast behavior on source preflight failures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from homesec.app import Application


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


@pytest.mark.asyncio
async def test_application_fails_startup_when_any_source_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Application should shutdown started sources and fail when one source start fails."""
    # Given: one source that starts and one that fails startup
    good_source = _StubSource(camera_name="front_door")
    bad_source = _StubSource(camera_name="garage", fail_start=True)
    app = Application(config_path=Path("config/config.yaml"))

    async def _fake_create_components(self: Application) -> None:
        self._sources = [good_source, bad_source]
        self._health_server = None

    monkeypatch.setattr(Application, "_create_components", _fake_create_components)
    monkeypatch.setattr("homesec.app.load_config", lambda _path: cast(Any, object()))
    monkeypatch.setattr(Application, "_setup_signal_handlers", lambda _self: None)

    # When: running the application startup sequence
    with pytest.raises(RuntimeError, match="Source startup preflight failed"):
        await app.run()

    # Then: already-started sources are cleanly shut down
    assert good_source.shutdown_called
