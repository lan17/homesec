"""Tests for APIServer lifecycle behavior."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI

from homesec.api.server import APIServer


@pytest.mark.asyncio
async def test_api_server_start_and_stop_waits_for_started(monkeypatch: pytest.MonkeyPatch) -> None:
    """APIServer should wait for startup and stop cleanly."""
    created_servers: list[_RunningServer] = []

    class _RunningServer:
        def __init__(self, config: object) -> None:
            self.config = config
            self.started = False
            self.should_exit = False
            self.install_signal_handlers = True
            created_servers.append(self)

        async def serve(self) -> None:
            self.started = True
            while not self.should_exit:
                await asyncio.sleep(0)

    # Given: uvicorn.Server is replaced with a controllable fake
    monkeypatch.setattr("homesec.api.server.uvicorn.Server", _RunningServer)
    server = APIServer(FastAPI(), host="127.0.0.1", port=8123, startup_timeout_s=0.2)

    # When: Starting and stopping the API server
    await server.start()
    await server.stop()

    # Then: Startup waited for readiness and shutdown requested exit
    assert len(created_servers) == 1
    assert created_servers[0].started is True
    assert created_servers[0].should_exit is True


@pytest.mark.asyncio
async def test_api_server_start_raises_when_uvicorn_crashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """APIServer should fail fast when uvicorn fails before startup completes."""

    class _FailingServer:
        def __init__(self, config: object) -> None:
            self.config = config
            self.started = False
            self.should_exit = False
            self.install_signal_handlers = True

        async def serve(self) -> None:
            raise RuntimeError("bind failed")

    # Given: uvicorn.Server fails immediately during startup
    monkeypatch.setattr("homesec.api.server.uvicorn.Server", _FailingServer)
    server = APIServer(FastAPI(), host="127.0.0.1", port=8124, startup_timeout_s=0.2)

    # When: Starting the API server
    with pytest.raises(RuntimeError, match="bind failed"):
        await server.start()

    # Then: Stop remains a no-op after failed startup
    await server.stop()
