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


@pytest.mark.asyncio
async def test_api_server_start_raises_when_server_exits_before_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """APIServer should fail when server task exits cleanly before startup flag is set."""

    class _EarlyExitServer:
        def __init__(self, config: object) -> None:
            self.config = config
            self.started = False
            self.should_exit = False
            self.install_signal_handlers = True

        async def serve(self) -> None:
            return None

    # Given: uvicorn.Server exits immediately without reporting started
    monkeypatch.setattr("homesec.api.server.uvicorn.Server", _EarlyExitServer)
    server = APIServer(FastAPI(), host="127.0.0.1", port=8125, startup_timeout_s=0.2)

    # When: Starting the API server
    with pytest.raises(RuntimeError, match="exited before startup completed"):
        await server.start()

    # Then: Stop remains a no-op after startup failure
    await server.stop()


@pytest.mark.asyncio
async def test_api_server_start_times_out_when_started_flag_never_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """APIServer should time out if startup never completes."""

    class _HangingServer:
        def __init__(self, config: object) -> None:
            self.config = config
            self.started = False
            self.should_exit = False
            self.install_signal_handlers = True

        async def serve(self) -> None:
            while not self.should_exit:
                await asyncio.sleep(0)

    # Given: uvicorn.Server keeps running without setting started=True
    monkeypatch.setattr("homesec.api.server.uvicorn.Server", _HangingServer)
    server = APIServer(FastAPI(), host="127.0.0.1", port=8126, startup_timeout_s=0.01)

    # When: Starting the API server
    with pytest.raises(TimeoutError, match="Timed out waiting for API server startup"):
        await server.start()

    # Then: stop remains safe even after timeout failure path
    await server.stop()


@pytest.mark.asyncio
async def test_api_server_start_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """APIServer.start should not create a second uvicorn server when already running."""
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

    # Given: a controllable running uvicorn server
    monkeypatch.setattr("homesec.api.server.uvicorn.Server", _RunningServer)
    server = APIServer(FastAPI(), host="127.0.0.1", port=8127, startup_timeout_s=0.2)

    # When: start() is called twice
    await server.start()
    await server.start()

    # Then: only one server is created and shutdown still works
    assert len(created_servers) == 1
    await server.stop()


@pytest.mark.asyncio
async def test_api_server_stop_swallows_server_exit_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """APIServer.stop should log and swallow task exceptions raised during shutdown."""

    class _ErrorOnStopServer:
        def __init__(self, config: object) -> None:
            self.config = config
            self.started = False
            self.should_exit = False
            self.install_signal_handlers = True

        async def serve(self) -> None:
            self.started = True
            while not self.should_exit:
                await asyncio.sleep(0)
            raise RuntimeError("stop failed")

    # Given: a server that raises when should_exit is set during stop
    monkeypatch.setattr("homesec.api.server.uvicorn.Server", _ErrorOnStopServer)
    server = APIServer(FastAPI(), host="127.0.0.1", port=8128, startup_timeout_s=0.2)
    await server.start()

    # When: stopping the API server
    await server.stop()

    # Then: stop completes without propagating the server exception
    await server.stop()
