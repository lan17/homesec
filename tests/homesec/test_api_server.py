"""Tests for APIServer lifecycle behavior."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI

from homesec.api import server as api_server
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


@pytest.mark.asyncio
async def test_wait_until_started_requires_initialized_server() -> None:
    """_wait_until_started should fail fast when start() has not initialized internals."""
    # Given: A freshly constructed API server that has not been started
    server = APIServer(FastAPI(), host="127.0.0.1", port=8130, startup_timeout_s=0.2)

    # When: Waiting on startup state directly
    with pytest.raises(RuntimeError, match="task not initialized"):
        await server._wait_until_started()


def test_spa_path_helpers_cover_edge_cases() -> None:
    """SPA helper predicates should classify reserved/static/route paths correctly."""
    # Given: Representative edge-case path values
    empty_path = ""
    health_path = "health"
    nested_api_path = "api/v1/health"
    favicon_path = "favicon.ico"
    dotted_route = "clips/clip-1.mp4"
    route_without_ext = "clips/clip-1"

    # When: Evaluating SPA path predicates
    is_empty_static = api_server._is_allowed_root_static_file(empty_path)
    is_favicon_static = api_server._is_allowed_root_static_file(favicon_path)
    empty_is_spa_route = api_server._is_spa_route_path(empty_path)
    dotted_is_spa_route = api_server._is_spa_route_path(dotted_route)
    route_is_spa_route = api_server._is_spa_route_path(route_without_ext)
    empty_is_reserved = api_server._is_reserved_spa_path(empty_path)
    health_is_reserved = api_server._is_reserved_spa_path(health_path)
    nested_api_is_reserved = api_server._is_reserved_spa_path(nested_api_path)

    # Then: Classification matches intended SPA routing behavior
    assert is_empty_static is False
    assert is_favicon_static is True
    assert empty_is_spa_route is True
    assert dotted_is_spa_route is False
    assert route_is_spa_route is True
    assert empty_is_reserved is False
    assert health_is_reserved is True
    assert nested_api_is_reserved is True
