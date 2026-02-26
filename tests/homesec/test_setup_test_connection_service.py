"""Tests for setup test-connection service flow."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from homesec.models.setup import TestConnectionRequest as SetupTestConnectionRequest
from homesec.services import setup as setup_service


@dataclass
class _StubApp:
    """Minimal app stub for setup service API."""


class _DummyConfig(BaseModel):
    value: str = "ok"


class _StubPingPlugin:
    def __init__(self, *, ping_result: bool = True, delay_s: float = 0.0) -> None:
        self._ping_result = ping_result
        self._delay_s = delay_s
        self.shutdown_calls = 0

    async def ping(self) -> bool:
        if self._delay_s > 0:
            await asyncio.sleep(self._delay_s)
        return self._ping_result

    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout
        self.shutdown_calls += 1


@pytest.mark.asyncio
async def test_test_connection_camera_local_folder_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Camera local_folder test should pass when watch dir is accessible."""
    # Given: A valid local_folder config that points at an existing writable directory
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir(parents=True)

    def _fake_validate_plugin(
        plugin_type: object,
        backend: str,
        config: object,
        **runtime_context: object,
    ) -> object:
        _ = (plugin_type, config, runtime_context)
        assert backend == "local_folder"
        return setup_service.LocalFolderSourceConfig(watch_dir=str(watch_dir))

    monkeypatch.setattr(setup_service, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["local_folder"]
        if plugin_type == setup_service.PluginType.SOURCE
        else [],
    )
    request = SetupTestConnectionRequest(type="camera", backend="local_folder", config={})

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: The connection test succeeds and reports the checked directory
    assert response.success is True
    assert "accessible" in response.message
    assert response.details is not None
    assert response.details["watch_dir"] == str(watch_dir.resolve())


@pytest.mark.asyncio
async def test_test_connection_camera_local_folder_missing_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Camera local_folder test should fail when watch dir is missing."""
    # Given: A local_folder config that points at a missing directory
    watch_dir = tmp_path / "missing-watch-dir"

    def _fake_validate_plugin(
        plugin_type: object,
        backend: str,
        config: object,
        **runtime_context: object,
    ) -> object:
        _ = (plugin_type, config, runtime_context)
        assert backend == "local_folder"
        return setup_service.LocalFolderSourceConfig(watch_dir=str(watch_dir))

    monkeypatch.setattr(setup_service, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["local_folder"]
        if plugin_type == setup_service.PluginType.SOURCE
        else [],
    )
    request = SetupTestConnectionRequest(type="camera", backend="local_folder", config={})

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: The connection test reports a non-successful validation outcome
    assert response.success is False
    assert "does not exist" in response.message


@pytest.mark.asyncio
async def test_test_connection_notifier_uses_ping_and_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Notifier test should ping ephemeral plugin and always shut it down."""
    # Given: A notifier backend that validates and creates a pingable ephemeral plugin
    plugin = _StubPingPlugin(ping_result=True)

    def _fake_get_plugin_names(plugin_type: setup_service.PluginType) -> list[str]:
        assert plugin_type == setup_service.PluginType.NOTIFIER
        return ["mqtt"]

    def _fake_validate_plugin(
        plugin_type: setup_service.PluginType,
        backend: str,
        config: dict[str, Any] | BaseModel,
        **runtime_context: object,
    ) -> BaseModel:
        _ = (config, runtime_context)
        assert plugin_type == setup_service.PluginType.NOTIFIER
        assert backend == "mqtt"
        return _DummyConfig()

    def _fake_load_plugin(
        plugin_type: setup_service.PluginType,
        backend: str,
        config: dict[str, Any] | BaseModel,
        **runtime_context: object,
    ) -> _StubPingPlugin:
        _ = (config, runtime_context)
        assert plugin_type == setup_service.PluginType.NOTIFIER
        assert backend == "mqtt"
        return plugin

    monkeypatch.setattr(setup_service, "get_plugin_names", _fake_get_plugin_names)
    monkeypatch.setattr(setup_service, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(setup_service, "load_plugin", _fake_load_plugin)
    request = SetupTestConnectionRequest(type="notifier", backend="mqtt", config={})

    # When: Running notifier test-connection
    response = await setup_service.test_connection(request, _StubApp())

    # Then: Probe succeeds and plugin shutdown executes exactly once
    assert response.success is True
    assert plugin.shutdown_calls == 1


@pytest.mark.asyncio
async def test_test_connection_notifier_timeout_returns_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Notifier test should return success=false when ping exceeds timeout."""
    # Given: A notifier plugin whose ping blocks longer than timeout budget
    plugin = _StubPingPlugin(ping_result=True, delay_s=0.05)

    def _fake_get_plugin_names(plugin_type: setup_service.PluginType) -> list[str]:
        assert plugin_type == setup_service.PluginType.NOTIFIER
        return ["mqtt"]

    def _fake_validate_plugin(
        plugin_type: setup_service.PluginType,
        backend: str,
        config: dict[str, Any] | BaseModel,
        **runtime_context: object,
    ) -> BaseModel:
        _ = (config, runtime_context)
        assert plugin_type == setup_service.PluginType.NOTIFIER
        assert backend == "mqtt"
        return _DummyConfig()

    def _fake_load_plugin(
        plugin_type: setup_service.PluginType,
        backend: str,
        config: dict[str, Any] | BaseModel,
        **runtime_context: object,
    ) -> _StubPingPlugin:
        _ = (config, runtime_context)
        assert plugin_type == setup_service.PluginType.NOTIFIER
        assert backend == "mqtt"
        return plugin

    monkeypatch.setattr(setup_service, "_TEST_CONNECTION_TIMEOUT_S", 0.01)
    monkeypatch.setattr(setup_service, "get_plugin_names", _fake_get_plugin_names)
    monkeypatch.setattr(setup_service, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(setup_service, "load_plugin", _fake_load_plugin)
    request = SetupTestConnectionRequest(type="notifier", backend="mqtt", config={})

    # When: Running notifier test-connection
    response = await setup_service.test_connection(request, _StubApp())

    # Then: Probe reports timeout as a non-successful test outcome
    assert response.success is False
    assert "timed out" in response.message
    assert plugin.shutdown_calls == 1


@pytest.mark.asyncio
async def test_test_connection_camera_rtsp_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camera rtsp test should return probe diagnostics on successful preflight."""

    # Given: An RTSP backend with a successful startup-preflight result
    def _fake_validate_plugin(
        plugin_type: object,
        backend: str,
        config: object,
        **runtime_context: object,
    ) -> object:
        _ = (plugin_type, config, runtime_context)
        assert backend == "rtsp"
        return setup_service.RTSPSourceConfig(rtsp_url="rtsp://example.local/stream")

    class _FakeStartupPreflight:
        def __init__(
            self,
            *,
            output_dir: Path,
            rtsp_connect_timeout_s: float,
            rtsp_io_timeout_s: float,
            command_timeout_s: float,
        ) -> None:
            _ = (output_dir, rtsp_connect_timeout_s, rtsp_io_timeout_s, command_timeout_s)

        def run(
            self,
            *,
            camera_name: str,
            primary_rtsp_url: str,
            detect_rtsp_url: str,
        ) -> object:
            _ = (camera_name, primary_rtsp_url, detect_rtsp_url)
            return SimpleNamespace(
                diagnostics=SimpleNamespace(
                    session_mode="single_stream",
                    selected_recording_profile="copy_audio",
                    probes=[{"stream": "main"}],
                )
            )

    monkeypatch.setattr(setup_service, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(setup_service, "RTSPStartupPreflight", _FakeStartupPreflight)
    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["rtsp"] if plugin_type == setup_service.PluginType.SOURCE else [],
    )
    request = SetupTestConnectionRequest(type="camera", backend="rtsp", config={})

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: The service reports RTSP probe success with diagnostic details
    assert response.success is True
    assert response.details is not None
    assert response.details["session_mode"] == "single_stream"
    assert response.details["selected_recording_profile"] == "copy_audio"
    assert response.details["probed_streams"] == 1


@pytest.mark.asyncio
async def test_test_connection_camera_ftp_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camera ftp test should pass when bind probe succeeds."""

    # Given: An FTP backend and a deterministic bind-probe result
    def _fake_validate_plugin(
        plugin_type: object,
        backend: str,
        config: object,
        **runtime_context: object,
    ) -> object:
        _ = (plugin_type, config, runtime_context)
        assert backend == "ftp"
        return setup_service.FtpSourceConfig(host="127.0.0.1", port=0)

    monkeypatch.setattr(setup_service, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(setup_service, "_probe_tcp_bind", lambda *_args: 42424)
    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["ftp"] if plugin_type == setup_service.PluginType.SOURCE else [],
    )
    request = SetupTestConnectionRequest(type="camera", backend="ftp", config={})

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: The service reports FTP probe success and returns bound port details
    assert response.success is True
    assert response.details is not None
    assert response.details["bound_port"] == 42424


@pytest.mark.asyncio
async def test_test_connection_camera_onvif_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camera onvif test should return profile/stream counts on successful probe."""

    # Given: An ONVIF service stub that returns a successful probe result
    class _FakeOnvifService:
        def __init__(self, *, discover_fn: object, client_factory: object) -> None:
            _ = (discover_fn, client_factory)

        async def probe(self, options: object) -> object:
            _ = options
            return SimpleNamespace(profiles=[object(), object()], streams=[object()])

    monkeypatch.setattr(setup_service, "OnvifService", _FakeOnvifService)
    request = SetupTestConnectionRequest(
        type="camera",
        backend="onvif",
        config={
            "host": "192.168.1.10",
            "username": "admin",
            "password": "secret",
        },
    )

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: The service reports ONVIF probe success with stream/profile counts
    assert response.success is True
    assert response.details is not None
    assert response.details["profiles"] == 2
    assert response.details["streams"] == 1


@pytest.mark.asyncio
async def test_test_connection_storage_local_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Storage local test should pass when root directory is accessible."""
    # Given: A local storage backend pointing at an existing writable directory
    root = tmp_path / "clips"
    root.mkdir(parents=True)

    def _fake_validate_plugin(
        plugin_type: object,
        backend: str,
        config: object,
        **runtime_context: object,
    ) -> object:
        _ = (plugin_type, config, runtime_context)
        assert backend == "local"
        return setup_service.LocalStorageConfig(root=str(root))

    monkeypatch.setattr(setup_service, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["local"] if plugin_type == setup_service.PluginType.STORAGE else [],
    )
    request = SetupTestConnectionRequest(type="storage", backend="local", config={})

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: The service reports local storage path is usable
    assert response.success is True
    assert response.details is not None
    assert response.details["root"] == str(root.resolve())


@pytest.mark.asyncio
async def test_test_connection_analyzer_uses_ping_and_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analyzer test should reuse generic plugin ping flow."""
    # Given: A registered analyzer backend with a pingable plugin implementation
    plugin = _StubPingPlugin(ping_result=True)

    def _fake_get_plugin_names(plugin_type: setup_service.PluginType) -> list[str]:
        assert plugin_type == setup_service.PluginType.ANALYZER
        return ["openai"]

    def _fake_validate_plugin(
        plugin_type: setup_service.PluginType,
        backend: str,
        config: dict[str, Any] | BaseModel,
        **runtime_context: object,
    ) -> BaseModel:
        _ = (config, runtime_context)
        assert plugin_type == setup_service.PluginType.ANALYZER
        assert backend == "openai"
        return _DummyConfig()

    def _fake_load_plugin(
        plugin_type: setup_service.PluginType,
        backend: str,
        config: dict[str, Any] | BaseModel,
        **runtime_context: object,
    ) -> _StubPingPlugin:
        _ = (config, runtime_context)
        assert plugin_type == setup_service.PluginType.ANALYZER
        assert backend == "openai"
        return plugin

    monkeypatch.setattr(setup_service, "get_plugin_names", _fake_get_plugin_names)
    monkeypatch.setattr(setup_service, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(setup_service, "load_plugin", _fake_load_plugin)
    request = SetupTestConnectionRequest(type="analyzer", backend="openai", config={})

    # When: Running analyzer test-connection
    response = await setup_service.test_connection(request, _StubApp())

    # Then: Probe succeeds and plugin shutdown executes exactly once
    assert response.success is True
    assert plugin.shutdown_calls == 1


@pytest.mark.asyncio
async def test_test_connection_analyzer_reports_ping_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analyzer test should fail when plugin ping returns false."""
    # Given: A registered analyzer backend whose ping reports unhealthy
    plugin = _StubPingPlugin(ping_result=False)

    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["openai"] if plugin_type == setup_service.PluginType.ANALYZER else [],
    )
    monkeypatch.setattr(setup_service, "validate_plugin", lambda *_args, **_kwargs: _DummyConfig())
    monkeypatch.setattr(setup_service, "load_plugin", lambda *_args, **_kwargs: plugin)
    request = SetupTestConnectionRequest(type="analyzer", backend="openai", config={})

    # When: Running analyzer test-connection
    response = await setup_service.test_connection(request, _StubApp())

    # Then: Probe returns a non-successful health result without raising
    assert response.success is False
    assert "did not pass health checks" in response.message
    assert plugin.shutdown_calls == 1


@pytest.mark.asyncio
async def test_test_connection_analyzer_reports_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analyzer test should fail when plugin initialization raises."""
    # Given: A registered analyzer backend whose plugin loader raises an exception
    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["openai"] if plugin_type == setup_service.PluginType.ANALYZER else [],
    )
    monkeypatch.setattr(setup_service, "validate_plugin", lambda *_args, **_kwargs: _DummyConfig())

    def _failing_loader(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("unable to initialize analyzer")

    monkeypatch.setattr(setup_service, "load_plugin", _failing_loader)
    request = SetupTestConnectionRequest(type="analyzer", backend="openai", config={})

    # When: Running analyzer test-connection
    response = await setup_service.test_connection(request, _StubApp())

    # Then: Probe returns failure response with plugin error details
    assert response.success is False
    assert "probe failed" in response.message


@pytest.mark.asyncio
async def test_test_connection_analyzer_ignores_shutdown_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analyzer test should preserve successful ping result even if shutdown fails."""

    # Given: A pingable analyzer plugin that raises during shutdown cleanup
    class _ShutdownFailingPlugin(_StubPingPlugin):
        async def shutdown(self, timeout: float | None = None) -> None:
            _ = timeout
            raise RuntimeError("cleanup failed")

    plugin = _ShutdownFailingPlugin(ping_result=True)
    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["openai"] if plugin_type == setup_service.PluginType.ANALYZER else [],
    )
    monkeypatch.setattr(setup_service, "validate_plugin", lambda *_args, **_kwargs: _DummyConfig())
    monkeypatch.setattr(setup_service, "load_plugin", lambda *_args, **_kwargs: plugin)
    request = SetupTestConnectionRequest(type="analyzer", backend="openai", config={})

    # When: Running analyzer test-connection
    response = await setup_service.test_connection(request, _StubApp())

    # Then: Successful probe result is returned despite shutdown cleanup error
    assert response.success is True


@pytest.mark.asyncio
async def test_test_connection_raises_for_unknown_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown backend should raise typed request error for route-level 400 mapping."""

    # Given: A storage test request for a backend that is not registered
    def _fake_get_plugin_names(plugin_type: setup_service.PluginType) -> list[str]:
        assert plugin_type == setup_service.PluginType.STORAGE
        return ["local", "dropbox"]

    monkeypatch.setattr(setup_service, "get_plugin_names", _fake_get_plugin_names)
    request = SetupTestConnectionRequest(type="storage", backend="s3", config={})

    # When / Then: Dispatch raises a typed request error with available backends attached
    with pytest.raises(setup_service.SetupTestConnectionRequestError) as exc_info:
        await setup_service.test_connection(request, _StubApp())

    assert "Unknown storage backend" in str(exc_info.value)
    assert exc_info.value.available_backends == ["local", "dropbox"]
