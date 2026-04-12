"""Tests for setup test-connection service flow."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

import homesec.onvif.setup_probe as onvif_setup_probe
import homesec.plugins.storage.local_setup_probe as local_storage_setup_probe
import homesec.sources.ftp_setup_probe as ftp_setup_probe
import homesec.sources.local_folder_setup_probe as local_folder_setup_probe
import homesec.sources.rtsp.setup_probe as rtsp_setup_probe
from homesec.models.setup import TestConnectionRequest as SetupTestConnectionRequest
from homesec.onvif.service import OnvifProbeError, OnvifProbeOptions, OnvifProbeTimeoutError
from homesec.plugins.storage.local import LocalStorageConfig
from homesec.services import setup as setup_service
from homesec.sources.ftp import FtpSourceConfig
from homesec.sources.local_folder import LocalFolderSourceConfig
from homesec.sources.rtsp.core import RTSPSourceConfig
from homesec.sources.rtsp.preflight import PreflightError


@dataclass
class _StubApp:
    """Minimal app stub for setup service API."""

    _setup_test_connection_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def setup_test_connection_lock(self) -> asyncio.Lock:
        return self._setup_test_connection_lock


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
        return LocalFolderSourceConfig(watch_dir=str(watch_dir))

    monkeypatch.setattr(local_folder_setup_probe, "validate_plugin", _fake_validate_plugin)
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
        return LocalFolderSourceConfig(watch_dir=str(watch_dir))

    monkeypatch.setattr(local_folder_setup_probe, "validate_plugin", _fake_validate_plugin)
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

    monkeypatch.setattr(setup_service, "_PLUGIN_TEST_CONNECTION_TIMEOUT_S", 0.01)
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
async def test_test_connection_notifier_normalizes_backend_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Notifier test should normalize backend strings before plugin dispatch."""
    # Given: A notifier request with whitespace and mixed-case backend name
    plugin = _StubPingPlugin(ping_result=True)

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

    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["mqtt"] if plugin_type == setup_service.PluginType.NOTIFIER else [],
    )
    monkeypatch.setattr(setup_service, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(setup_service, "load_plugin", _fake_load_plugin)
    request = SetupTestConnectionRequest(type="notifier", backend=" MQTT ", config={})

    # When: Running notifier test-connection
    response = await setup_service.test_connection(request, _StubApp())

    # Then: The probe succeeds after backend normalization in dispatcher
    assert response.success is True
    assert plugin.shutdown_calls == 1


@pytest.mark.asyncio
async def test_test_connection_camera_uses_registered_setup_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camera test-connection should prefer registered setup probes over generic ping."""

    # Given: A registry-backed camera probe for a backend without generic plugin wiring
    async def _fake_probe(*, config: dict[str, object]) -> setup_service.TestConnectionResponse:
        assert config == {"host": "192.168.1.10"}
        return setup_service.TestConnectionResponse(
            success=True,
            message="camera probe ok",
            details={"backend": "custom"},
        )

    monkeypatch.setattr(
        setup_service,
        "get_setup_probe",
        lambda target, backend: _fake_probe if target == "camera" and backend == "custom" else None,
    )
    monkeypatch.setattr(
        setup_service,
        "_test_plugin_ping_connection",
        lambda *_args, **_kwargs: pytest.fail("generic ping fallback should not be used"),
    )
    request = SetupTestConnectionRequest(
        type="camera",
        backend="custom",
        config={"host": "192.168.1.10"},
    )

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: The registered probe handles the request without falling back to plugin ping
    assert response.success is True
    assert response.message == "camera probe ok"
    assert response.details == {"backend": "custom"}


@pytest.mark.asyncio
async def test_test_connection_camera_registered_probe_failure_returns_failed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registered setup probe failures should return a failed probe response."""

    # Given: A registered camera probe that raises during connectivity testing
    async def _failing_probe(*, config: dict[str, object]) -> setup_service.TestConnectionResponse:
        _ = config
        raise RuntimeError("socket connect failed")

    monkeypatch.setattr(
        setup_service,
        "get_setup_probe",
        lambda target, backend: _failing_probe
        if target == "camera" and backend == "custom"
        else None,
    )
    request = SetupTestConnectionRequest(
        type="camera",
        backend="custom",
        config={"host": "192.168.1.10"},
    )

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: The API returns a failed probe response instead of raising
    assert response.success is False
    assert "failed during connectivity test" in response.message


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
        return RTSPSourceConfig(rtsp_url="rtsp://example.local/stream")

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

    monkeypatch.setattr(rtsp_setup_probe, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(rtsp_setup_probe, "RTSPStartupPreflight", _FakeStartupPreflight)
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
async def test_test_connection_camera_rtsp_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camera rtsp test should return config validation failure details."""

    # Given: An RTSP backend whose plugin validation raises a ValidationError
    def _fake_validate_plugin(
        plugin_type: object,
        backend: str,
        config: object,
        **runtime_context: object,
    ) -> object:
        _ = (plugin_type, backend, config, runtime_context)
        return RTSPSourceConfig.model_validate({})

    monkeypatch.setattr(rtsp_setup_probe, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["rtsp"] if plugin_type == setup_service.PluginType.SOURCE else [],
    )
    request = SetupTestConnectionRequest(type="camera", backend="rtsp", config={})

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: Validation failure is returned as a non-successful probe response
    assert response.success is False
    assert "Configuration validation failed" in response.message


@pytest.mark.asyncio
async def test_test_connection_camera_rtsp_env_url_missing_returns_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camera rtsp test should fail when env-based URL is configured but unset."""

    # Given: An RTSP config that only references a missing environment variable
    def _fake_validate_plugin(
        plugin_type: object,
        backend: str,
        config: object,
        **runtime_context: object,
    ) -> object:
        _ = (plugin_type, backend, config, runtime_context)
        return RTSPSourceConfig(rtsp_url_env="HOMESEC_RTSP_MISSING")

    monkeypatch.delenv("HOMESEC_RTSP_MISSING", raising=False)
    monkeypatch.setattr(rtsp_setup_probe, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["rtsp"] if plugin_type == setup_service.PluginType.SOURCE else [],
    )
    request = SetupTestConnectionRequest(type="camera", backend="rtsp", config={})

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: Service returns a clear failure explaining unresolved RTSP URL
    assert response.success is False
    assert "RTSP URL not resolved" in response.message


@pytest.mark.asyncio
async def test_test_connection_camera_rtsp_timeout_returns_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camera rtsp test should fail when startup preflight exceeds timeout budget."""

    # Given: An RTSP preflight implementation that takes longer than timeout budget
    def _fake_validate_plugin(
        plugin_type: object,
        backend: str,
        config: object,
        **runtime_context: object,
    ) -> object:
        _ = (plugin_type, backend, config, runtime_context)
        return RTSPSourceConfig(rtsp_url="rtsp://example.local/stream")

    class _SlowStartupPreflight:
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
            time.sleep(0.05)
            return SimpleNamespace(
                diagnostics=SimpleNamespace(
                    session_mode="single_stream",
                    selected_recording_profile="copy_audio",
                    probes=[{"stream": "main"}],
                )
            )

    monkeypatch.setattr(rtsp_setup_probe, "RTSP_TEST_CONNECTION_TIMEOUT_S", 0.01)
    monkeypatch.setattr(rtsp_setup_probe, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(rtsp_setup_probe, "RTSPStartupPreflight", _SlowStartupPreflight)
    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["rtsp"] if plugin_type == setup_service.PluginType.SOURCE else [],
    )
    request = SetupTestConnectionRequest(type="camera", backend="rtsp", config={})

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: Service returns timeout failure response
    assert response.success is False
    assert "timed out" in response.message


@pytest.mark.asyncio
async def test_test_connection_camera_rtsp_preflight_error_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camera rtsp test should include stage/camera details on preflight error outcome."""

    # Given: An RTSP preflight run that returns a PreflightError result value
    def _fake_validate_plugin(
        plugin_type: object,
        backend: str,
        config: object,
        **runtime_context: object,
    ) -> object:
        _ = (plugin_type, backend, config, runtime_context)
        return RTSPSourceConfig(rtsp_url="rtsp://example.local/stream")

    class _FailingStartupPreflight:
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
            return PreflightError(
                camera_key="front-door",
                stage="selection",
                message="No compatible streams found",
            )

    monkeypatch.setattr(rtsp_setup_probe, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(rtsp_setup_probe, "RTSPStartupPreflight", _FailingStartupPreflight)
    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["rtsp"] if plugin_type == setup_service.PluginType.SOURCE else [],
    )
    request = SetupTestConnectionRequest(type="camera", backend="rtsp", config={})

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: Response captures preflight failure details for UI troubleshooting
    assert response.success is False
    assert response.details == {"stage": "selection", "camera_key": "front-door"}


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
        return FtpSourceConfig(host="127.0.0.1", port=0)

    monkeypatch.setattr(ftp_setup_probe, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(ftp_setup_probe, "probe_tcp_bind", lambda *_args: 42424)
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
async def test_test_connection_camera_ftp_timeout_returns_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camera ftp test should fail when bind probe exceeds timeout."""

    # Given: An FTP bind probe that blocks longer than timeout budget
    def _fake_validate_plugin(
        plugin_type: object,
        backend: str,
        config: object,
        **runtime_context: object,
    ) -> object:
        _ = (plugin_type, backend, config, runtime_context)
        return FtpSourceConfig(host="127.0.0.1", port=0)

    def _slow_probe(*_args: object, **_kwargs: object) -> int:
        time.sleep(0.05)
        return 42424

    monkeypatch.setattr(ftp_setup_probe, "FTP_TEST_CONNECTION_TIMEOUT_S", 0.01)
    monkeypatch.setattr(ftp_setup_probe, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(ftp_setup_probe, "probe_tcp_bind", _slow_probe)
    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["ftp"] if plugin_type == setup_service.PluginType.SOURCE else [],
    )
    request = SetupTestConnectionRequest(type="camera", backend="ftp", config={})

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: Service returns timeout failure response
    assert response.success is False
    assert "timed out" in response.message


@pytest.mark.asyncio
async def test_test_connection_camera_ftp_bind_oserror_returns_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camera ftp test should fail when bind probe raises OSError."""

    # Given: An FTP bind probe that fails with socket-level OSError
    def _fake_validate_plugin(
        plugin_type: object,
        backend: str,
        config: object,
        **runtime_context: object,
    ) -> object:
        _ = (plugin_type, backend, config, runtime_context)
        return FtpSourceConfig(host="127.0.0.1", port=2121)

    def _failing_probe(*_args: object, **_kwargs: object) -> int:
        raise OSError("Address already in use")

    monkeypatch.setattr(ftp_setup_probe, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(ftp_setup_probe, "probe_tcp_bind", _failing_probe)
    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["ftp"] if plugin_type == setup_service.PluginType.SOURCE else [],
    )
    request = SetupTestConnectionRequest(type="camera", backend="ftp", config={})

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: Service reports bind-probe failure without raising
    assert response.success is False
    assert "bind probe failed" in response.message


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

    monkeypatch.setattr(onvif_setup_probe, "OnvifService", _FakeOnvifService)
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
async def test_test_connection_camera_onvif_uses_requested_timeout_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camera onvif test should not inherit the shorter generic plugin timeout."""

    # Given: An ONVIF probe that outlives the generic plugin timeout but stays within request budget
    class _SlowSuccessfulOnvifService:
        def __init__(self, *, discover_fn: object, client_factory: object) -> None:
            _ = (discover_fn, client_factory)

        async def probe(self, options: OnvifProbeOptions) -> object:
            assert options.timeout_s == 0.05
            await asyncio.sleep(0.02)
            return SimpleNamespace(profiles=[object()], streams=[object()])

    monkeypatch.setattr(setup_service, "_PLUGIN_TEST_CONNECTION_TIMEOUT_S", 0.01)
    monkeypatch.setattr(onvif_setup_probe, "OnvifService", _SlowSuccessfulOnvifService)
    request = SetupTestConnectionRequest(
        type="camera",
        backend="onvif",
        config={
            "host": "192.168.1.10",
            "username": "admin",
            "password": "secret",
            "timeout_s": 0.05,
        },
    )

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: The setup-specific ONVIF timeout governs the registered probe
    assert response.success is True
    assert response.message == "ONVIF probe succeeded."


@pytest.mark.asyncio
async def test_test_connection_camera_onvif_validation_failure() -> None:
    """Camera onvif test should fail when required config fields are missing."""
    # Given: An ONVIF request missing required host/username/password fields
    request = SetupTestConnectionRequest(type="camera", backend="onvif", config={})

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: Service returns a validation failure response
    assert response.success is False
    assert "Configuration validation failed" in response.message


@pytest.mark.asyncio
async def test_test_connection_camera_onvif_timeout_returns_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camera onvif test should map timeout errors to non-successful probe responses."""

    # Given: An ONVIF service whose probe raises a timeout-specific service error
    class _TimeoutOnvifService:
        def __init__(self, *, discover_fn: object, client_factory: object) -> None:
            _ = (discover_fn, client_factory)

        async def probe(self, options: object) -> object:
            _ = options
            raise OnvifProbeTimeoutError(1.0, cause=TimeoutError())

    monkeypatch.setattr(onvif_setup_probe, "OnvifService", _TimeoutOnvifService)
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

    # Then: Service returns timeout failure response
    assert response.success is False
    assert "timed out" in response.message


@pytest.mark.asyncio
async def test_test_connection_camera_onvif_probe_error_returns_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camera onvif test should map probe errors to non-successful probe responses."""

    # Given: An ONVIF service whose probe raises a non-timeout service error
    class _FailingOnvifService:
        def __init__(self, *, discover_fn: object, client_factory: object) -> None:
            _ = (discover_fn, client_factory)

        async def probe(self, options: object) -> object:
            _ = options
            raise OnvifProbeError("Authentication failed", cause=RuntimeError())

    monkeypatch.setattr(onvif_setup_probe, "OnvifService", _FailingOnvifService)
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

    # Then: Service returns probe failure response with ONVIF-specific context
    assert response.success is False
    assert "ONVIF probe failed" in response.message


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
        return LocalStorageConfig(root=str(root))

    monkeypatch.setattr(local_storage_setup_probe, "validate_plugin", _fake_validate_plugin)
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
async def test_test_connection_storage_local_rejects_file_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Storage local test should fail when configured root points to a file."""
    # Given: A local storage config rooted at an existing regular file
    file_root = tmp_path / "not-a-directory.txt"
    file_root.write_text("x")

    def _fake_validate_plugin(
        plugin_type: object,
        backend: str,
        config: object,
        **runtime_context: object,
    ) -> object:
        _ = (plugin_type, backend, config, runtime_context)
        return LocalStorageConfig(root=str(file_root))

    monkeypatch.setattr(local_storage_setup_probe, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["local"] if plugin_type == setup_service.PluginType.STORAGE else [],
    )
    request = SetupTestConnectionRequest(type="storage", backend="local", config={})

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: Service reports root path must be a directory
    assert response.success is False
    assert "not a directory" in response.message


@pytest.mark.asyncio
async def test_test_connection_storage_local_rejects_unwritable_existing_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Storage local test should fail when existing root lacks rwx access."""
    # Given: A local storage config rooted at an existing directory with denied access
    root = tmp_path / "clips-denied"
    root.mkdir(parents=True)

    def _fake_validate_plugin(
        plugin_type: object,
        backend: str,
        config: object,
        **runtime_context: object,
    ) -> object:
        _ = (plugin_type, backend, config, runtime_context)
        return LocalStorageConfig(root=str(root))

    original_access = local_storage_setup_probe.os.access

    def _fake_access(path: object, mode: int) -> bool:
        if Path(str(path)) == root and mode == (
            local_storage_setup_probe.os.R_OK
            | local_storage_setup_probe.os.W_OK
            | local_storage_setup_probe.os.X_OK
        ):
            return False
        return original_access(path, mode)

    monkeypatch.setattr(local_storage_setup_probe, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(local_storage_setup_probe.os, "access", _fake_access)
    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["local"] if plugin_type == setup_service.PluginType.STORAGE else [],
    )
    request = SetupTestConnectionRequest(type="storage", backend="local", config={})

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: Service reports root access constraints as a failed probe result
    assert response.success is False
    assert "not readable/writable" in response.message


@pytest.mark.asyncio
async def test_test_connection_storage_local_rejects_unwritable_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Storage local test should fail when missing root's nearest parent is not writable."""
    # Given: A local storage config rooted under an existing but unwritable parent
    parent = tmp_path / "parent"
    parent.mkdir(parents=True)
    missing_root = parent / "child" / "clips"

    def _fake_validate_plugin(
        plugin_type: object,
        backend: str,
        config: object,
        **runtime_context: object,
    ) -> object:
        _ = (plugin_type, backend, config, runtime_context)
        return LocalStorageConfig(root=str(missing_root))

    original_access = local_storage_setup_probe.os.access

    def _fake_access(path: object, mode: int) -> bool:
        if Path(str(path)) == parent and mode == (
            local_storage_setup_probe.os.W_OK | local_storage_setup_probe.os.X_OK
        ):
            return False
        return original_access(path, mode)

    monkeypatch.setattr(local_storage_setup_probe, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(local_storage_setup_probe.os, "access", _fake_access)
    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["local"] if plugin_type == setup_service.PluginType.STORAGE else [],
    )
    request = SetupTestConnectionRequest(type="storage", backend="local", config={})

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: Service reports missing root cannot be created under unwritable parent
    assert response.success is False
    assert "parent is not writable" in response.message


@pytest.mark.asyncio
async def test_test_connection_storage_local_rejects_missing_parent_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Storage local test should fail when no existing parent directory is found."""

    # Given: A local storage config with helper returning no existing parent
    def _fake_validate_plugin(
        plugin_type: object,
        backend: str,
        config: object,
        **runtime_context: object,
    ) -> object:
        _ = (plugin_type, backend, config, runtime_context)
        return LocalStorageConfig(root="/nonexistent/root/dir")

    monkeypatch.setattr(local_storage_setup_probe, "validate_plugin", _fake_validate_plugin)
    monkeypatch.setattr(local_storage_setup_probe, "nearest_existing_parent", lambda _path: None)
    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["local"] if plugin_type == setup_service.PluginType.STORAGE else [],
    )
    request = SetupTestConnectionRequest(type="storage", backend="local", config={})

    # When: Running the setup test-connection service
    response = await setup_service.test_connection(request, _StubApp())

    # Then: Service reports missing parent chain as a failed probe result
    assert response.success is False
    assert "No existing parent directory" in response.message


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
async def test_test_connection_analyzer_validation_failure_returns_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analyzer test should map plugin-validation errors to non-successful responses."""

    # Given: A registered analyzer backend whose config validation raises ValidationError
    class _RequiredConfig(BaseModel):
        required: str

    monkeypatch.setattr(
        setup_service,
        "get_plugin_names",
        lambda plugin_type: ["openai"] if plugin_type == setup_service.PluginType.ANALYZER else [],
    )
    monkeypatch.setattr(
        setup_service,
        "validate_plugin",
        lambda *_args, **_kwargs: _RequiredConfig.model_validate({}),
    )
    request = SetupTestConnectionRequest(type="analyzer", backend="openai", config={})

    # When: Running analyzer test-connection
    response = await setup_service.test_connection(request, _StubApp())

    # Then: Response returns canonical validation failure details
    assert response.success is False
    assert "Configuration validation failed" in response.message


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


@pytest.mark.asyncio
async def test_test_connection_camera_unknown_backend_includes_onvif_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown camera backend should report available backends including ONVIF support."""
    # Given: Source plugin names that omit ONVIF because it is a setup-only camera backend
    monkeypatch.setattr(
        setup_service, "get_plugin_names", lambda *_args, **_kwargs: ["rtsp", "ftp"]
    )
    request = SetupTestConnectionRequest(type="camera", backend="unknown", config={})

    # When / Then: Dispatch raises typed request error with ONVIF included in hints
    with pytest.raises(setup_service.SetupTestConnectionRequestError) as exc_info:
        await setup_service.test_connection(request, _StubApp())

    assert "Unknown camera backend" in str(exc_info.value)
    assert exc_info.value.available_backends == ["ftp", "local_folder", "onvif", "rtsp"]


@pytest.mark.asyncio
async def test_test_connection_serializes_concurrent_calls_per_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setup test-connection should process only one probe at a time per app instance."""
    # Given: A shared app instance and a probe handler that tracks active concurrency
    app = _StubApp()
    request = SetupTestConnectionRequest(type="storage", backend="local", config={})
    active_calls = 0
    max_active_calls = 0

    async def _fake_test_storage_connection(
        *,
        backend: str,
        config: dict[str, object],
    ) -> setup_service.TestConnectionResponse:
        _ = (backend, config)
        nonlocal active_calls
        nonlocal max_active_calls
        active_calls += 1
        max_active_calls = max(max_active_calls, active_calls)
        await asyncio.sleep(0.02)
        active_calls -= 1
        return setup_service.TestConnectionResponse(
            success=True,
            message="ok",
        )

    monkeypatch.setattr(setup_service, "_test_storage_connection", _fake_test_storage_connection)

    # When: Running two setup test-connection requests concurrently on the same app
    await asyncio.gather(
        setup_service.test_connection(request, app),
        setup_service.test_connection(request, app),
    )

    # Then: The underlying probe handler never executes concurrently
    assert max_active_calls == 1
