"""Tests for application wiring and startup validation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from homesec.app import Application
from homesec.config.loader import ConfigError, ConfigErrorCode
from homesec.models.config import (
    AlertPolicyConfig,
    CameraConfig,
    CameraSourceConfig,
    Config,
    FastAPIServerConfig,
    NotifierConfig,
    StateStoreConfig,
    StorageConfig,
)
from homesec.models.filter import FilterConfig
from homesec.models.storage import StorageUploadResult
from homesec.models.vlm import VLMConfig
from homesec.plugins.analyzers.openai import OpenAIConfig
from homesec.plugins.filters.yolo import YoloFilterConfig
from homesec.plugins.storage.dropbox import DropboxStorageConfig
from homesec.runtime.controller import RuntimeController
from homesec.runtime.errors import RuntimeReloadConfigError
from homesec.runtime.models import (
    RuntimeCameraStatus,
    RuntimeReloadRequest,
    RuntimeState,
    RuntimeStatusSnapshot,
    config_signature,
)
from homesec.runtime.subprocess_controller import SubprocessRuntimeHandle
from homesec.sources.local_folder import LocalFolderSourceConfig


class _StubStorage:
    def __init__(self, config: object) -> None:
        self.config = config
        self.shutdown_called = False

    async def put_file(self, local_path: object, dest_path: str) -> StorageUploadResult:
        return StorageUploadResult(storage_uri=f"mock:{dest_path}", view_url=None)

    async def get_view_url(self, storage_uri: str) -> str | None:
        _ = storage_uri
        return None

    async def get(self, storage_uri: str, local_path: object) -> None:
        _ = storage_uri
        _ = local_path
        return None

    async def exists(self, storage_uri: str) -> bool:
        _ = storage_uri
        return False

    async def delete(self, storage_uri: str) -> None:
        _ = storage_uri
        return None

    async def ping(self) -> bool:
        return True

    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout
        self.shutdown_called = True


class _StubStateStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.shutdown_called = False

    async def initialize(self) -> bool:
        return True

    async def ping(self) -> bool:
        return True

    def create_event_store(self) -> object:
        from homesec.state import NoopEventStore

        return NoopEventStore()

    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout
        self.shutdown_called = True


class _FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None


class _StubRuntimeController(RuntimeController):
    def __init__(self) -> None:
        self._handles: list[SubprocessRuntimeHandle] = []

    async def build_candidate(self, config: Config, generation: int) -> SubprocessRuntimeHandle:
        handle = SubprocessRuntimeHandle(
            generation=generation,
            config=config,
            config_signature=config_signature(config),
            correlation_id=f"test-{generation}",
            temp_dir=Path("/tmp"),
            control_socket_path=Path("/tmp/homesec-test.sock"),
            config_json_path=Path("/tmp/homesec-test.json"),
        )
        self._handles.append(handle)
        return handle

    async def start_runtime(self, runtime: SubprocessRuntimeHandle) -> None:
        runtime.process = cast(
            asyncio.subprocess.Process, _FakeProcess(pid=1000 + runtime.generation)
        )
        runtime.last_heartbeat_at = datetime.now(timezone.utc)
        runtime.camera_statuses = {
            camera.name: RuntimeCameraStatus(healthy=camera.enabled, last_heartbeat=1.0)
            for camera in runtime.config.cameras
        }

    async def shutdown_runtime(self, runtime: SubprocessRuntimeHandle) -> None:
        process = runtime.process
        if process is not None:
            process.returncode = 0
        runtime.process = None

    async def shutdown_all(self) -> None:
        for handle in self._handles:
            await self.shutdown_runtime(handle)


class _StubRuntimeManagerStatus:
    def __init__(
        self,
        *,
        status: RuntimeStatusSnapshot,
        active_runtime: SubprocessRuntimeHandle | None,
    ) -> None:
        self._status = status
        self.active_runtime = active_runtime

    def get_status(self) -> RuntimeStatusSnapshot:
        return self._status


def _make_config(notifiers: list[object]) -> Config:
    return Config(
        cameras=[
            CameraConfig(
                name="front_door",
                source=CameraSourceConfig(
                    backend="local_folder",
                    config=LocalFolderSourceConfig(
                        watch_dir="recordings",
                        poll_interval=1.0,
                    ),
                ),
            )
        ],
        storage=StorageConfig(
            backend="dropbox",
            config=DropboxStorageConfig(root="/homecam"),
        ),
        state_store=StateStoreConfig(dsn="postgresql://user:pass@localhost/db"),
        notifiers=notifiers,  # type: ignore[arg-type]
        filter=FilterConfig(
            backend="yolo",
            config=YoloFilterConfig(model_path="yolov8n.pt"),
        ),
        vlm=VLMConfig(
            backend="openai",
            config=OpenAIConfig(api_key_env="OPENAI_API_KEY", model="gpt-4o"),
            trigger_classes=["person"],
        ),
        alert_policy=AlertPolicyConfig(backend="default", config={}),
    )


@pytest.fixture
def _mock_runtime_environment(monkeypatch: pytest.MonkeyPatch) -> _StubRuntimeController:
    """Mock runtime environment and return controller stub."""
    # Given: Runtime dependencies mocked for deterministic tests
    controller = _StubRuntimeController()
    monkeypatch.setattr("homesec.app.load_storage_plugin", lambda cfg: _StubStorage(cfg))
    monkeypatch.setattr("homesec.app.PostgresStateStore", _StubStateStore)
    monkeypatch.setattr("homesec.plugins.discover_all_plugins", lambda: None)
    monkeypatch.setattr("homesec.app.validate_plugin_names", lambda *args, **kwargs: None)
    monkeypatch.setattr("homesec.app.validate_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        Application,
        "_create_runtime_controller",
        lambda self: controller,
    )
    return controller


@pytest.mark.asyncio
async def test_application_wires_runtime_health_and_api(
    _mock_runtime_environment: _StubRuntimeController,
) -> None:
    """Application should expose runtime camera health via app accessors."""
    # Given: A config and runtime environment with subprocess controller stub
    config = _make_config(
        [
            NotifierConfig(
                backend="mqtt",
                config={"host": "localhost"},
            )
        ]
    )
    app = Application(config_path=__file__)
    app._config = config

    # When: Creating components
    await app._create_components()

    # Then: Runtime manager, API server, and camera health accessors are wired
    assert app._runtime_manager is not None
    assert app._health_server is not None
    assert app._api_server is not None
    assert app.pipeline_running is True
    assert len(app.sources) == 1
    assert app.sources[0].is_healthy() is True
    source = app.get_source("front_door")
    assert source is not None
    assert source.last_heartbeat() == 1.0
    assert app._health_server._state_store is app._state_store
    assert app._health_server._storage is app._storage


@pytest.mark.asyncio
async def test_application_does_not_create_api_server_when_disabled(
    _mock_runtime_environment: _StubRuntimeController,
) -> None:
    """API server should not be created when disabled in config."""
    # Given: A config with API server disabled
    config = _make_config(
        [
            NotifierConfig(
                backend="mqtt",
                config={"host": "localhost"},
            )
        ]
    )
    config.server = FastAPIServerConfig(enabled=False)
    app = Application(config_path=__file__)
    app._config = config

    # When: Creating components
    await app._create_components()

    # Then: API server is not created
    assert app._api_server is None


def test_get_runtime_status_preserves_reloading_when_heartbeat_is_stale() -> None:
    """Runtime status should stay reloading during transitions despite stale heartbeat."""
    # Given: A stale runtime heartbeat while manager state is actively reloading
    config = _make_config([NotifierConfig(backend="mqtt", config={"host": "localhost"})])
    runtime = SubprocessRuntimeHandle(
        generation=2,
        config=config,
        config_signature="cfgsig",
        correlation_id="test-2",
        temp_dir=Path("/tmp"),
        control_socket_path=Path("/tmp/homesec-test.sock"),
        config_json_path=Path("/tmp/homesec-test.json"),
    )
    runtime.process = cast(asyncio.subprocess.Process, _FakeProcess(pid=1002))
    runtime.last_heartbeat_at = datetime.now(timezone.utc) - timedelta(seconds=60)
    runtime.camera_statuses = {
        "front_door": RuntimeCameraStatus(healthy=True, last_heartbeat=1.0),
    }
    status = RuntimeStatusSnapshot(
        state=RuntimeState.RELOADING,
        generation=2,
        reload_in_progress=True,
        active_config_version="cfgsig",
        last_reload_at=None,
        last_reload_error=None,
    )
    app = Application(config_path=__file__)
    app._runtime_manager = cast(
        Any,
        _StubRuntimeManagerStatus(status=status, active_runtime=runtime),
    )

    # When: Reading runtime status from the app facade
    current = app.get_runtime_status()

    # Then: Reloading state is preserved (not projected to failed)
    assert current.state == RuntimeState.RELOADING
    assert current.last_reload_error is None


def test_camera_health_degrades_when_runtime_heartbeat_is_stale() -> None:
    """Camera snapshots should be unhealthy when runtime heartbeat is stale."""
    # Given: A stale runtime heartbeat with cached healthy camera statuses
    config = _make_config([NotifierConfig(backend="mqtt", config={"host": "localhost"})])
    runtime = SubprocessRuntimeHandle(
        generation=2,
        config=config,
        config_signature="cfgsig",
        correlation_id="test-2",
        temp_dir=Path("/tmp"),
        control_socket_path=Path("/tmp/homesec-test.sock"),
        config_json_path=Path("/tmp/homesec-test.json"),
    )
    runtime.process = cast(asyncio.subprocess.Process, _FakeProcess(pid=1002))
    runtime.last_heartbeat_at = datetime.now(timezone.utc) - timedelta(seconds=60)
    runtime.camera_statuses = {
        "front_door": RuntimeCameraStatus(healthy=True, last_heartbeat=10.0),
    }
    app = Application(config_path=__file__)
    app._runtime_manager = cast(
        Any,
        _StubRuntimeManagerStatus(
            status=RuntimeStatusSnapshot(
                state=RuntimeState.IDLE,
                generation=2,
                reload_in_progress=False,
                active_config_version="cfgsig",
                last_reload_at=None,
                last_reload_error=None,
            ),
            active_runtime=runtime,
        ),
    )

    # When: Reading camera health from app accessors
    source = app.get_source("front_door")
    all_sources = app.sources

    # Then: Source health is degraded to unhealthy due to stale runtime heartbeat
    assert source is not None
    assert source.is_healthy() is False
    assert all_sources
    assert all_sources[0].is_healthy() is False


@pytest.mark.asyncio
async def test_application_shutdown_stops_api_before_runtime_manager() -> None:
    """Shutdown should stop API server before runtime manager cleanup."""

    class _StubAPIServer:
        def __init__(self, calls: list[str]) -> None:
            self._calls = calls

        async def stop(self) -> None:
            self._calls.append("api")

    class _StubRuntimeManager:
        def __init__(self, calls: list[str]) -> None:
            self._calls = calls

        async def shutdown(self) -> None:
            self._calls.append("runtime")

    class _StubHealthServer:
        def __init__(self, calls: list[str]) -> None:
            self._calls = calls

        async def stop(self) -> None:
            self._calls.append("health")

    class _StubShutdownable:
        def __init__(self, name: str, calls: list[str]) -> None:
            self._name = name
            self._calls = calls

        async def shutdown(self, timeout: float | None = None) -> None:
            _ = timeout
            self._calls.append(self._name)

    # Given: An application with lifecycle dependencies attached
    calls: list[str] = []
    app = Application(config_path=__file__)
    app._api_server = cast(Any, _StubAPIServer(calls))
    app._runtime_manager = cast(Any, _StubRuntimeManager(calls))
    app._health_server = cast(Any, _StubHealthServer(calls))
    app._state_store = cast(Any, _StubShutdownable("state", calls))
    app._storage = cast(Any, _StubShutdownable("storage", calls))

    # When: Triggering graceful shutdown
    await app.shutdown()

    # Then: API server is stopped before runtime manager shutdown
    assert calls[:2] == ["api", "runtime"]


@pytest.mark.asyncio
async def test_request_runtime_reload_maps_semantic_config_error_to_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reload should classify semantic config failures as 422."""
    # Given: Config loading fails with a semantic validation error code
    app = Application(config_path=__file__)

    def _raise_config_error() -> Config:
        raise ConfigError(
            "Config validation failed",
            code=ConfigErrorCode.VALIDATION_FAILED,
        )

    monkeypatch.setattr(app._config_manager, "get_config", _raise_config_error)

    # When: Requesting runtime reload
    with pytest.raises(RuntimeReloadConfigError) as exc_info:
        await app.request_runtime_reload()

    # Then: API-facing error status/code are mapped from ConfigErrorCode
    assert exc_info.value.status_code == 422
    assert exc_info.value.error_code == ConfigErrorCode.VALIDATION_FAILED.value


@pytest.mark.asyncio
async def test_request_runtime_reload_maps_source_config_error_to_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reload should classify source/input config failures as 400."""
    # Given: Config loading fails because the config file cannot be read
    app = Application(config_path=__file__)

    def _raise_config_error() -> Config:
        raise ConfigError(
            "Config file not found",
            code=ConfigErrorCode.FILE_NOT_FOUND,
        )

    monkeypatch.setattr(app._config_manager, "get_config", _raise_config_error)

    # When: Requesting runtime reload
    with pytest.raises(RuntimeReloadConfigError) as exc_info:
        await app.request_runtime_reload()

    # Then: API-facing error status/code are mapped from ConfigErrorCode
    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == ConfigErrorCode.FILE_NOT_FOUND.value


@pytest.mark.asyncio
async def test_request_system_restart_delegates_to_runtime_reload() -> None:
    """System restart should reuse runtime reload control-plane behavior."""
    # Given: An app with request_runtime_reload mocked
    app = Application(config_path=__file__)
    expected = RuntimeReloadRequest(accepted=True, message="ok", target_generation=2)

    async def _reload() -> RuntimeReloadRequest:
        return expected

    app.request_runtime_reload = _reload  # type: ignore[method-assign]

    # When: Requesting system restart
    result = await app.request_system_restart()

    # Then: It delegates to runtime reload semantics
    assert result is expected
