"""Tests for application wiring and startup validation."""

from __future__ import annotations

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
from homesec.notifiers.multiplex import MultiplexNotifier
from homesec.plugins.analyzers.openai import OpenAIConfig
from homesec.plugins.filters.yolo import YoloFilterConfig
from homesec.plugins.storage.dropbox import DropboxStorageConfig
from homesec.runtime.errors import RuntimeReloadConfigError
from homesec.sources.local_folder import LocalFolderSourceConfig


class _StubStorage:
    def __init__(self, config: object) -> None:
        self.config = config
        self.shutdown_called = False

    async def put_file(self, local_path: object, dest_path: str) -> StorageUploadResult:
        return StorageUploadResult(storage_uri=f"mock:{dest_path}", view_url=None)

    async def get_view_url(self, storage_uri: str) -> str | None:
        return None

    async def get(self, storage_uri: str, local_path: object) -> None:
        return None

    async def exists(self, storage_uri: str) -> bool:
        return False

    async def delete(self, storage_uri: str) -> None:
        return None

    async def ping(self) -> bool:
        return True

    async def shutdown(self, timeout: float | None = None) -> None:
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
        self.shutdown_called = True


class _StubNotifier:
    def __init__(self, config: object) -> None:
        self.config = config
        self.shutdown_called = False

    async def send(self, alert: object) -> None:
        return None

    async def ping(self) -> bool:
        return True

    async def shutdown(self, timeout: float | None = None) -> None:
        self.shutdown_called = True


class _StubFilter:
    async def shutdown(self, timeout: float | None = None) -> None:
        return None


class _StubVLM:
    async def shutdown(self, timeout: float | None = None) -> None:
        return None


class _StubAlertPolicy:
    def should_notify(
        self,
        camera_name: str,
        filter_result: object,
        analysis: object,
    ) -> tuple[bool, str]:
        _ = camera_name
        _ = filter_result
        _ = analysis
        return False, "stub"


class _StubSource:
    """Stub source for testing."""

    def __init__(self) -> None:
        self._callback: object = None
        self.shutdown_called = False

    def register_callback(self, callback: object) -> None:
        self._callback = callback

    async def start(self) -> None:
        return None

    def is_healthy(self) -> bool:
        return True

    async def shutdown(self, timeout: float | None = None) -> None:
        self.shutdown_called = True


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
def _mock_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock plugin loading to return stubs."""
    monkeypatch.setattr("homesec.app.load_storage_plugin", lambda cfg: _StubStorage(cfg))
    monkeypatch.setattr("homesec.app.PostgresStateStore", _StubStateStore)
    monkeypatch.setattr("homesec.plugins.discover_all_plugins", lambda: None)

    # Mock specific loads used by runtime assembler
    monkeypatch.setattr("homesec.runtime.assembly.load_filter", lambda _: _StubFilter())
    monkeypatch.setattr("homesec.runtime.assembly.load_analyzer", lambda _: _StubVLM())
    monkeypatch.setattr("homesec.app.load_alert_policy", lambda *args, **kwargs: _StubAlertPolicy())
    monkeypatch.setattr(
        "homesec.app.load_source_plugin", lambda source_backend, config, camera_name: _StubSource()
    )

    # Mock registry validation
    def _mock_validate(*args, **kwargs):
        pass

    monkeypatch.setattr("homesec.app.validate_plugin_names", _mock_validate)
    monkeypatch.setattr("homesec.app.validate_config", _mock_validate)

    # Mock notifier loading loop in app.py manually if needed,
    # but app.py uses load_notifier_plugin.
    monkeypatch.setattr(
        "homesec.app.load_notifier_plugin", lambda backend, config: _StubNotifier(config)
    )


@pytest.mark.asyncio
async def test_application_wires_pipeline_and_health(_mock_plugins: None) -> None:
    """Application should wire pipeline, sources, and health server."""
    # Given a config and stubbed dependencies
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

    # When creating components
    await app._create_components()

    # Then pipeline, sources, and health server are wired
    assert app._pipeline is not None
    assert app._runtime_manager is not None
    assert app._health_server is not None
    assert app._api_server is not None
    assert app._sources
    for source in app._sources:
        callback = source._callback
        assert callback is not None
        assert getattr(callback, "__self__", None) is app._pipeline
        assert getattr(callback, "__func__", None) is app._pipeline.on_new_clip.__func__
    assert app._health_server._state_store is app._state_store
    assert app._health_server._storage is app._storage
    assert app._health_server._notifier is app._notifier


@pytest.mark.asyncio
async def test_application_uses_multiplex_for_multiple_notifiers(
    _mock_plugins: None,
) -> None:
    """Multiple notifiers should use MultiplexNotifier."""
    # Given a config with multiple notifiers and stubbed dependencies
    config = _make_config(
        [
            NotifierConfig(
                backend="mqtt",
                config={"host": "localhost"},
            ),
            NotifierConfig(
                backend="sendgrid_email",
                config={
                    "from_email": "sender@example.com",
                    "to_emails": ["to@example.com"],
                },
            ),
        ]
    )
    app = Application(config_path=__file__)
    app._config = config

    # When creating components
    await app._create_components()

    # Then a MultiplexNotifier is used
    assert isinstance(app._notifier, MultiplexNotifier)


@pytest.mark.asyncio
async def test_application_does_not_create_api_server_when_disabled(_mock_plugins: None) -> None:
    """API server should not be created when disabled in config."""
    # Given a config with API server disabled
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

    # When creating components
    await app._create_components()

    # Then API server is not created
    assert app._api_server is None


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
