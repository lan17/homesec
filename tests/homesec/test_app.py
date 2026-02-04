"""Tests for application wiring and startup validation."""

from __future__ import annotations

import pytest

from homesec.app import Application
from homesec.models.config import (
    AlertPolicyConfig,
    CameraConfig,
    CameraSourceConfig,
    Config,
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

    def last_heartbeat(self) -> float:
        return 0.0

    async def ping(self) -> bool:
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

    # Mock specific loads used in app.py
    monkeypatch.setattr("homesec.app.load_filter", lambda _: _StubFilter())
    monkeypatch.setattr("homesec.app.load_analyzer", lambda _: _StubVLM())
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
async def test_application_wires_pipeline_and_sources(_mock_plugins: None) -> None:
    """Application should wire pipeline and sources."""
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

    # Then pipeline and sources are wired
    assert app._pipeline is not None
    assert app._sources
    for source in app._sources:
        callback = source._callback
        assert callback is not None
        assert getattr(callback, "__self__", None) is app._pipeline
        assert getattr(callback, "__func__", None) is app._pipeline.on_new_clip.__func__


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
