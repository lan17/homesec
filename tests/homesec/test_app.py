"""Tests for application wiring and startup validation."""

from __future__ import annotations

import pytest

from homesec.app import Application
from homesec.models.config import (
    AlertPolicyConfig,
    CameraConfig,
    CameraSourceConfig,
    Config,
    DropboxStorageConfig,
    MQTTConfig,
    NotifierConfig,
    SendGridEmailConfig,
    StateStoreConfig,
    StorageConfig,
)
from homesec.models.filter import FilterConfig, YoloFilterSettings
from homesec.models.source import LocalFolderSourceConfig
from homesec.models.storage import StorageUploadResult
from homesec.models.vlm import OpenAILLMConfig, VLMConfig
from homesec.plugins.notifiers import NOTIFIER_REGISTRY, MultiplexNotifier, NotifierPlugin


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


def _make_config(notifiers: list[object]) -> Config:
    return Config(
        cameras=[
            CameraConfig(
                name="front_door",
                source=CameraSourceConfig(
                    type="local_folder",
                    config=LocalFolderSourceConfig(
                        watch_dir="recordings",
                        poll_interval=1.0,
                    ),
                ),
            )
        ],
        storage=StorageConfig(
            backend="dropbox",
            dropbox=DropboxStorageConfig(root="/homecam"),
        ),
        state_store=StateStoreConfig(dsn="postgresql://user:pass@localhost/db"),
        notifiers=notifiers,  # type: ignore[arg-type]
        filter=FilterConfig(
            plugin="yolo",
            config=YoloFilterSettings(model_path="yolov8n.pt"),
        ),
        vlm=VLMConfig(
            backend="openai",
            llm=OpenAILLMConfig(api_key_env="OPENAI_API_KEY", model="gpt-4o"),
            trigger_classes=["person"],
        ),
        alert_policy=AlertPolicyConfig(backend="default", config={}),
    )


@pytest.mark.asyncio
async def test_application_wires_pipeline_and_health(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr("homesec.app.create_storage", lambda cfg: _StubStorage(cfg))
    monkeypatch.setattr("homesec.app.PostgresStateStore", _StubStateStore)
    monkeypatch.setattr("homesec.plugins.discover_all_plugins", lambda: None)
    NOTIFIER_REGISTRY.clear()
    NOTIFIER_REGISTRY["mqtt"] = NotifierPlugin(
        name="mqtt",
        config_model=MQTTConfig,
        factory=lambda cfg: _StubNotifier(cfg),
    )
    monkeypatch.setattr("homesec.app.load_filter_plugin", lambda _: _StubFilter())
    monkeypatch.setattr("homesec.app.load_vlm_plugin", lambda _: _StubVLM())

    # When creating components
    await app._create_components()

    # Then pipeline, sources, and health server are wired
    assert app._pipeline is not None
    assert app._health_server is not None
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
    monkeypatch: pytest.MonkeyPatch,
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

    monkeypatch.setattr("homesec.app.create_storage", lambda cfg: _StubStorage(cfg))
    monkeypatch.setattr("homesec.app.PostgresStateStore", _StubStateStore)
    monkeypatch.setattr("homesec.plugins.discover_all_plugins", lambda: None)
    NOTIFIER_REGISTRY.clear()
    NOTIFIER_REGISTRY["mqtt"] = NotifierPlugin(
        name="mqtt",
        config_model=MQTTConfig,
        factory=lambda cfg: _StubNotifier(cfg),
    )
    NOTIFIER_REGISTRY["sendgrid_email"] = NotifierPlugin(
        name="sendgrid_email",
        config_model=SendGridEmailConfig,
        factory=lambda cfg: _StubNotifier(cfg),
    )
    monkeypatch.setattr("homesec.app.load_filter_plugin", lambda _: _StubFilter())
    monkeypatch.setattr("homesec.app.load_vlm_plugin", lambda _: _StubVLM())

    # When creating components
    await app._create_components()

    # Then a MultiplexNotifier is used
    assert isinstance(app._notifier, MultiplexNotifier)
