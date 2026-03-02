"""Tests for runtime worker entrypoint behavior."""

from __future__ import annotations

from argparse import Namespace
from typing import Any, cast

import homesec.runtime.worker as worker_module
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
from homesec.models.vlm import VLMConfig


class _StubEmitter:
    def send(self, event: object) -> None:
        _ = event

    def close(self) -> None:
        return None


def _make_config(*, notifiers: list[NotifierConfig]) -> Config:
    return Config(
        cameras=[
            CameraConfig(
                name="front",
                source=CameraSourceConfig(backend="local_folder", config={}),
            )
        ],
        storage=StorageConfig(backend="local", config={}),
        state_store=StateStoreConfig(dsn="postgresql://user:pass@localhost/homesec"),
        notifiers=notifiers,
        filter=FilterConfig(backend="yolo", config={}),
        vlm=VLMConfig(backend="openai", config={}),
        alert_policy=AlertPolicyConfig(backend="default", config={}),
    )


def _make_service(config: Config) -> worker_module._RuntimeWorkerService:
    return worker_module._RuntimeWorkerService(
        config=config,
        generation=1,
        correlation_id="test-correlation-id",
        heartbeat_interval_s=1.0,
        emitter=cast(Any, _StubEmitter()),
    )


def test_worker_main_uses_shared_logging_configuration(monkeypatch) -> None:
    """Worker main should configure shared logging before running service."""
    # Given: Parsed args and patched runtime runner hooks
    parsed_args = Namespace(generation=7)
    calls: dict[str, object] = {}

    def _fake_parse_args(argv: list[str]) -> Namespace:
        calls["argv"] = list(argv)
        return parsed_args

    def _fake_configure_logging(*, log_level: str, camera_name: str | None = None) -> None:
        calls["log_level"] = log_level
        calls["camera_name"] = camera_name

    def _fake_asyncio_run(coro: object) -> None:
        calls["ran"] = True
        # Avoid un-awaited coroutine warnings in tests.
        coro.close()

    monkeypatch.setattr(worker_module, "_parse_args", _fake_parse_args)
    monkeypatch.setattr(worker_module, "configure_logging", _fake_configure_logging)
    monkeypatch.setattr(worker_module.asyncio, "run", _fake_asyncio_run)

    # When: Running worker main entrypoint
    worker_module.main()

    # Then: Worker uses the same logging pipeline with worker-scoped camera name
    assert calls["log_level"] == "INFO"
    assert calls["camera_name"] == "runtime-worker-g7"
    assert calls["ran"] is True


def test_runtime_worker_create_notifier_returns_noop_when_notifier_list_empty() -> None:
    # Given: Runtime worker config with no notifier entries
    config = _make_config(notifiers=[])
    service = _make_service(config)

    # When: Building notifier stack for runtime bundle
    notifier, entries = service._create_notifier(config)

    # Then: Worker uses a noop notifier and exposes no notifier entries
    assert isinstance(notifier, worker_module._NoopNotifier)
    assert entries == []


def test_runtime_worker_create_notifier_skips_disabled_entries(
    monkeypatch,
) -> None:
    # Given: Runtime worker config with only disabled notifiers
    config = _make_config(
        notifiers=[
            NotifierConfig(backend="mqtt", enabled=False, config={"host": "localhost"}),
            NotifierConfig(backend="sendgrid", enabled=False, config={"api_key_env": "SENDGRID"}),
        ]
    )
    service = _make_service(config)

    def _unexpected_plugin_load(*_: object) -> object:
        raise AssertionError("Disabled notifiers should not be loaded")

    monkeypatch.setattr(worker_module, "load_notifier_plugin", _unexpected_plugin_load)

    # When: Building notifier stack for runtime bundle
    notifier, entries = service._create_notifier(config)

    # Then: Worker keeps notifications disabled and does not load plugins
    assert isinstance(notifier, worker_module._NoopNotifier)
    assert entries == []
