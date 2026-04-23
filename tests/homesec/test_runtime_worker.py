"""Tests for runtime worker entrypoint behavior."""

from __future__ import annotations

import asyncio
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

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
from homesec.models.enums import VLMRunMode
from homesec.models.filter import FilterConfig
from homesec.models.vlm import VLMConfig
from homesec.runtime.bootstrap import RuntimePersistenceStack
from homesec.runtime.models import PreviewRefusalReason, PreviewState
from homesec.runtime.subprocess_protocol import (
    WorkerCommand,
    WorkerCommandType,
)
from homesec.sources.rtsp.live_publisher import (
    LivePublisherRefusalReason,
    LivePublisherStartRefusal,
    LivePublisherState,
    LivePublisherStatus,
)


class _StubEmitter:
    def send(self, event: object) -> None:
        _ = event

    def close(self) -> None:
        return None


def _make_config(
    *,
    notifiers: list[NotifierConfig],
    run_mode: VLMRunMode = VLMRunMode.TRIGGER_ONLY,
    source_backend: str = "local_folder",
    preview_enabled: bool = False,
    camera_names: list[str] | None = None,
) -> Config:
    cameras = [
        CameraConfig(
            name=name,
            source=CameraSourceConfig(backend=source_backend, config={}),
        )
        for name in (camera_names or ["front"])
    ]
    return Config(
        cameras=cameras,
        storage=StorageConfig(backend="local", config={}),
        state_store=StateStoreConfig(dsn="postgresql://user:pass@localhost/homesec"),
        notifiers=notifiers,
        filter=FilterConfig(backend="yolo", config={}),
        vlm=VLMConfig(backend="openai", run_mode=run_mode, config={}),
        alert_policy=AlertPolicyConfig(backend="default", config={}),
        preview={"enabled": preview_enabled},
    )


def _make_service(
    config: Config,
    *,
    emitter: object | None = None,
) -> worker_module._RuntimeWorkerService:
    return worker_module._RuntimeWorkerService(
        config=config,
        generation=1,
        correlation_id="test-correlation-id",
        heartbeat_interval_s=1.0,
        command_socket_path=Path("/tmp/homesec-worker-test.sock"),
        emitter=cast(Any, emitter or _StubEmitter()),
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

    def _fake_asyncio_run(coro: Any) -> None:
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


@pytest.mark.asyncio
async def test_runtime_worker_run_runtime_skips_analyzer_load_when_run_mode_never(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_mode=never should avoid analyzer plugin loading in worker startup."""
    # Given: Runtime worker with VLM disabled and analyzer loader guarded
    config = _make_config(notifiers=[], run_mode=VLMRunMode.NEVER)
    service = _make_service(config)
    stop_event = asyncio.Event()
    stop_event.set()

    class _StubStorage:
        async def shutdown(self, timeout: float | None = None) -> None:
            _ = timeout

    class _StubStateStore:
        async def shutdown(self, timeout: float | None = None) -> None:
            _ = timeout

    class _StubEventStore:
        async def shutdown(self, timeout: float | None = None) -> None:
            _ = timeout

    class _StubFilter:
        async def shutdown(self, timeout: float | None = None) -> None:
            _ = timeout

    monkeypatch.setattr(worker_module, "discover_all_plugins", lambda: None)

    async def _fake_build_runtime_persistence_stack() -> RuntimePersistenceStack:
        return RuntimePersistenceStack(
            storage=cast(Any, _StubStorage()),
            state_store=cast(Any, _StubStateStore()),
            event_store=cast(Any, _StubEventStore()),
            repository=cast(Any, object()),
        )

    monkeypatch.setattr(
        service, "_build_runtime_persistence_stack", _fake_build_runtime_persistence_stack
    )
    monkeypatch.setattr(service, "_create_alert_policy", lambda _cfg: cast(Any, object()))
    monkeypatch.setattr(service, "_create_sources", lambda _cfg: ([], {}))
    monkeypatch.setattr("homesec.runtime.assembly.load_filter", lambda _cfg: _StubFilter())

    def _fail_if_called(_: object) -> object:
        raise AssertionError("load_analyzer should not be called for run_mode=never")

    monkeypatch.setattr("homesec.runtime.assembly.load_analyzer", _fail_if_called)

    # When: Running runtime worker startup/shutdown cycle
    await service.run_runtime(stop_event)

    # Then: Worker completes lifecycle without invoking analyzer loader
    assert service._runtime_bundle is None


def test_runtime_worker_preview_status_command_reports_runtime_preview_state() -> None:
    """Preview status command should map source preview status into runtime fields."""

    class _PreviewSource:
        def preview_status(self) -> LivePublisherStatus:
            return LivePublisherStatus(
                state=LivePublisherState.DEGRADED,
                viewer_count=None,
                degraded_reason="viewer_count_unavailable",
                last_error=None,
                idle_shutdown_at=42.0,
            )

        def ensure_preview_active(self) -> LivePublisherStatus | LivePublisherStartRefusal:
            raise AssertionError("status test should not call ensure_preview_active")

        def stop_preview(self) -> None:
            raise AssertionError("status test should not call stop_preview")

        def note_preview_viewer_activity(self, viewer_id: str | None = None) -> None:
            raise AssertionError("status test should not call note_preview_viewer_activity")

    # Given: A preview-enabled RTSP camera with a preview-capable source in the runtime bundle
    config = _make_config(notifiers=[], source_backend="rtsp", preview_enabled=True)
    service = _make_service(config)
    service._runtime_bundle = cast(
        Any,
        SimpleNamespace(sources_by_camera={"front": _PreviewSource()}),
    )
    command = WorkerCommand(
        command=WorkerCommandType.PREVIEW_STATUS,
        command_id="cmd-status",
        generation=1,
        correlation_id="test-correlation-id",
        camera_name="front",
    )

    # When: Handling a preview status command
    result = service._handle_command(command)

    # Then: The worker returns runtime preview status aligned with the contract
    assert result.status is not None
    assert result.status.enabled is True
    assert result.status.state == PreviewState.DEGRADED
    assert result.status.degraded_reason == "viewer_count_unavailable"
    assert result.status.idle_shutdown_at == 42.0


def test_runtime_worker_preview_status_collection_degrades_when_source_raises() -> None:
    """Preview status collection should fail closed when a source raises."""

    class _PreviewSource:
        def preview_status(self) -> LivePublisherStatus:
            raise RuntimeError("publisher crashed")

        def ensure_preview_active(self) -> LivePublisherStatus | LivePublisherStartRefusal:
            raise AssertionError("status collection should not call ensure_preview_active")

        def stop_preview(self) -> None:
            raise AssertionError("status collection should not call stop_preview")

        def note_preview_viewer_activity(self, viewer_id: str | None = None) -> None:
            raise AssertionError("status collection should not call note_preview_viewer_activity")

    # Given: A preview-enabled RTSP camera whose source raises during status lookup
    config = _make_config(notifiers=[], source_backend="rtsp", preview_enabled=True)
    service = _make_service(config)
    service._runtime_bundle = cast(
        Any,
        SimpleNamespace(sources_by_camera={"front": _PreviewSource()}),
    )

    # When: Collecting preview status payloads for worker events
    payloads = service._collect_preview_statuses()

    # Then: The worker reports preview error state instead of raising
    assert payloads["front"].enabled is True
    assert payloads["front"].state == PreviewState.ERROR
    assert payloads["front"].last_error == "Preview status unavailable"


def test_runtime_worker_event_payload_stays_under_unix_datagram_limit() -> None:
    """Heartbeat events should stay within the conservative Unix datagram size budget."""

    class _CapturingEmitter:
        def __init__(self) -> None:
            self.payload: object | None = None

        def send(self, event: object) -> None:
            self.payload = event

        def close(self) -> None:
            return None

    # Given: A preview-enabled worker with enough cameras to stress the heartbeat payload
    config = _make_config(
        notifiers=[],
        source_backend="rtsp",
        preview_enabled=True,
        camera_names=[f"camera_{index}" for index in range(8)],
    )
    emitter = _CapturingEmitter()
    service = _make_service(config, emitter=emitter)

    # When: Emitting a heartbeat event
    service._emit_event(worker_module.WorkerEventType.HEARTBEAT)

    # Then: The serialized datagram fits within the 2048-byte macOS AF_UNIX limit
    event = cast(worker_module.WorkerEvent, emitter.payload)
    assert event.previews == {}
    assert len(event.model_dump_json().encode("utf-8")) <= 2048


def test_runtime_worker_preview_ensure_command_returns_machine_readable_refusal() -> None:
    """Preview ensure command should preserve refusal semantics as structured data."""

    class _PreviewSource:
        def preview_status(self) -> LivePublisherStatus:
            return LivePublisherStatus(state=LivePublisherState.IDLE)

        def ensure_preview_active(self) -> LivePublisherStatus | LivePublisherStartRefusal:
            return LivePublisherStartRefusal(
                reason=LivePublisherRefusalReason.RECORDING_PRIORITY,
                message="Recording currently owns the session budget",
            )

        def stop_preview(self) -> None:
            raise AssertionError("ensure test should not call stop_preview")

        def note_preview_viewer_activity(self, viewer_id: str | None = None) -> None:
            raise AssertionError("ensure test should not call note_preview_viewer_activity")

    # Given: A preview-enabled RTSP camera whose source refuses preview startup
    config = _make_config(notifiers=[], source_backend="rtsp", preview_enabled=True)
    service = _make_service(config)
    service._runtime_bundle = cast(
        Any,
        SimpleNamespace(sources_by_camera={"front": _PreviewSource()}),
    )
    command = WorkerCommand(
        command=WorkerCommandType.PREVIEW_ENSURE_ACTIVE,
        command_id="cmd-ensure",
        generation=1,
        correlation_id="test-correlation-id",
        camera_name="front",
    )

    # When: Handling a preview ensure command
    result = service._handle_command(command)

    # Then: The worker returns a machine-readable refusal reason and message
    assert result.refusal is not None
    assert result.refusal.reason == PreviewRefusalReason.RECORDING_PRIORITY
    assert result.refusal.message == "Recording currently owns the session budget"


def test_runtime_worker_preview_ensure_command_degrades_when_source_raises() -> None:
    """Preview ensure command should fail closed when a source raises."""

    class _PreviewSource:
        def preview_status(self) -> LivePublisherStatus:
            return LivePublisherStatus(state=LivePublisherState.IDLE)

        def ensure_preview_active(self) -> LivePublisherStatus | LivePublisherStartRefusal:
            raise RuntimeError("publisher crashed")

        def stop_preview(self) -> None:
            raise AssertionError("ensure test should not call stop_preview")

        def note_preview_viewer_activity(self, viewer_id: str | None = None) -> None:
            raise AssertionError("ensure test should not call note_preview_viewer_activity")

    # Given: A preview-enabled RTSP camera whose source raises on preview activation
    config = _make_config(notifiers=[], source_backend="rtsp", preview_enabled=True)
    service = _make_service(config)
    service._runtime_bundle = cast(
        Any,
        SimpleNamespace(sources_by_camera={"front": _PreviewSource()}),
    )
    command = WorkerCommand(
        command=WorkerCommandType.PREVIEW_ENSURE_ACTIVE,
        command_id="cmd-ensure-error",
        generation=1,
        correlation_id="test-correlation-id",
        camera_name="front",
    )

    # When: Handling a preview ensure command
    result = service._handle_command(command)

    # Then: The worker returns a temporary-unavailable refusal instead of bubbling the error
    assert result.refusal is not None
    assert result.refusal.reason == PreviewRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE
    assert result.refusal.message == "Preview activation failed"


def test_runtime_worker_preview_force_stop_command_returns_stopping_ack() -> None:
    """Preview force-stop command should return the async stopping acknowledgement."""

    class _PreviewSource:
        def __init__(self) -> None:
            self.stop_calls = 0

        def preview_status(self) -> LivePublisherStatus:
            return LivePublisherStatus(state=LivePublisherState.READY)

        def ensure_preview_active(self) -> LivePublisherStatus | LivePublisherStartRefusal:
            return LivePublisherStatus(state=LivePublisherState.READY)

        def stop_preview(self) -> None:
            self.stop_calls += 1

        def note_preview_viewer_activity(self, viewer_id: str | None = None) -> None:
            raise AssertionError("stop test should not call note_preview_viewer_activity")

    # Given: A preview-enabled RTSP camera with a preview-capable source
    config = _make_config(notifiers=[], source_backend="rtsp", preview_enabled=True)
    service = _make_service(config)
    source = _PreviewSource()
    service._runtime_bundle = cast(
        Any,
        SimpleNamespace(sources_by_camera={"front": source}),
    )
    command = WorkerCommand(
        command=WorkerCommandType.PREVIEW_FORCE_STOP,
        command_id="cmd-stop",
        generation=1,
        correlation_id="test-correlation-id",
        camera_name="front",
    )

    # When: Handling a preview force-stop command
    result = service._handle_command(command)

    # Then: The worker requests source shutdown and returns a stopping acknowledgement
    assert result.stop_result is not None
    assert result.stop_result.accepted is True
    assert result.stop_result.state == PreviewState.STOPPING
    assert source.stop_calls == 1


def test_runtime_worker_preview_force_stop_command_rejects_when_source_raises() -> None:
    """Preview force-stop should fail closed when a source raises."""

    class _PreviewSource:
        def preview_status(self) -> LivePublisherStatus:
            return LivePublisherStatus(state=LivePublisherState.READY)

        def ensure_preview_active(self) -> LivePublisherStatus | LivePublisherStartRefusal:
            return LivePublisherStatus(state=LivePublisherState.READY)

        def stop_preview(self) -> None:
            raise RuntimeError("publisher crashed")

        def note_preview_viewer_activity(self, viewer_id: str | None = None) -> None:
            raise AssertionError("stop test should not call note_preview_viewer_activity")

    # Given: A preview-enabled RTSP camera whose source raises while stopping preview
    config = _make_config(notifiers=[], source_backend="rtsp", preview_enabled=True)
    service = _make_service(config)
    service._runtime_bundle = cast(
        Any,
        SimpleNamespace(sources_by_camera={"front": _PreviewSource()}),
    )
    command = WorkerCommand(
        command=WorkerCommandType.PREVIEW_FORCE_STOP,
        command_id="cmd-stop-error",
        generation=1,
        correlation_id="test-correlation-id",
        camera_name="front",
    )

    # When: Handling a preview force-stop command
    result = service._handle_command(command)

    # Then: The worker returns an explicit rejected stop result
    assert result.stop_result is not None
    assert result.stop_result.accepted is False
    assert result.stop_result.state == PreviewState.ERROR
