"""Behavioral subprocess tests for runtime supervision."""

from __future__ import annotations

import asyncio
import signal
import tempfile
import uuid
from pathlib import Path
from typing import cast

import pytest

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
from homesec.plugins.analyzers.openai import OpenAIConfig
from homesec.plugins.filters.yolo import YoloFilterConfig
from homesec.plugins.storage.dropbox import DropboxStorageConfig
from homesec.runtime.manager import RuntimeManager
from homesec.runtime.subprocess_controller import (
    SubprocessRuntimeController,
    SubprocessRuntimeHandle,
    _WorkerEventProtocol,
)
from homesec.runtime.subprocess_protocol import WorkerEvent, WorkerEventType
from homesec.sources.local_folder import LocalFolderSourceConfig


def _make_config(*, watch_dir: str) -> Config:
    return Config(
        cameras=[
            CameraConfig(
                name="front",
                source=CameraSourceConfig(
                    backend="local_folder",
                    config=LocalFolderSourceConfig(watch_dir=watch_dir, poll_interval=1.0),
                ),
            )
        ],
        storage=StorageConfig(
            backend="dropbox",
            config=DropboxStorageConfig(root="/homecam"),
        ),
        state_store=StateStoreConfig(dsn="postgresql://user:pass@localhost/db"),
        notifiers=[NotifierConfig(backend="mqtt", config={"host": "localhost"})],
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


@pytest.mark.asyncio
@pytest.mark.subprocess
async def test_subprocess_controller_starts_and_stops_test_worker() -> None:
    """Controller should start and stop the dedicated test harness worker."""
    # Given: A subprocess runtime controller and candidate runtime handle
    controller = SubprocessRuntimeController(
        startup_timeout_s=3.0,
        shutdown_timeout_s=1.0,
        kill_timeout_s=1.0,
        worker_heartbeat_interval_s=0.05,
        heartbeat_stale_s=1.0,
        worker_module="homesec.runtime.test_worker_harness",
    )
    runtime = cast(
        SubprocessRuntimeHandle,
        await controller.build_candidate(_make_config(watch_dir="/tmp/a"), 1),
    )

    # When: Starting and then shutting down the runtime worker
    await controller.start_runtime(runtime)
    await controller.shutdown_runtime(runtime)

    # Then: Worker lifecycle is clean and camera status was observed
    assert runtime.worker_pid is not None
    assert runtime.worker_exit_code == 0
    assert runtime.is_running is False
    assert "front" in runtime.camera_statuses
    assert runtime.camera_statuses["front"].healthy is True


@pytest.mark.asyncio
@pytest.mark.subprocess
async def test_subprocess_controller_escalates_to_sigkill_when_worker_ignores_term() -> None:
    """Controller should SIGKILL runtime worker when graceful TERM does not stop it."""
    # Given: A harness worker configured to ignore SIGTERM
    controller = SubprocessRuntimeController(
        startup_timeout_s=3.0,
        shutdown_timeout_s=0.1,
        kill_timeout_s=1.0,
        worker_heartbeat_interval_s=0.05,
        heartbeat_stale_s=1.0,
        worker_module="homesec.runtime.test_worker_harness",
        worker_extra_args=("--harness-scenario", "ignore-term"),
    )
    runtime = cast(
        SubprocessRuntimeHandle,
        await controller.build_candidate(_make_config(watch_dir="/tmp/b"), 1),
    )

    # When: Starting and shutting down the worker
    await controller.start_runtime(runtime)
    await controller.shutdown_runtime(runtime)

    # Then: Shutdown escalates and process exits due to SIGKILL
    assert runtime.worker_exit_code is not None
    assert runtime.worker_exit_code < 0
    assert abs(runtime.worker_exit_code) == signal.SIGKILL
    assert runtime.is_running is False


@pytest.mark.asyncio
@pytest.mark.subprocess
async def test_runtime_manager_rolls_back_when_subprocess_candidate_fails_startup() -> None:
    """Reload should rollback to prior generation when candidate worker fails startup."""
    # Given: Runtime manager with generation 1 active using harness worker
    controller = SubprocessRuntimeController(
        startup_timeout_s=3.0,
        shutdown_timeout_s=1.0,
        kill_timeout_s=1.0,
        worker_heartbeat_interval_s=0.05,
        heartbeat_stale_s=1.0,
        worker_module="homesec.runtime.test_worker_harness",
    )
    manager = RuntimeManager(controller)
    config_v1 = _make_config(watch_dir="/tmp/v1")
    config_v2 = _make_config(watch_dir="/tmp/v2")
    await manager.start_initial_runtime(config_v1)

    try:
        # When: Reloading with a candidate configured to fail startup
        controller.worker_extra_args = ("--harness-scenario", "startup-fail")
        request = manager.request_reload(config_v2)
        result = await manager.wait_for_reload()

        # Then: Reload fails and generation 1 remains active and running
        assert request.accepted is True
        assert result is not None
        assert result.success is False
        assert result.generation == 1
        active = manager.active_runtime
        assert isinstance(active, SubprocessRuntimeHandle)
        assert active.generation == 1
        assert active.is_running is True
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_build_candidate_cleans_temp_dir_when_socket_bind_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_candidate should clean temporary artifacts when IPC endpoint setup fails."""

    class _FailingLoop:
        async def create_datagram_endpoint(self, *args: object, **kwargs: object) -> object:
            _ = (args, kwargs)
            raise RuntimeError("socket bind failed")

    uuid_values = iter(
        [
            uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        ]
    )
    monkeypatch.setattr(
        "homesec.runtime.subprocess_controller.uuid.uuid4",
        lambda: next(uuid_values),
    )
    monkeypatch.setattr(
        "homesec.runtime.subprocess_controller.asyncio.get_running_loop",
        lambda: _FailingLoop(),
    )

    # Given: Controller candidate creation where socket binding fails after temp files are written
    controller = SubprocessRuntimeController()
    expected_temp_dir = Path(tempfile.gettempdir()) / "hsrt-aaaaaaaaaa"

    # When: Building a candidate runtime
    with pytest.raises(RuntimeError, match="socket bind failed"):
        await controller.build_candidate(_make_config(watch_dir="/tmp/fail"), generation=1)

    # Then: Temporary directory and config artifacts are cleaned up
    assert not expected_temp_dir.exists()


def test_worker_event_protocol_drops_mismatched_correlation_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Worker protocol should reject events with a mismatched correlation_id."""
    # Given: Worker protocol configured for generation 1 and correlation 'expected'
    queue: asyncio.Queue[WorkerEvent] = asyncio.Queue()
    protocol = _WorkerEventProtocol(queue, generation=1, correlation_id="expected")
    event = WorkerEvent(
        event=WorkerEventType.HEARTBEAT,
        generation=1,
        correlation_id="unexpected",
        pid=1234,
    )

    # When: Receiving a datagram from a mismatched correlation id
    with caplog.at_level("WARNING"):
        protocol.datagram_received(event.model_dump_json().encode("utf-8"), ("", 0))

    # Then: Event is dropped and warning is emitted
    assert queue.empty()
    assert "Dropped worker event with mismatched correlation id" in caplog.text


def test_signal_process_group_logs_warning_on_getpgid_permission_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Process-group signaling should warn on unexpected getpgid OS errors."""

    def _raise_permission_error(pid: int) -> int:
        _ = pid
        raise PermissionError("permission denied")

    # Given: getpgid unexpectedly fails with permission error
    monkeypatch.setattr("homesec.runtime.subprocess_controller.os.getpgid", _raise_permission_error)

    # When: Signaling the process group
    with caplog.at_level("WARNING"):
        SubprocessRuntimeController._signal_process_group(1234, signal.SIGTERM)

    # Then: Controller logs a warning instead of silently swallowing the error
    assert "Failed to resolve process group" in caplog.text


def test_signal_process_group_logs_warning_on_killpg_permission_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Process-group signaling should warn on unexpected killpg OS errors."""

    def _return_pgid(pid: int) -> int:
        return pid

    def _raise_permission_error(pgid: int, sig: signal.Signals) -> None:
        _ = (pgid, sig)
        raise PermissionError("permission denied")

    # Given: getpgid succeeds but killpg fails with permission error
    monkeypatch.setattr("homesec.runtime.subprocess_controller.os.getpgid", _return_pgid)
    monkeypatch.setattr("homesec.runtime.subprocess_controller.os.killpg", _raise_permission_error)

    # When: Signaling the process group
    with caplog.at_level("WARNING"):
        SubprocessRuntimeController._signal_process_group(1234, signal.SIGTERM)

    # Then: Controller logs a warning for unexpected killpg failure
    assert "Failed to signal process group" in caplog.text
