"""Behavioral subprocess tests for runtime supervision."""

from __future__ import annotations

import signal
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
)
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
