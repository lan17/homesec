"""Behavioral tests for RuntimeManager."""

from __future__ import annotations

import asyncio
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
from homesec.models.filter import FilterConfig, FilterResult
from homesec.models.vlm import AnalysisResult, VLMConfig
from homesec.notifiers.multiplex import NotifierEntry
from homesec.pipeline import ClipPipeline
from homesec.plugins.analyzers.openai import OpenAIConfig
from homesec.plugins.filters.yolo import YoloFilterConfig
from homesec.plugins.storage.dropbox import DropboxStorageConfig
from homesec.runtime.controller import RuntimeController
from homesec.runtime.manager import RuntimeManager
from homesec.runtime.models import RuntimeBundle, RuntimeState, config_signature
from homesec.sources.local_folder import LocalFolderSourceConfig


class _StubNotifier:
    async def send(self, alert: object) -> None:
        _ = alert

    async def ping(self) -> bool:
        return True

    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout


class _StubFilter:
    async def detect(self, video_path: Path, overrides: object | None = None) -> FilterResult:
        _ = video_path
        _ = overrides
        return FilterResult(
            detected_classes=[],
            confidence=0.0,
            model="stub",
            sampled_frames=0,
        )

    async def ping(self) -> bool:
        return True

    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout


class _StubVLM:
    async def analyze(
        self, video_path: Path, filter_result: object, config: object
    ) -> AnalysisResult:
        _ = video_path
        _ = filter_result
        _ = config
        return AnalysisResult(risk_level="low", activity_type="none", summary="none")

    async def ping(self) -> bool:
        return True

    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout


class _StubAlertPolicy:
    def should_notify(
        self, camera_name: str, filter_result: object, analysis: object
    ) -> tuple[bool, str]:
        _ = camera_name
        _ = filter_result
        _ = analysis
        return False, "stub"


class _StubPipeline:
    async def shutdown(self, timeout: float = 30.0) -> None:
        _ = timeout


class _FakeController(RuntimeController):
    def __init__(self) -> None:
        self.build_calls: list[int] = []
        self.start_calls: list[int] = []
        self.shutdown_calls: list[int] = []
        self.shutdown_all_calls = 0
        self.running_generations: set[int] = set()
        self.fail_build_generations: set[int] = set()
        self.fail_start_generations: set[int] = set()
        self.fail_shutdown_generations: set[int] = set()
        self.block_start_generations: set[int] = set()
        self.start_gate: asyncio.Event | None = None

    async def build_candidate(self, config: Config, generation: int) -> RuntimeBundle:
        self.build_calls.append(generation)
        if generation in self.fail_build_generations:
            raise RuntimeError("build failed")
        return _make_runtime_bundle(config=config, generation=generation)

    async def start_runtime(self, runtime: RuntimeBundle) -> None:
        self.start_calls.append(runtime.generation)
        self.running_generations.add(runtime.generation)
        if runtime.generation in self.block_start_generations:
            if self.start_gate is None:
                raise AssertionError("start_gate must be set when blocking start")
            await self.start_gate.wait()
        if runtime.generation in self.fail_start_generations:
            raise RuntimeError("start failed")

    async def shutdown_runtime(self, runtime: RuntimeBundle) -> None:
        self.shutdown_calls.append(runtime.generation)
        self.running_generations.discard(runtime.generation)
        if runtime.generation in self.fail_shutdown_generations:
            raise RuntimeError("shutdown failed")

    async def shutdown_all(self) -> None:
        self.shutdown_all_calls += 1
        self.running_generations.clear()


def _make_config(*, camera_name: str, watch_dir: str) -> Config:
    return Config(
        cameras=[
            CameraConfig(
                name=camera_name,
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


def _make_runtime_bundle(config: Config, generation: int) -> RuntimeBundle:
    notifier = _StubNotifier()
    return RuntimeBundle(
        generation=generation,
        config=config,
        config_signature=config_signature(config),
        notifier=notifier,
        notifier_entries=[NotifierEntry(name="stub", notifier=notifier)],
        filter_plugin=_StubFilter(),
        vlm_plugin=_StubVLM(),
        alert_policy=_StubAlertPolicy(),
        pipeline=cast(ClipPipeline, _StubPipeline()),
        sources=[],
        sources_by_camera={},
    )


@pytest.mark.asyncio
async def test_runtime_manager_happy_path_swaps_runtime() -> None:
    """Reload should atomically swap runtime and drain previous generation."""
    # Given: A manager with an active initial runtime
    controller = _FakeController()
    manager = RuntimeManager(controller)
    config_v1 = _make_config(camera_name="front", watch_dir="/tmp/front-v1")
    config_v2 = _make_config(camera_name="front", watch_dir="/tmp/front-v2")
    await manager.start_initial_runtime(config_v1)

    # When: A runtime reload is requested and awaited
    request = manager.request_reload(config_v2)
    result = await manager.wait_for_reload()

    # Then: Reload succeeds, generation increments, and old runtime is shut down
    assert request.accepted is True
    assert result is not None
    assert result.success is True
    assert result.generation == 2
    assert manager.active_runtime is not None
    assert manager.active_runtime.generation == 2
    assert manager.generation == 2
    assert controller.shutdown_calls == [1]
    status = manager.get_status()
    assert status.state == RuntimeState.IDLE
    assert status.generation == 2
    assert status.last_reload_error is None


@pytest.mark.asyncio
async def test_runtime_manager_start_initial_runtime_logs_warning_on_double_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Second initial-start call should be explicit and idempotent."""
    # Given: A manager with an already-started initial runtime
    controller = _FakeController()
    manager = RuntimeManager(controller)
    config_v1 = _make_config(camera_name="front", watch_dir="/tmp/front-v1")
    first_runtime = await manager.start_initial_runtime(config_v1)

    # When: start_initial_runtime is called a second time
    with caplog.at_level("WARNING", logger="homesec.runtime.manager"):
        second_runtime = await manager.start_initial_runtime(config_v1)

    # Then: Existing runtime is returned and warning is emitted
    assert second_runtime is first_runtime
    assert controller.build_calls == [1]
    assert controller.start_calls == [1]
    assert manager.generation == 1
    assert "Initial runtime already active" in caplog.text


@pytest.mark.asyncio
async def test_runtime_manager_rolls_back_when_candidate_start_fails() -> None:
    """Failed candidate start should preserve active runtime."""
    # Given: A manager with generation 1 active and generation 2 start failure configured
    controller = _FakeController()
    manager = RuntimeManager(controller)
    config_v1 = _make_config(camera_name="front", watch_dir="/tmp/front-v1")
    config_v2 = _make_config(camera_name="front", watch_dir="/tmp/front-v2")
    await manager.start_initial_runtime(config_v1)
    controller.fail_start_generations.add(2)

    # When: Reloading to generation 2
    request = manager.request_reload(config_v2)
    result = await manager.wait_for_reload()

    # Then: Reload fails, candidate is cleaned up, and generation 1 remains active
    assert request.accepted is True
    assert result is not None
    assert result.success is False
    assert result.generation == 1
    assert manager.active_runtime is not None
    assert manager.active_runtime.generation == 1
    assert manager.generation == 1
    assert 2 in controller.shutdown_calls
    assert 1 not in controller.shutdown_calls
    status = manager.get_status()
    assert status.state == RuntimeState.IDLE
    assert status.generation == 1
    assert status.last_reload_error == "start failed"


@pytest.mark.asyncio
async def test_runtime_manager_rejects_concurrent_reload_requests() -> None:
    """Manager should enforce single-flight reload semantics."""
    # Given: A blocked generation-2 startup so reload stays in-progress
    controller = _FakeController()
    controller.block_start_generations.add(2)
    controller.start_gate = asyncio.Event()
    manager = RuntimeManager(controller)
    config_v1 = _make_config(camera_name="front", watch_dir="/tmp/front-v1")
    config_v2 = _make_config(camera_name="front", watch_dir="/tmp/front-v2")
    config_v3 = _make_config(camera_name="front", watch_dir="/tmp/front-v3")
    await manager.start_initial_runtime(config_v1)

    # When: Requesting two reloads before the first one completes
    first = manager.request_reload(config_v2)
    await asyncio.sleep(0)
    second = manager.request_reload(config_v3)
    mid_status = manager.get_status()
    controller.start_gate.set()
    result = await manager.wait_for_reload()

    # Then: The second request is rejected deterministically and state transitions complete
    assert first.accepted is True
    assert second.accepted is False
    assert second.message == "Runtime reload already in progress"
    assert mid_status.state == RuntimeState.RELOADING
    assert mid_status.reload_in_progress is True
    assert result is not None
    assert result.success is True
    assert manager.generation == 2
    final_status = manager.get_status()
    assert final_status.state == RuntimeState.IDLE
    assert final_status.reload_in_progress is False


@pytest.mark.asyncio
async def test_runtime_manager_survives_candidate_cleanup_errors() -> None:
    """Candidate cleanup failures should not corrupt active runtime."""
    # Given: A reload where candidate start fails and candidate shutdown also fails
    controller = _FakeController()
    manager = RuntimeManager(controller)
    config_v1 = _make_config(camera_name="front", watch_dir="/tmp/front-v1")
    config_v2 = _make_config(camera_name="front", watch_dir="/tmp/front-v2")
    await manager.start_initial_runtime(config_v1)
    controller.fail_start_generations.add(2)
    controller.fail_shutdown_generations.add(2)

    # When: Reloading to generation 2
    request = manager.request_reload(config_v2)
    result = await manager.wait_for_reload()

    # Then: Manager keeps generation 1 active and reports failure without raising
    assert request.accepted is True
    assert result is not None
    assert result.success is False
    assert manager.active_runtime is not None
    assert manager.active_runtime.generation == 1
    assert manager.generation == 1
    status = manager.get_status()
    assert status.state == RuntimeState.IDLE
    assert status.last_reload_error == "start failed"


@pytest.mark.asyncio
async def test_runtime_manager_fails_initial_start_when_activation_callback_fails() -> None:
    """Initial runtime should fail and clean up when activation callback raises."""

    def _fail_activation(_: RuntimeBundle) -> None:
        raise RuntimeError("bind failed")

    # Given: Runtime manager with activation callback that always fails
    controller = _FakeController()
    manager = RuntimeManager(controller, on_runtime_activated=_fail_activation)
    config_v1 = _make_config(camera_name="front", watch_dir="/tmp/front-v1")

    # When: Starting initial runtime
    with pytest.raises(RuntimeError, match="Initial runtime activation failed"):
        await manager.start_initial_runtime(config_v1)

    # Then: Runtime remains inactive and candidate has been cleaned up
    assert manager.active_runtime is None
    assert manager.generation == 0
    assert controller.shutdown_calls == [1]
    status = manager.get_status()
    assert status.state == RuntimeState.FAILED
    assert status.last_reload_error == "bind failed"


@pytest.mark.asyncio
async def test_runtime_manager_rollback_when_activation_callback_fails() -> None:
    """Activation callback failures should rollback to the previous runtime."""

    class _FailOnSecondActivation:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, _: RuntimeBundle) -> None:
            self.calls += 1
            if self.calls >= 2:
                raise RuntimeError("bind failed")

    # Given: Activation callback succeeds once (initial), then fails during reload
    activation = _FailOnSecondActivation()
    controller = _FakeController()
    manager = RuntimeManager(controller, on_runtime_activated=activation)
    config_v1 = _make_config(camera_name="front", watch_dir="/tmp/front-v1")
    config_v2 = _make_config(camera_name="front", watch_dir="/tmp/front-v2")
    await manager.start_initial_runtime(config_v1)

    # When: Requesting reload with failing activation callback
    request = manager.request_reload(config_v2)
    result = await manager.wait_for_reload()

    # Then: Reload fails and previous runtime remains active
    assert request.accepted is True
    assert result is not None
    assert result.success is False
    assert result.error == "bind failed"
    assert manager.active_runtime is not None
    assert manager.active_runtime.generation == 1
    assert manager.generation == 1
    assert 2 in controller.shutdown_calls
    status = manager.get_status()
    assert status.state == RuntimeState.IDLE
    assert status.last_reload_error == "bind failed"


@pytest.mark.asyncio
async def test_runtime_manager_shuts_down_candidate_before_rollback_on_activation_failure() -> None:
    """Reload rollback should restore previous runtime only after candidate shutdown is attempted."""

    class _FailOnSecondActivation:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, _: RuntimeBundle) -> None:
            self.calls += 1
            if self.calls >= 2:
                raise RuntimeError("bind failed")

    class _ObservingController(_FakeController):
        def __init__(self) -> None:
            super().__init__()
            self.manager: RuntimeManager | None = None
            self.active_generation_seen_during_candidate_shutdown: int | None = None

        async def shutdown_runtime(self, runtime: RuntimeBundle) -> None:
            if runtime.generation == 2 and self.manager is not None:
                active_runtime = self.manager.active_runtime
                if active_runtime is not None:
                    self.active_generation_seen_during_candidate_shutdown = (
                        active_runtime.generation
                    )
            await super().shutdown_runtime(runtime)

    # Given: Reload activation callback fails and controller observes active runtime during cleanup
    activation = _FailOnSecondActivation()
    controller = _ObservingController()
    manager = RuntimeManager(controller, on_runtime_activated=activation)
    controller.manager = manager
    config_v1 = _make_config(camera_name="front", watch_dir="/tmp/front-v1")
    config_v2 = _make_config(camera_name="front", watch_dir="/tmp/front-v2")
    await manager.start_initial_runtime(config_v1)

    # When: Requesting reload that fails during activation callback
    request = manager.request_reload(config_v2)
    result = await manager.wait_for_reload()

    # Then: Candidate shutdown sees generation 2 active before rollback restores generation 1
    assert request.accepted is True
    assert result is not None
    assert result.success is False
    assert controller.active_generation_seen_during_candidate_shutdown == 2
    assert manager.active_runtime is not None
    assert manager.active_runtime.generation == 1
    assert manager.generation == 1


@pytest.mark.asyncio
async def test_runtime_manager_shutdown_cancels_stuck_reload_task() -> None:
    """Shutdown should cancel in-flight reloads that exceed graceful wait."""
    # Given: A manager with a reload stuck in start_runtime
    controller = _FakeController()
    controller.block_start_generations.add(2)
    controller.start_gate = asyncio.Event()
    manager = RuntimeManager(
        controller,
        reload_shutdown_grace_s=0.01,
        reload_cancel_wait_s=0.05,
    )
    config_v1 = _make_config(camera_name="front", watch_dir="/tmp/front-v1")
    config_v2 = _make_config(camera_name="front", watch_dir="/tmp/front-v2")
    await manager.start_initial_runtime(config_v1)
    request = manager.request_reload(config_v2)
    assert request.accepted is True
    await asyncio.sleep(0)

    # When: Shutting down while reload is still in-flight
    await manager.shutdown()

    # Then: Shutdown completes and clears active runtime state
    assert manager.active_runtime is None
    assert manager.generation == 0
    assert controller.running_generations == set()
    assert controller.shutdown_all_calls >= 1
    status = manager.get_status()
    assert status.state == RuntimeState.IDLE
    assert status.reload_in_progress is False
