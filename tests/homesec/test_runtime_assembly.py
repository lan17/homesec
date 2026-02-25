"""Behavioral tests for runtime assembly and partial-build cleanup."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

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
from homesec.plugins.analyzers.openai import OpenAIConfig
from homesec.plugins.filters.yolo import YoloFilterConfig
from homesec.plugins.storage.dropbox import DropboxStorageConfig
from homesec.runtime.assembly import RuntimeAssembler
from homesec.runtime.models import RuntimeBundle
from homesec.sources.local_folder import LocalFolderSourceConfig


class _StubStorage:
    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout


class _StubNotifier:
    def __init__(self) -> None:
        self.shutdown_called = False

    async def send(self, alert: object) -> None:
        _ = alert

    async def ping(self) -> bool:
        return True

    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout
        self.shutdown_called = True


class _StubFilter:
    def __init__(self) -> None:
        self.shutdown_called = False

    async def detect(self, video_path: object, overrides: object | None = None) -> FilterResult:
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
        self.shutdown_called = True


class _StubVLM:
    async def analyze(
        self, video_path: object, filter_result: object, config: object
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
    def __init__(self, *, camera_name: str, fail_timeout: bool = False) -> None:
        self.camera_name = camera_name
        self.fail_timeout = fail_timeout
        self.shutdown_called = False

    def register_callback(self, callback: object) -> None:
        _ = callback

    async def start(self) -> None:
        if self.fail_timeout:
            await asyncio.sleep(0.05)
            return

    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout
        self.shutdown_called = True


class _SlowShutdownComponent:
    def __init__(self) -> None:
        self.shutdown_called = False

    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout
        self.shutdown_called = True
        await asyncio.sleep(0.05)


class _StubPipeline:
    def __init__(self, *, raise_on_backstop: bool = False) -> None:
        self.shutdown_called = False
        self.prune_reasons: list[str] = []
        self.raise_on_backstop = raise_on_backstop

    def request_retention_prune(self, *, reason: str) -> None:
        self.prune_reasons.append(reason)
        if self.raise_on_backstop:
            raise RuntimeError("simulated startup backstop failure")

    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout
        self.shutdown_called = True


def _make_config() -> Config:
    return Config(
        cameras=[
            CameraConfig(
                name="front",
                source=CameraSourceConfig(
                    backend="local_folder",
                    config=LocalFolderSourceConfig(watch_dir="/tmp/front", poll_interval=1.0),
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


def _make_assembler(notifier: _StubNotifier) -> RuntimeAssembler:
    return RuntimeAssembler(
        storage=_StubStorage(),
        repository=cast(Any, object()),
        notifier_factory=lambda _config: (
            notifier,
            [NotifierEntry(name="stub", notifier=notifier)],
        ),
        notifier_health_logger=_noop_notifier_health,
        alert_policy_factory=lambda _config: _StubAlertPolicy(),
        source_factory=lambda _config: ([], {}),
    )


async def _noop_notifier_health(entries: list[NotifierEntry]) -> None:
    _ = entries


@pytest.mark.asyncio
async def test_runtime_assembly_cleans_notifier_when_filter_build_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial build should clean notifier if filter plugin creation fails."""
    # Given: Runtime assembly with notifier created and filter loader failing
    notifier = _StubNotifier()
    assembler = _make_assembler(notifier)
    config = _make_config()

    def _raise_filter_error(_: object) -> object:
        raise RuntimeError("filter build failed")

    monkeypatch.setattr("homesec.runtime.assembly.load_filter", _raise_filter_error)

    # When: Building runtime bundle
    with pytest.raises(RuntimeError, match="filter build failed"):
        await assembler.build_bundle(config, generation=1)

    # Then: Notifier is shut down during partial-build cleanup
    assert notifier.shutdown_called is True


@pytest.mark.asyncio
async def test_runtime_assembly_cleans_filter_and_notifier_when_analyzer_build_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial build should clean all created plugins when analyzer creation fails."""
    # Given: Runtime assembly with notifier + filter created and analyzer loader failing
    notifier = _StubNotifier()
    filter_plugin = _StubFilter()
    assembler = _make_assembler(notifier)
    config = _make_config()

    monkeypatch.setattr("homesec.runtime.assembly.load_filter", lambda _cfg: filter_plugin)

    def _raise_analyzer_error(_: object) -> object:
        raise RuntimeError("analyzer build failed")

    monkeypatch.setattr("homesec.runtime.assembly.load_analyzer", _raise_analyzer_error)

    # When: Building runtime bundle
    with pytest.raises(RuntimeError, match="analyzer build failed"):
        await assembler.build_bundle(config, generation=1)

    # Then: Created components are shut down during partial-build cleanup
    assert notifier.shutdown_called is True
    assert filter_plugin.shutdown_called is True


@pytest.mark.asyncio
async def test_runtime_assembly_start_timeout_cleans_started_sources() -> None:
    """Startup timeout should fail fast and clean up already-started sources."""
    # Given: A runtime bundle with one source started and one source timing out
    started_source = _StubSource(camera_name="front")
    timed_out_source = _StubSource(camera_name="garage", fail_timeout=True)
    notifier = _StubNotifier()
    assembler = RuntimeAssembler(
        storage=_StubStorage(),
        repository=cast(Any, object()),
        notifier_factory=lambda _config: (
            notifier,
            [NotifierEntry(name="stub", notifier=notifier)],
        ),
        notifier_health_logger=_noop_notifier_health,
        alert_policy_factory=lambda _config: _StubAlertPolicy(),
        source_factory=lambda _config: ([], {}),
        source_start_timeout_s=0.01,
    )
    runtime = RuntimeBundle(
        generation=1,
        config=_make_config(),
        config_signature="cfgsig",
        notifier=notifier,
        notifier_entries=[],
        filter_plugin=_StubFilter(),
        vlm_plugin=_StubVLM(),
        alert_policy=_StubAlertPolicy(),
        pipeline=cast(Any, _SlowShutdownComponent()),
        sources=[started_source, timed_out_source],
        sources_by_camera={"front": started_source, "garage": timed_out_source},
    )

    # When: Starting runtime bundle preflight
    with pytest.raises(RuntimeError, match="Source startup preflight failed"):
        await assembler.start_bundle(runtime)

    # Then: Already-started sources are cleaned up
    assert started_source.shutdown_called is True


@pytest.mark.asyncio
async def test_runtime_assembly_start_triggers_startup_backstop_prune() -> None:
    """Successful startup should trigger one startup backstop prune request."""
    # Given: A runtime bundle with healthy startup source and pipeline
    source = _StubSource(camera_name="front")
    pipeline = _StubPipeline()
    notifier = _StubNotifier()
    assembler = RuntimeAssembler(
        storage=_StubStorage(),
        repository=cast(Any, object()),
        notifier_factory=lambda _config: (
            notifier,
            [NotifierEntry(name="stub", notifier=notifier)],
        ),
        notifier_health_logger=_noop_notifier_health,
        alert_policy_factory=lambda _config: _StubAlertPolicy(),
        source_factory=lambda _config: ([source], {"front": source}),
    )
    runtime = RuntimeBundle(
        generation=1,
        config=_make_config(),
        config_signature="cfgsig",
        notifier=notifier,
        notifier_entries=[],
        filter_plugin=_StubFilter(),
        vlm_plugin=_StubVLM(),
        alert_policy=_StubAlertPolicy(),
        pipeline=cast(Any, pipeline),
        sources=[source],
        sources_by_camera={"front": source},
    )

    # When: Starting runtime bundle preflight
    await assembler.start_bundle(runtime)

    # Then: Startup backstop prune is requested once
    assert pipeline.prune_reasons == ["startup_backstop"]


@pytest.mark.asyncio
async def test_runtime_assembly_startup_backstop_failure_is_non_fatal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Startup backstop trigger errors should be logged but not fail startup."""
    # Given: A runtime bundle whose pipeline raises on startup backstop request
    source = _StubSource(camera_name="front")
    pipeline = _StubPipeline(raise_on_backstop=True)
    notifier = _StubNotifier()
    assembler = RuntimeAssembler(
        storage=_StubStorage(),
        repository=cast(Any, object()),
        notifier_factory=lambda _config: (
            notifier,
            [NotifierEntry(name="stub", notifier=notifier)],
        ),
        notifier_health_logger=_noop_notifier_health,
        alert_policy_factory=lambda _config: _StubAlertPolicy(),
        source_factory=lambda _config: ([source], {"front": source}),
    )
    runtime = RuntimeBundle(
        generation=1,
        config=_make_config(),
        config_signature="cfgsig",
        notifier=notifier,
        notifier_entries=[],
        filter_plugin=_StubFilter(),
        vlm_plugin=_StubVLM(),
        alert_policy=_StubAlertPolicy(),
        pipeline=cast(Any, pipeline),
        sources=[source],
        sources_by_camera={"front": source},
    )
    caplog.set_level(logging.ERROR)

    # When: Starting runtime bundle preflight
    await assembler.start_bundle(runtime)

    # Then: Startup remains successful and failure is logged prominently
    assert pipeline.prune_reasons == ["startup_backstop"]
    assert "Startup backstop retention trigger failed" in caplog.text


@pytest.mark.asyncio
async def test_runtime_assembly_component_shutdown_timeout_is_bounded() -> None:
    """Component shutdown should remain bounded when a plugin hangs."""
    # Given: A runtime bundle with a hanging component shutdown
    slow_component = _SlowShutdownComponent()
    notifier = _StubNotifier()
    assembler = RuntimeAssembler(
        storage=_StubStorage(),
        repository=cast(Any, object()),
        notifier_factory=lambda _config: (
            notifier,
            [NotifierEntry(name="stub", notifier=notifier)],
        ),
        notifier_health_logger=_noop_notifier_health,
        alert_policy_factory=lambda _config: _StubAlertPolicy(),
        source_factory=lambda _config: ([], {}),
        component_shutdown_timeout_s=0.01,
    )
    runtime = RuntimeBundle(
        generation=1,
        config=_make_config(),
        config_signature="cfgsig",
        notifier=notifier,
        notifier_entries=[],
        filter_plugin=slow_component,
        vlm_plugin=_StubVLM(),
        alert_policy=_StubAlertPolicy(),
        pipeline=cast(Any, _SlowShutdownComponent()),
        sources=[],
        sources_by_camera={},
    )

    # When: Shutting down the runtime bundle
    await asyncio.wait_for(assembler.shutdown_bundle(runtime), timeout=0.2)

    # Then: Shutdown returns without hanging
    assert slow_component.shutdown_called is True
