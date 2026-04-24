"""Integration tests for pipeline event emission."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from homesec.models.clip import Clip
from homesec.models.config import (
    AlertPolicyConfig,
    CameraConfig,
    CameraSourceConfig,
    Config,
    NotifierConfig,
    RetryConfig,
    StateStoreConfig,
    StorageConfig,
)
from homesec.models.enums import VLMSkipReason
from homesec.models.filter import FilterConfig, FilterResult
from homesec.models.vlm import AnalysisResult, VLMConfig
from homesec.notifiers.multiplex import NotifierEntry
from homesec.pipeline import ClipPipeline
from homesec.plugins.alert_policies.default import DefaultAlertPolicy, DefaultAlertPolicySettings
from homesec.plugins.analyzers.openai import OpenAIConfig
from homesec.plugins.filters.yolo import YoloFilterConfig
from homesec.plugins.storage.dropbox import DropboxStorageConfig
from homesec.repository import ClipRepository
from homesec.state.postgres import (
    PostgresEventStore,
    PostgresStateStore,
    create_event_store_for_postgres_state_store,
)
from tests.homesec.mocks import (
    MockFilter,
    MockNotifier,
    MockRetentionPruner,
    MockStorage,
    MockVLM,
)


def build_config(
    *,
    notify_on_motion: bool = False,
    notifier_count: int = 1,
    run_mode: str = "trigger_only",
) -> Config:
    """Build a minimal config for pipeline tests."""
    cameras = [
        CameraConfig(
            name="front_door",
            source=CameraSourceConfig(
                backend="local_folder",
                config={"watch_dir": "recordings", "poll_interval": 1.0},
            ),
        )
    ]
    return Config(
        cameras=cameras,
        storage=StorageConfig(
            backend="dropbox",
            config=DropboxStorageConfig(root="/homecam"),
        ),
        state_store=StateStoreConfig(dsn="postgresql://user:pass@localhost/db"),
        notifiers=[
            NotifierConfig(
                backend="mqtt",
                config={"host": "localhost"},
            )
            for _ in range(notifier_count)
        ],
        filter=FilterConfig(
            backend="yolo",
            config=YoloFilterConfig(model_path="yolov8n.pt"),
        ),
        vlm=VLMConfig(
            backend="openai",
            run_mode=run_mode,
            trigger_classes=["person", "car"],
            config=OpenAIConfig(api_key_env="OPENAI_API_KEY", model="gpt-4o"),
        ),
        alert_policy=AlertPolicyConfig(
            backend="default",
            config={
                "min_risk_level": "low",
                "notify_on_motion": notify_on_motion,
            },
        ),
        retry=RetryConfig(max_attempts=1, backoff_s=0.0),
    )


def make_alert_policy(config: Config) -> DefaultAlertPolicy:
    settings = DefaultAlertPolicySettings.model_validate(config.alert_policy.config)
    settings.trigger_classes = list(config.vlm.trigger_classes)
    return DefaultAlertPolicy(settings)


def make_clip(tmp_path: Path, clip_id: str) -> Clip:
    """Create a sample clip for tests."""
    video_path = tmp_path / f"{clip_id}.mp4"
    video_path.write_bytes(b"fake video content")
    start_ts = datetime.now()
    return Clip(
        clip_id=clip_id,
        camera_name="front_door",
        local_path=video_path,
        start_ts=start_ts,
        end_ts=start_ts + timedelta(seconds=10),
        duration_s=10.0,
        source_backend="mock",
    )


@pytest.mark.asyncio
async def test_pipeline_emits_success_events(
    postgres_dsn: str, tmp_path: Path, clean_test_db: None
) -> None:
    # Given: A real Postgres event store and a pipeline with successful mocks
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    event_store = create_event_store_for_postgres_state_store(state_store)
    assert isinstance(event_store, PostgresEventStore)
    config = build_config()
    repository = ClipRepository(state_store, event_store, retry=config.retry)

    filter_result = FilterResult(
        detected_classes=["person"],
        confidence=0.9,
        model="mock",
        sampled_frames=30,
    )
    pipeline = ClipPipeline(
        config=config,
        storage=MockStorage(),
        repository=repository,
        filter_plugin=MockFilter(result=filter_result),
        vlm_plugin=MockVLM(
            result=AnalysisResult(
                risk_level="low",
                activity_type="person_passing",
                summary="Person walked through frame",
                prompt_tokens=1234,
                completion_tokens=567,
            )
        ),
        notifier=MockNotifier(),
        alert_policy=make_alert_policy(config),
        retention_pruner=MockRetentionPruner(),
    )
    clip = make_clip(tmp_path, "test-clip-events-001")

    # When: A clip is processed
    pipeline.on_new_clip(clip)
    await pipeline.shutdown()

    # Then: The expected lifecycle events are recorded
    events = await event_store.get_events(clip.clip_id)
    event_types = {event.event_type for event in events}
    assert "clip_recorded" in event_types
    assert "upload_started" in event_types
    assert "upload_completed" in event_types
    assert "filter_started" in event_types
    assert "filter_completed" in event_types
    assert "vlm_started" in event_types
    assert "vlm_completed" in event_types
    assert "alert_decision_made" in event_types
    assert "notification_sent" in event_types

    upload_event = next(event for event in events if event.event_type == "upload_completed")
    assert upload_event.storage_uri.startswith("mock://")

    alert_event = next(event for event in events if event.event_type == "alert_decision_made")
    assert alert_event.should_notify is True

    vlm_event = next(event for event in events if event.event_type == "vlm_completed")
    assert vlm_event.prompt_tokens == 1234
    assert vlm_event.completion_tokens == 567

    notify_event = next(event for event in events if event.event_type == "notification_sent")
    assert notify_event.attempt == 1

    await state_store.shutdown()


@pytest.mark.asyncio
async def test_pipeline_emits_notification_events_per_notifier(
    postgres_dsn: str, tmp_path: Path, clean_test_db: None
) -> None:
    # Given: A real Postgres event store and two notifier entries
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    event_store = create_event_store_for_postgres_state_store(state_store)
    assert isinstance(event_store, PostgresEventStore)
    config = build_config(notifier_count=2)
    repository = ClipRepository(state_store, event_store, retry=config.retry)

    filter_result = FilterResult(
        detected_classes=["person"],
        confidence=0.9,
        model="mock",
        sampled_frames=30,
    )
    notifier_a = MockNotifier()
    notifier_b = MockNotifier()
    entries = [
        NotifierEntry(name="mqtt[0]", notifier=notifier_a),
        NotifierEntry(name="mqtt[1]", notifier=notifier_b),
    ]
    pipeline = ClipPipeline(
        config=config,
        storage=MockStorage(),
        repository=repository,
        filter_plugin=MockFilter(result=filter_result),
        vlm_plugin=MockVLM(),
        notifier=MockNotifier(),
        notifier_entries=entries,
        alert_policy=make_alert_policy(config),
        retention_pruner=MockRetentionPruner(),
    )
    clip = make_clip(tmp_path, "test-clip-events-004")

    # When: A clip is processed
    pipeline.on_new_clip(clip)
    await pipeline.shutdown()

    # Then: Notification events are recorded per notifier entry
    events = await event_store.get_events(clip.clip_id)
    notify_events = [event for event in events if event.event_type == "notification_sent"]
    assert len(notify_events) == 2
    assert {event.notifier_name for event in notify_events} == {"mqtt[0]", "mqtt[1]"}
    assert len(notifier_a.sent_alerts) == 1
    assert len(notifier_b.sent_alerts) == 1

    await state_store.shutdown()


@pytest.mark.asyncio
async def test_pipeline_records_alert_decision_without_notification_events_when_no_notifiers(
    postgres_dsn: str, tmp_path: Path, clean_test_db: None
) -> None:
    # Given: A real Postgres event store and runtime-provided empty notifier entries
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    event_store = create_event_store_for_postgres_state_store(state_store)
    assert isinstance(event_store, PostgresEventStore)
    config = build_config(notifier_count=0)
    repository = ClipRepository(state_store, event_store, retry=config.retry)

    filter_result = FilterResult(
        detected_classes=["person"],
        confidence=0.9,
        model="mock",
        sampled_frames=30,
    )
    pipeline = ClipPipeline(
        config=config,
        storage=MockStorage(),
        repository=repository,
        filter_plugin=MockFilter(result=filter_result),
        vlm_plugin=MockVLM(),
        notifier=MockNotifier(),
        notifier_entries=[],
        alert_policy=make_alert_policy(config),
        retention_pruner=MockRetentionPruner(),
    )
    clip = make_clip(tmp_path, "test-clip-events-005")

    # When: A clip is processed
    pipeline.on_new_clip(clip)
    await pipeline.shutdown()

    # Then: Alert decision is recorded, but no notification sent/failed events are emitted
    events = await event_store.get_events(clip.clip_id)
    event_types = {event.event_type for event in events}
    assert "alert_decision_made" in event_types
    assert "notification_sent" not in event_types
    assert "notification_failed" not in event_types

    await state_store.shutdown()


@pytest.mark.asyncio
async def test_pipeline_emits_vlm_skipped_event(
    postgres_dsn: str, tmp_path: Path, clean_test_db: None
) -> None:
    # Given: A clip with non-trigger classes and notify_on_motion disabled
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    event_store = create_event_store_for_postgres_state_store(state_store)
    assert isinstance(event_store, PostgresEventStore)
    config = build_config()
    repository = ClipRepository(state_store, event_store, retry=config.retry)

    filter_result = FilterResult(
        detected_classes=["dog"],
        confidence=0.9,
        model="mock",
        sampled_frames=30,
    )
    pipeline = ClipPipeline(
        config=config,
        storage=MockStorage(),
        repository=repository,
        filter_plugin=MockFilter(result=filter_result),
        vlm_plugin=MockVLM(),
        notifier=MockNotifier(),
        alert_policy=make_alert_policy(config),
        retention_pruner=MockRetentionPruner(),
    )
    clip = make_clip(tmp_path, "test-clip-events-002")

    # When: A clip is processed
    pipeline.on_new_clip(clip)
    await pipeline.shutdown()

    # Then: VLM is skipped and no VLM start/complete events are emitted
    events = await event_store.get_events(clip.clip_id)
    event_types = {event.event_type for event in events}
    assert "vlm_skipped" in event_types
    assert "vlm_started" not in event_types
    assert "vlm_completed" not in event_types
    skipped_event = next(event for event in events if event.event_type == "vlm_skipped")
    assert skipped_event.reason == VLMSkipReason.NO_TRIGGER_CLASSES

    await state_store.shutdown()


@pytest.mark.asyncio
async def test_pipeline_emits_vlm_skipped_event_for_run_mode_never(
    postgres_dsn: str, tmp_path: Path, clean_test_db: None
) -> None:
    # Given: A clip with trigger classes but run_mode forcing VLM disabled
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    event_store = create_event_store_for_postgres_state_store(state_store)
    assert isinstance(event_store, PostgresEventStore)
    config = build_config(run_mode="never")
    repository = ClipRepository(state_store, event_store, retry=config.retry)

    filter_result = FilterResult(
        detected_classes=["person"],
        confidence=0.9,
        model="mock",
        sampled_frames=30,
    )
    pipeline = ClipPipeline(
        config=config,
        storage=MockStorage(),
        repository=repository,
        filter_plugin=MockFilter(result=filter_result),
        vlm_plugin=MockVLM(),
        notifier=MockNotifier(),
        alert_policy=make_alert_policy(config),
        retention_pruner=MockRetentionPruner(),
    )
    clip = make_clip(tmp_path, "test-clip-events-006")

    # When: A clip is processed
    pipeline.on_new_clip(clip)
    await pipeline.shutdown()

    # Then: VLM is skipped because run mode is disabled
    events = await event_store.get_events(clip.clip_id)
    event_types = {event.event_type for event in events}
    assert "vlm_skipped" in event_types
    assert "vlm_started" not in event_types
    assert "vlm_completed" not in event_types
    skipped_event = next(event for event in events if event.event_type == "vlm_skipped")
    assert skipped_event.reason == VLMSkipReason.RUN_MODE_NEVER

    await state_store.shutdown()


@pytest.mark.asyncio
async def test_pipeline_emits_upload_failed_event(
    postgres_dsn: str, tmp_path: Path, clean_test_db: None
) -> None:
    # Given: A pipeline where upload fails but filter/VLM succeed
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    event_store = create_event_store_for_postgres_state_store(state_store)
    assert isinstance(event_store, PostgresEventStore)
    config = build_config()
    repository = ClipRepository(state_store, event_store, retry=config.retry)

    filter_result = FilterResult(
        detected_classes=["person"],
        confidence=0.9,
        model="mock",
        sampled_frames=30,
    )
    pipeline = ClipPipeline(
        config=config,
        storage=MockStorage(simulate_failure=True),
        repository=repository,
        filter_plugin=MockFilter(result=filter_result),
        vlm_plugin=MockVLM(),
        notifier=MockNotifier(),
        alert_policy=make_alert_policy(config),
        retention_pruner=MockRetentionPruner(),
    )
    clip = make_clip(tmp_path, "test-clip-events-003")

    # When: A clip is processed
    pipeline.on_new_clip(clip)
    await pipeline.shutdown()

    # Then: Upload failure is recorded and notification still sends
    events = await event_store.get_events(clip.clip_id)
    event_types = {event.event_type for event in events}
    assert "upload_failed" in event_types
    assert "notification_sent" in event_types
    upload_failed_event = next(event for event in events if event.event_type == "upload_failed")
    assert upload_failed_event.will_retry is False

    await state_store.shutdown()
