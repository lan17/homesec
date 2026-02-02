"""Tests for ClipRepository."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import text

from homesec.models.alert import AlertDecision
from homesec.models.clip import Clip
from homesec.models.enums import ClipStatus, RiskLevel
from homesec.models.events import ClipRecheckedEvent
from homesec.models.filter import FilterResult
from homesec.models.vlm import AnalysisResult, SequenceAnalysis
from homesec.repository import ClipRepository
from homesec.state.postgres import PostgresEventStore, PostgresStateStore


@pytest.mark.asyncio
async def test_initialize_clip(postgres_dsn: str, tmp_path: Path, clean_test_db: None) -> None:
    # Given: A repository with state and event stores
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    event_store = state_store.create_event_store()
    assert isinstance(event_store, PostgresEventStore)
    repository = ClipRepository(state_store, event_store)

    clip = Clip(
        clip_id="test-clip-001",
        camera_name="front_door",
        local_path=tmp_path / "test.mp4",
        start_ts=datetime.now(),
        end_ts=datetime.now() + timedelta(seconds=10),
        duration_s=10.0,
        source_backend="test",
    )

    # When: We initialize a clip
    state = await repository.initialize_clip(clip)

    # Then: State is created with correct values
    assert state.camera_name == "front_door"
    assert state.status == "queued_local"
    assert state.storage_uri is None

    # And: Event is recorded
    events = await event_store.get_events(clip.clip_id)
    assert len(events) == 1
    assert events[0].event_type == "clip_recorded"

    # Cleanup
    await state_store.shutdown()


@pytest.mark.asyncio
async def test_count_clips_since(postgres_dsn: str, tmp_path: Path, clean_test_db: None) -> None:
    """count_clips_since should include clips created after the cutoff."""
    # Given: A repository with two clips created around a cutoff
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    event_store = state_store.create_event_store()
    repository = ClipRepository(state_store, event_store)

    clip_old = Clip(
        clip_id="test-clip-old",
        camera_name="front_door",
        local_path=tmp_path / "old.mp4",
        start_ts=datetime.now(),
        end_ts=datetime.now(),
        duration_s=1.0,
        source_backend="test",
    )
    await repository.initialize_clip(clip_old)
    clip_new = Clip(
        clip_id="test-clip-new",
        camera_name="front_door",
        local_path=tmp_path / "new.mp4",
        start_ts=datetime.now(),
        end_ts=datetime.now(),
        duration_s=1.0,
        source_backend="test",
    )
    await repository.initialize_clip(clip_new)

    # Given: Normalize timestamps using database time to avoid clock skew
    assert state_store._engine is not None
    async with state_store._engine.begin() as conn:
        result = await conn.execute(text("SELECT now()"))
        db_now = result.scalar_one()
        await conn.execute(
            text("UPDATE clip_states SET created_at = :created_at WHERE clip_id = :clip_id"),
            {"created_at": db_now - timedelta(hours=2), "clip_id": clip_old.clip_id},
        )

    cutoff = db_now - timedelta(hours=1)

    # When: Counting clips since the cutoff
    count = await repository.count_clips_since(cutoff)

    # Then: Only the newer clip is counted
    assert count == 1

    await state_store.shutdown()


@pytest.mark.asyncio
async def test_count_alerts_since(postgres_dsn: str, tmp_path: Path, clean_test_db: None) -> None:
    """count_alerts_since should count notification_sent events."""
    # Given: A repository with one notification_sent event
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    event_store = state_store.create_event_store()
    repository = ClipRepository(state_store, event_store)

    clip = Clip(
        clip_id="test-clip-alert",
        camera_name="front_door",
        local_path=tmp_path / "alert.mp4",
        start_ts=datetime.now(),
        end_ts=datetime.now(),
        duration_s=1.0,
        source_backend="test",
    )
    await repository.initialize_clip(clip)
    await repository.record_notification_sent(
        clip_id=clip.clip_id,
        notifier_name="test",
        dedupe_key=clip.clip_id,
    )

    # When: Counting alerts since a recent timestamp
    since = datetime.now() - timedelta(minutes=1)
    count = await repository.count_alerts_since(since)

    # Then: The alert is counted
    assert count == 1

    await state_store.shutdown()


@pytest.mark.asyncio
async def test_list_clips_filters(postgres_dsn: str, tmp_path: Path, clean_test_db: None) -> None:
    """list_clips should apply filters correctly."""
    # Given: Two clips with different cameras and risk levels
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    event_store = state_store.create_event_store()
    repository = ClipRepository(state_store, event_store)

    clip_front = Clip(
        clip_id="test-clip-front",
        camera_name="front_door",
        local_path=tmp_path / "front.mp4",
        start_ts=datetime.now(),
        end_ts=datetime.now(),
        duration_s=1.0,
        source_backend="test",
    )
    clip_back = Clip(
        clip_id="test-clip-back",
        camera_name="back_door",
        local_path=tmp_path / "back.mp4",
        start_ts=datetime.now(),
        end_ts=datetime.now(),
        duration_s=1.0,
        source_backend="test",
    )
    await repository.initialize_clip(clip_front)
    await repository.initialize_clip(clip_back)

    await repository.record_vlm_completed(
        clip_id=clip_front.clip_id,
        result=AnalysisResult(
            risk_level="high",
            activity_type="suspicious_behavior",
            summary="Suspicious",
            analysis=None,
        ),
        prompt_tokens=None,
        completion_tokens=None,
        duration_ms=10,
    )
    await repository.record_vlm_completed(
        clip_id=clip_back.clip_id,
        result=AnalysisResult(
            risk_level="low",
            activity_type="passerby",
            summary="Normal",
            analysis=None,
        ),
        prompt_tokens=None,
        completion_tokens=None,
        duration_ms=10,
    )

    await repository.record_alert_decision(
        clip_id=clip_front.clip_id,
        decision=AlertDecision(notify=True, notify_reason="risk_level=high"),
        detected_classes=["person"],
        vlm_risk="high",
    )
    await repository.record_alert_decision(
        clip_id=clip_back.clip_id,
        decision=AlertDecision(notify=False, notify_reason="low risk"),
        detected_classes=["person"],
        vlm_risk="low",
    )

    # When: Listing clips by camera, alert status, and risk level
    clips_by_camera, total_by_camera = await repository.list_clips(camera="front_door")
    clips_alerted, total_alerted = await repository.list_clips(alerted=True)
    clips_high, total_high = await repository.list_clips(risk_level="high")

    # Then: Each filter returns the expected clip
    assert total_by_camera == 1
    assert clips_by_camera[0].camera_name == "front_door"

    assert total_alerted == 1
    assert clips_alerted[0].clip_id == clip_front.clip_id

    assert total_high == 1
    assert clips_high[0].clip_id == clip_front.clip_id

    await state_store.shutdown()


@pytest.mark.asyncio
async def test_delete_clip_marks_deleted(
    postgres_dsn: str, tmp_path: Path, clean_test_db: None
) -> None:
    """delete_clip should mark clip as deleted."""
    # Given: A clip with uploaded storage
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    event_store = state_store.create_event_store()
    repository = ClipRepository(state_store, event_store)

    clip = Clip(
        clip_id="test-clip-delete",
        camera_name="front_door",
        local_path=tmp_path / "delete.mp4",
        start_ts=datetime.now(),
        end_ts=datetime.now(),
        duration_s=1.0,
        source_backend="test",
    )
    await repository.initialize_clip(clip)
    await repository.record_upload_completed(
        clip_id=clip.clip_id,
        storage_uri="dropbox://delete.mp4",
        view_url=None,
        duration_ms=10,
    )

    # When: Deleting the clip
    state = await repository.delete_clip(clip.clip_id)

    # Then: State is marked deleted
    assert state.status == ClipStatus.DELETED
    assert state.storage_uri == "dropbox://delete.mp4"

    await state_store.shutdown()


@pytest.mark.asyncio
async def test_record_upload_completed(
    postgres_dsn: str, tmp_path: Path, clean_test_db: None
) -> None:
    # Given: A clip that's been initialized
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    event_store = state_store.create_event_store()
    assert isinstance(event_store, PostgresEventStore)
    repository = ClipRepository(state_store, event_store)

    clip = Clip(
        clip_id="test-clip-002",
        camera_name="front_door",
        local_path=tmp_path / "test.mp4",
        start_ts=datetime.now(),
        end_ts=datetime.now() + timedelta(seconds=10),
        duration_s=10.0,
        source_backend="test",
    )
    await repository.initialize_clip(clip)

    # When: Upload completes
    await repository.record_upload_completed(
        clip.clip_id,
        "dropbox://test.mp4",
        "https://dropbox.com/test.mp4",
        5000,
    )

    # Then: State is updated
    state = await state_store.get(clip.clip_id)
    assert state is not None
    assert state.storage_uri == "dropbox://test.mp4"
    assert state.view_url == "https://dropbox.com/test.mp4"
    assert state.status == "uploaded"

    # And: Event is recorded
    events = await event_store.get_events(clip.clip_id)
    assert len(events) == 2
    assert events[1].event_type == "upload_completed"

    # Cleanup
    await state_store.shutdown()


@pytest.mark.asyncio
async def test_record_filter_completed(
    postgres_dsn: str, tmp_path: Path, clean_test_db: None
) -> None:
    # Given: A clip that's been initialized
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    event_store = state_store.create_event_store()
    assert isinstance(event_store, PostgresEventStore)
    repository = ClipRepository(state_store, event_store)

    clip = Clip(
        clip_id="test-clip-003",
        camera_name="front_door",
        local_path=tmp_path / "test.mp4",
        start_ts=datetime.now(),
        end_ts=datetime.now() + timedelta(seconds=10),
        duration_s=10.0,
        source_backend="test",
    )
    await repository.initialize_clip(clip)

    filter_result = FilterResult(
        detected_classes=["person", "car"],
        confidence=0.95,
        model="yolov8n",
        sampled_frames=10,
    )

    # When: Filter completes
    await repository.record_filter_completed(clip.clip_id, filter_result, 2000)

    # Then: State is updated with filter result
    state = await state_store.get(clip.clip_id)
    assert state is not None
    assert state.filter_result is not None
    assert state.filter_result.detected_classes == ["person", "car"]

    # And: Event is recorded
    events = await event_store.get_events(clip.clip_id)
    assert len(events) == 2
    assert events[1].event_type == "filter_completed"

    # Cleanup
    await state_store.shutdown()


@pytest.mark.asyncio
async def test_record_clip_rechecked_updates_state_and_event(
    postgres_dsn: str,
    tmp_path: Path,
    clean_test_db: None,
) -> None:
    # Given: A clip initialized in the repository
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    event_store = state_store.create_event_store()
    assert isinstance(event_store, PostgresEventStore)
    repository = ClipRepository(state_store, event_store)

    clip = Clip(
        clip_id="test-clip-recheck-001",
        camera_name="front_door",
        local_path=tmp_path / "test.mp4",
        start_ts=datetime.now(),
        end_ts=datetime.now() + timedelta(seconds=10),
        duration_s=10.0,
        source_backend="test",
    )
    await repository.initialize_clip(clip)

    prior = FilterResult(
        detected_classes=[],
        confidence=0.1,
        model="yolo11n.pt",
        sampled_frames=5,
    )
    recheck = FilterResult(
        detected_classes=["person"],
        confidence=0.95,
        model="yolo11x.pt",
        sampled_frames=8,
    )

    # When: Recording a recheck result
    await repository.record_clip_rechecked(
        clip.clip_id,
        result=recheck,
        prior_filter=prior,
        reason="cleanup_cli",
        run_id="run-123",
    )

    # Then: State reflects the recheck result
    state = await state_store.get(clip.clip_id)
    assert state is not None
    assert state.filter_result == recheck

    # And: A clip_rechecked event is recorded
    events = await event_store.get_events(clip.clip_id)
    assert events[-1].event_type == "clip_rechecked"
    assert isinstance(events[-1], ClipRecheckedEvent)
    assert events[-1].prior_filter == prior
    assert events[-1].recheck_filter == recheck

    # Cleanup
    await state_store.shutdown()


@pytest.mark.asyncio
async def test_record_vlm_completed(postgres_dsn: str, tmp_path: Path, clean_test_db: None) -> None:
    # Given: A clip that's been initialized
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    event_store = state_store.create_event_store()
    assert isinstance(event_store, PostgresEventStore)
    repository = ClipRepository(state_store, event_store)

    clip = Clip(
        clip_id="test-clip-004",
        camera_name="front_door",
        local_path=tmp_path / "test.mp4",
        start_ts=datetime.now(),
        end_ts=datetime.now() + timedelta(seconds=10),
        duration_s=10.0,
        source_backend="test",
    )
    await repository.initialize_clip(clip)

    vlm_result = AnalysisResult(
        risk_level="low",
        activity_type="person_walking",
        summary="Person walking past camera",
        analysis=SequenceAnalysis(
            sequence_description="Person walks by",
            primary_activity="passerby",
            max_risk_level="low",
            entities_timeline=[],
            observations=[],
            requires_review=False,
            frame_count=10,
            video_start_time="00:00:00",
            video_end_time="00:00:10",
        ),
    )

    # When: VLM completes
    await repository.record_vlm_completed(clip.clip_id, vlm_result, 1000, 500, 15000)

    # Then: State is updated with VLM result and status is analyzed
    state = await state_store.get(clip.clip_id)
    assert state is not None
    assert state.analysis_result is not None
    assert state.analysis_result.risk_level == RiskLevel.LOW
    assert state.status == "analyzed"

    # And: Event is recorded with token usage
    events = await event_store.get_events(clip.clip_id)
    assert len(events) == 2
    assert events[1].event_type == "vlm_completed"

    # Cleanup
    await state_store.shutdown()


@pytest.mark.asyncio
async def test_record_notification_sent(
    postgres_dsn: str, tmp_path: Path, clean_test_db: None
) -> None:
    # Given: A clip that's been analyzed
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    event_store = state_store.create_event_store()
    assert isinstance(event_store, PostgresEventStore)
    repository = ClipRepository(state_store, event_store)

    clip = Clip(
        clip_id="test-clip-005",
        camera_name="front_door",
        local_path=tmp_path / "test.mp4",
        start_ts=datetime.now(),
        end_ts=datetime.now() + timedelta(seconds=10),
        duration_s=10.0,
        source_backend="test",
    )
    await repository.initialize_clip(clip)

    # When: Notification is sent
    await repository.record_notification_sent(clip.clip_id, "mqtt", clip.clip_id)

    # Then: State is marked as done
    state = await state_store.get(clip.clip_id)
    assert state is not None
    assert state.status == "done"

    # And: Event is recorded
    events = await event_store.get_events(clip.clip_id)
    assert len(events) == 2
    assert events[1].event_type == "notification_sent"

    # Cleanup
    await state_store.shutdown()
