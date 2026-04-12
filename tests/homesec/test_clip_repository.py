"""Tests for ClipRepository."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from homesec.models.alert import AlertDecision
from homesec.models.clip import Clip, ClipStateData
from homesec.models.enums import ClipStatus, RiskLevel
from homesec.models.events import ClipRecheckedEvent
from homesec.models.filter import FilterResult
from homesec.models.vlm import (
    AnalysisResult,
    SequenceAnalysis,
)
from homesec.repository import ClipRepository
from homesec.state.postgres import PostgresEventStore, PostgresStateStore
from tests.homesec.mocks import MockEventStore, MockStateStore


def _build_state(*, status: ClipStatus) -> ClipStateData:
    return ClipStateData(
        camera_name="front_door",
        status=status,
        local_path="/tmp/test.mp4",
    )


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
async def test_initialize_clip_records_timezone_aware_timestamp(tmp_path: Path) -> None:
    """Initialize clip events should be emitted with UTC-aware timestamps."""
    # Given: A repository using in-memory mock stores
    event_store = MockEventStore()
    repository = ClipRepository(MockStateStore(), event_store)
    clip = Clip(
        clip_id="test-clip-tz-001",
        camera_name="front_door",
        local_path=tmp_path / "test.mp4",
        start_ts=datetime.now(timezone.utc),
        end_ts=datetime.now(timezone.utc) + timedelta(seconds=10),
        duration_s=10.0,
        source_backend="test",
    )

    # When: Initializing the clip through the repository
    await repository.initialize_clip(clip)
    events = event_store.events

    # Then: The recorded event timestamp is timezone-aware
    assert len(events) == 1
    assert events[0].timestamp.tzinfo is not None
    assert events[0].timestamp.utcoffset() is not None


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
async def test_record_alert_decision_records_state_and_event(tmp_path: Path) -> None:
    # Given: A repository using in-memory mock stores
    state_store = MockStateStore()
    event_store = MockEventStore()
    repository = ClipRepository(state_store, event_store)

    clip = Clip(
        clip_id="test-clip-004a",
        camera_name="front_door",
        local_path=tmp_path / "test.mp4",
        start_ts=datetime.now(),
        end_ts=datetime.now() + timedelta(seconds=10),
        duration_s=10.0,
        source_backend="test",
    )
    await repository.initialize_clip(clip)

    decision = AlertDecision(notify=True, notify_reason="risk_level=high")

    # When: An alert decision is recorded
    await repository.record_alert_decision(
        clip.clip_id,
        decision,
        detected_classes=["person"],
        vlm_risk=RiskLevel.HIGH,
    )

    # Then: State records the alert decision
    state = await state_store.get(clip.clip_id)
    assert state is not None
    assert state.alert_decision == decision

    # And: Event is recorded
    events = event_store.events
    assert len(events) == 2
    assert events[1].event_type == "alert_decision_made"


@pytest.mark.asyncio
async def test_record_alert_decision_returns_none_for_missing_clip() -> None:
    # Given: A repository with no stored clip state
    state_store = MockStateStore()
    event_store = MockEventStore()
    repository = ClipRepository(state_store, event_store)
    decision = AlertDecision(notify=True, notify_reason="risk_level=high")

    # When: Recording an alert decision for an unknown clip
    result = await repository.record_alert_decision(
        "missing-clip",
        decision,
        detected_classes=["person"],
        vlm_risk=RiskLevel.HIGH,
    )

    # Then: No state or event is recorded and None is returned
    assert result is None
    assert event_store.events == []
    assert state_store.states == {}


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_status", "expected_status"),
    [
        (ClipStatus.QUEUED_LOCAL, ClipStatus.UPLOADED),
        (ClipStatus.UPLOADED, ClipStatus.UPLOADED),
        (ClipStatus.ANALYZED, ClipStatus.ANALYZED),
        (ClipStatus.DONE, ClipStatus.DONE),
        (ClipStatus.ERROR, ClipStatus.ERROR),
        (ClipStatus.DELETED, ClipStatus.DELETED),
    ],
)
async def test_record_upload_completed_applies_explicit_status_transition_rules(
    initial_status: ClipStatus,
    expected_status: ClipStatus,
) -> None:
    # Given: A repository with a clip already in a known status
    state_store = MockStateStore()
    event_store = MockEventStore()
    repository = ClipRepository(state_store, event_store)
    clip_id = f"upload-{initial_status.value}"
    await state_store.upsert(clip_id, _build_state(status=initial_status))

    # When: Upload completion is recorded
    state = await repository.record_upload_completed(
        clip_id,
        "dropbox://test.mp4",
        "https://example.com/test.mp4",
        5000,
    )

    # Then: The centralized transition rule preserves the existing semantics
    assert state is not None
    assert state.status == expected_status
    persisted = await state_store.get(clip_id)
    assert persisted is not None
    assert persisted.status == expected_status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_status", "will_retry", "expected_status"),
    [
        (ClipStatus.QUEUED_LOCAL, False, ClipStatus.ERROR),
        (ClipStatus.DELETED, False, ClipStatus.DELETED),
        (ClipStatus.UPLOADED, True, ClipStatus.UPLOADED),
    ],
)
async def test_record_filter_failed_applies_explicit_status_transition_rules(
    initial_status: ClipStatus,
    will_retry: bool,
    expected_status: ClipStatus,
) -> None:
    # Given: A repository with a clip already in a known status
    state_store = MockStateStore()
    event_store = MockEventStore()
    repository = ClipRepository(state_store, event_store)
    clip_id = f"filter-failed-{initial_status.value}-{will_retry}"
    await state_store.upsert(clip_id, _build_state(status=initial_status))

    # When: A filter failure is recorded
    state = await repository.record_filter_failed(
        clip_id,
        "boom",
        "RuntimeError",
        will_retry=will_retry,
    )

    # Then: The centralized transition rule preserves the existing semantics
    if will_retry:
        assert state is None
    else:
        assert state is not None
        assert state.status == expected_status
    persisted = await state_store.get(clip_id)
    assert persisted is not None
    assert persisted.status == expected_status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_status", "expected_status"),
    [
        (ClipStatus.UPLOADED, ClipStatus.ANALYZED),
        (ClipStatus.ERROR, ClipStatus.ANALYZED),
        (ClipStatus.DELETED, ClipStatus.DELETED),
    ],
)
async def test_record_vlm_completed_applies_explicit_status_transition_rules(
    initial_status: ClipStatus,
    expected_status: ClipStatus,
) -> None:
    # Given: A repository with a clip already in a known status
    state_store = MockStateStore()
    event_store = MockEventStore()
    repository = ClipRepository(state_store, event_store)
    clip_id = f"vlm-{initial_status.value}"
    await state_store.upsert(clip_id, _build_state(status=initial_status))
    result = AnalysisResult(
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

    # When: VLM completion is recorded
    state = await repository.record_vlm_completed(clip_id, result, 100, 50, 1000)

    # Then: The centralized transition rule preserves the existing semantics
    assert state is not None
    assert state.status == expected_status
    persisted = await state_store.get(clip_id)
    assert persisted is not None
    assert persisted.status == expected_status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_status", "expected_status"),
    [
        (ClipStatus.ANALYZED, ClipStatus.DONE),
        (ClipStatus.ERROR, ClipStatus.DONE),
        (ClipStatus.DELETED, ClipStatus.DELETED),
    ],
)
async def test_record_notification_sent_applies_explicit_status_transition_rules(
    initial_status: ClipStatus,
    expected_status: ClipStatus,
) -> None:
    # Given: A repository with a clip already in a known status
    state_store = MockStateStore()
    event_store = MockEventStore()
    repository = ClipRepository(state_store, event_store)
    clip_id = f"notify-{initial_status.value}"
    await state_store.upsert(clip_id, _build_state(status=initial_status))

    # When: Notification delivery is recorded
    state = await repository.record_notification_sent(clip_id, "mqtt", clip_id)

    # Then: The centralized transition rule preserves the existing semantics
    assert state is not None
    assert state.status == expected_status
    persisted = await state_store.get(clip_id)
    assert persisted is not None
    assert persisted.status == expected_status


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_status", list(ClipStatus))
async def test_record_clip_deleted_applies_explicit_status_transition_rules(
    initial_status: ClipStatus,
) -> None:
    # Given: A repository with a clip already in a known status
    state_store = MockStateStore()
    event_store = MockEventStore()
    repository = ClipRepository(state_store, event_store)
    clip_id = f"delete-{initial_status.value}"
    await state_store.upsert(clip_id, _build_state(status=initial_status))

    # When: Clip deletion is recorded
    state = await repository.record_clip_deleted(
        clip_id,
        reason="cleanup",
        run_id="run-123",
        deleted_local=True,
        deleted_storage=True,
    )

    # Then: Delete always wins regardless of the prior status
    assert state is not None
    assert state.status == ClipStatus.DELETED
    persisted = await state_store.get(clip_id)
    assert persisted is not None
    assert persisted.status == ClipStatus.DELETED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_status", "expected_status"),
    [
        (ClipStatus.ANALYZED, ClipStatus.DONE),
        (ClipStatus.DONE, ClipStatus.DONE),
        (ClipStatus.DELETED, ClipStatus.DELETED),
    ],
)
async def test_mark_done_applies_explicit_status_transition_rules(
    initial_status: ClipStatus,
    expected_status: ClipStatus,
) -> None:
    # Given: A repository with a clip already in a known status
    state_store = MockStateStore()
    event_store = MockEventStore()
    repository = ClipRepository(state_store, event_store)
    clip_id = f"done-{initial_status.value}"
    await state_store.upsert(clip_id, _build_state(status=initial_status))

    # When: Completion is recorded without an event
    state = await repository.mark_done(clip_id)

    # Then: The centralized transition rule preserves the existing semantics
    assert state is not None
    assert state.status == expected_status
    persisted = await state_store.get(clip_id)
    assert persisted is not None
    assert persisted.status == expected_status


@pytest.mark.asyncio
async def test_get_clip_states_with_created_at_uses_batch_lookup() -> None:
    # Given: A repository with in-memory mock state and two uploaded clips
    state_store = MockStateStore()
    event_store = MockEventStore()
    repository = ClipRepository(state_store, event_store)
    clip_a = "test-batch-a"
    clip_b = "test-batch-b"
    await state_store.upsert(
        clip_a,
        ClipStateData(
            camera_name="front_door",
            status="done",
            local_path="/tmp/a.mp4",
            storage_uri="mock://a",
        ),
    )
    await state_store.upsert(
        clip_b,
        ClipStateData(
            camera_name="front_door",
            status="done",
            local_path="/tmp/b.mp4",
            storage_uri="mock://b",
        ),
    )

    # When: Reading many clip states in one repository call
    states = await repository.get_clip_states_with_created_at([clip_a, clip_b, "missing"])

    # Then: Existing clip ids are returned and batch store API is used once
    assert set(states.keys()) == {clip_a, clip_b}
    state_a, created_at_a = states[clip_a]
    state_b, created_at_b = states[clip_b]
    assert state_a.clip_id == clip_a
    assert state_b.clip_id == clip_b
    assert state_a.created_at == created_at_a
    assert state_b.created_at == created_at_b
    assert state_store.get_many_count == 1
    assert state_store.get_count == 0


@pytest.mark.asyncio
async def test_count_alerts_since_returns_zero_when_state_store_count_fails() -> None:
    class FailingAlertCountStateStore(MockStateStore):
        async def count_alerts_since(self, since: datetime) -> int:
            _ = since
            raise RuntimeError("Simulated alert count failure")

    # Given: A repository whose state store raises during alert counting
    state_store = FailingAlertCountStateStore()
    repository = ClipRepository(state_store, MockEventStore())
    since = datetime.now(timezone.utc) - timedelta(hours=1)

    # When: Counting alerts through the repository
    count = await repository.count_alerts_since(since)

    # Then: The repository degrades to zero instead of raising
    assert count == 0
