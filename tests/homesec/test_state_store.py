"""Tests for PostgresStateStore."""

import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete

from homesec.models.alert import AlertDecision
from homesec.models.clip import ClipStateData
from homesec.models.enums import ClipStatus
from homesec.models.events import NotificationSentEvent
from homesec.models.filter import FilterResult
from homesec.models.vlm import AnalysisResult
from homesec.state import PostgresStateStore
from homesec.state.postgres import Base, ClipState, _normalize_async_dsn

# Default DSN for local Docker Postgres (matches docker-compose.postgres.yml)
DEFAULT_DSN = "postgresql://homesec:homesec@localhost:5432/homesec"


def get_test_dsn() -> str:
    """Get test database DSN from environment or use default."""
    return os.environ.get("TEST_DB_DSN", DEFAULT_DSN)


@pytest.fixture
async def state_store() -> PostgresStateStore:
    """Create and initialize a PostgresStateStore for testing."""
    dsn = get_test_dsn()
    assert dsn is not None
    store = PostgresStateStore(dsn)
    initialized = await store.initialize()
    assert initialized, "Failed to initialize state store"
    if store._engine is not None:
        async with store._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
    yield store
    # Cleanup: drop test data
    if store._engine is not None:
        async with store._engine.begin() as conn:
            await conn.execute(delete(ClipState).where(ClipState.clip_id.like("test_%")))
    await store.shutdown()


def sample_state(clip_id: str = "test_clip_001") -> ClipStateData:
    """Create a sample ClipStateData for testing."""
    return ClipStateData(
        camera_name="front_door",
        status="queued_local",
        local_path="/tmp/test.mp4",
    )


def test_parse_state_data_accepts_dict_and_str() -> None:
    """Test JSONB parsing handles dicts and JSON strings."""
    # Given a clip state dict and JSON string
    state = sample_state()
    raw_dict = state.model_dump()
    raw_str = state.model_dump_json()

    # When parsing raw JSONB payloads
    parsed_dict = PostgresStateStore._parse_state_data(raw_dict)
    parsed_str = PostgresStateStore._parse_state_data(raw_str)

    # Then both parse to dicts
    assert parsed_dict == raw_dict
    assert parsed_str == raw_dict
    ClipStateData.model_validate(parsed_str)


def test_parse_state_data_accepts_bytes() -> None:
    """Test JSONB parsing handles bytes payloads."""
    # Given a JSON payload as bytes
    state = sample_state()
    raw_bytes = state.model_dump_json().encode("utf-8")

    # When parsing bytes
    parsed = PostgresStateStore._parse_state_data(raw_bytes)

    # Then bytes parse to dict
    assert parsed["camera_name"] == "front_door"
    ClipStateData.model_validate(parsed)


def test_normalize_async_dsn() -> None:
    """Test DSN normalization adds asyncpg driver."""
    # Given different Postgres DSN formats
    dsn_plain = "postgresql://user:pass@localhost/db"
    dsn_short = "postgres://user:pass@localhost/db"
    dsn_async = "postgresql+asyncpg://user:pass@localhost/db"

    # When normalizing DSNs
    norm_plain = _normalize_async_dsn(dsn_plain)
    norm_short = _normalize_async_dsn(dsn_short)
    norm_async = _normalize_async_dsn(dsn_async)

    # Then asyncpg is used
    assert norm_plain.startswith("postgresql+asyncpg://")
    assert norm_short.startswith("postgresql+asyncpg://")
    assert norm_async == dsn_async


@pytest.mark.asyncio
async def test_upsert_and_get_roundtrip(state_store: PostgresStateStore) -> None:
    """Test that upsert and get work correctly."""
    # Given a clip state
    clip_id = "test_roundtrip_001"
    state = sample_state(clip_id)

    # When upserting and fetching
    await state_store.upsert(clip_id, state)
    retrieved = await state_store.get(clip_id)

    # Then the roundtrip returns the state
    assert retrieved is not None
    assert retrieved.camera_name == "front_door"
    assert retrieved.status == "queued_local"


@pytest.mark.asyncio
async def test_upsert_updates_existing(state_store: PostgresStateStore) -> None:
    """Test that upsert updates existing records."""
    # Given an existing clip state
    clip_id = "test_update_001"
    state = sample_state(clip_id)

    # When inserting and updating
    await state_store.upsert(clip_id, state)

    state.status = "uploaded"
    state.storage_uri = "dropbox:/front_door/test.mp4"
    await state_store.upsert(clip_id, state)

    # Then the updated fields are persisted
    retrieved = await state_store.get(clip_id)
    assert retrieved is not None
    assert retrieved.status == "uploaded"
    assert retrieved.storage_uri == "dropbox:/front_door/test.mp4"


@pytest.mark.asyncio
async def test_get_returns_none_for_missing(state_store: PostgresStateStore) -> None:
    """Test that get returns None for non-existent clip_id."""
    # Given a missing clip id
    # When retrieving a missing clip id
    result = await state_store.get("test_nonexistent_999")
    # Then None is returned
    assert result is None


@pytest.mark.asyncio
async def test_ping_returns_true(state_store: PostgresStateStore) -> None:
    """Test that ping returns True when connected."""
    # Given an initialized store
    # When ping is called
    result = await state_store.ping()
    # Then ping is True
    assert result is True


@pytest.mark.asyncio
async def test_graceful_degradation_uninitialized() -> None:
    """Test graceful degradation when store is not initialized."""
    # Given an uninitialized store
    store = PostgresStateStore("postgresql://invalid:5432/nonexistent")

    # When operations are called
    await store.upsert("test_fail", sample_state())
    result = await store.get("test_fail")
    ping = await store.ping()

    # Then operations degrade gracefully
    assert result is None
    assert ping is False

    await store.shutdown()  # Should not raise


@pytest.mark.asyncio
async def test_initialize_returns_false_on_bad_dsn() -> None:
    """Test that initialize returns False for invalid DSN."""
    # Given an invalid DSN
    store = PostgresStateStore("postgresql://invalid:5432/nonexistent")
    # When initializing
    result = await store.initialize()
    # Then initialization fails
    assert result is False
    await store.shutdown()


@pytest.mark.asyncio
async def test_list_candidate_clips_for_cleanup_skips_deleted_and_filters_camera(
    state_store: PostgresStateStore,
) -> None:
    """Cleanup listing should skip deleted clips and respect camera filter."""
    # Given: Three clip states (one deleted)
    empty = FilterResult(
        detected_classes=[],
        confidence=0.0,
        model="yolo",
        sampled_frames=1,
    )
    found = FilterResult(
        detected_classes=["person"],
        confidence=0.9,
        model="yolo",
        sampled_frames=1,
    )

    await state_store.upsert(
        "test-cleanup-a",
        ClipStateData(
            camera_name="front_door",
            status="analyzed",
            local_path="/tmp/a.mp4",
            filter_result=empty,
        ),
    )
    await state_store.upsert(
        "test-cleanup-b",
        ClipStateData(
            camera_name="front_door",
            status="deleted",
            local_path="/tmp/b.mp4",
            filter_result=empty,
        ),
    )
    await state_store.upsert(
        "test-cleanup-c",
        ClipStateData(
            camera_name="backyard",
            status="analyzed",
            local_path="/tmp/c.mp4",
            filter_result=found,
        ),
    )

    # When: Listing without a camera filter
    rows = await state_store.list_candidate_clips_for_cleanup(
        older_than_days=None,
        camera_name=None,
        batch_size=100,
        cursor=None,
    )
    ids = {clip_id for clip_id, _state, _created_at in rows}

    # Then: Deleted clips are excluded
    assert "test-cleanup-b" not in ids
    assert "test-cleanup-a" in ids
    assert "test-cleanup-c" in ids

    # When: Listing for a single camera
    rows_front = await state_store.list_candidate_clips_for_cleanup(
        older_than_days=None,
        camera_name="front_door",
        batch_size=100,
        cursor=None,
    )
    ids_front = {clip_id for clip_id, _state, _created_at in rows_front}

    # Then: Only that camera's non-deleted rows are returned
    assert ids_front == {"test-cleanup-a"}


@pytest.mark.asyncio
async def test_get_clip_returns_clip_id_and_created_at(state_store: PostgresStateStore) -> None:
    """get_clip should hydrate clip_id and created_at metadata."""
    # Given: A persisted clip state
    clip_id = "test-get-clip-meta-001"
    await state_store.upsert(
        clip_id,
        ClipStateData(
            camera_name="front_door",
            status="queued_local",
            local_path="/tmp/meta.mp4",
        ),
    )

    # When: Fetching it by clip id
    clip = await state_store.get_clip(clip_id)

    # Then: Metadata fields are populated from row columns
    assert clip is not None
    assert clip.clip_id == clip_id
    assert clip.created_at is not None
    assert clip.created_at.tzinfo is not None


@pytest.mark.asyncio
async def test_mark_clip_deleted_updates_persisted_status(state_store: PostgresStateStore) -> None:
    """mark_clip_deleted should set status to deleted and persist the update."""
    # Given: A clip state in a non-deleted status
    clip_id = "test-mark-deleted-001"
    await state_store.upsert(
        clip_id,
        ClipStateData(
            camera_name="front_door",
            status="uploaded",
            local_path="/tmp/deleted.mp4",
        ),
    )

    # When: Marking the clip as deleted
    deleted = await state_store.mark_clip_deleted(clip_id)

    # Then: Returned state and persisted state both report deleted status
    assert deleted.status == ClipStatus.DELETED
    persisted = await state_store.get_clip(clip_id)
    assert persisted is not None
    assert persisted.status == ClipStatus.DELETED


@pytest.mark.asyncio
async def test_list_clips_applies_filters_and_includes_clip_ids(
    state_store: PostgresStateStore,
) -> None:
    """list_clips should support camera/status/alerted filters."""
    # Given: Clips with mixed status and alert outcomes
    await state_store.upsert(
        "test-list-filters-a",
        ClipStateData(
            camera_name="front_door",
            status=ClipStatus.ANALYZED,
            local_path="/tmp/a.mp4",
            alert_decision=AlertDecision(notify=True, notify_reason="risk_level=high"),
        ),
    )
    await state_store.upsert(
        "test-list-filters-b",
        ClipStateData(
            camera_name="front_door",
            status=ClipStatus.ANALYZED,
            local_path="/tmp/b.mp4",
            alert_decision=AlertDecision(notify=False, notify_reason="risk_level=low"),
        ),
    )
    await state_store.upsert(
        "test-list-filters-c",
        ClipStateData(
            camera_name="back_door",
            status=ClipStatus.UPLOADED,
            local_path="/tmp/c.mp4",
        ),
    )

    # When: Listing only front-door analyzed clips with alerts enabled
    clips, total = await state_store.list_clips(
        camera="front_door",
        status=ClipStatus.ANALYZED,
        alerted=True,
        offset=0,
        limit=10,
    )

    # Then: Only the matching clip is returned with hydrated clip_id metadata
    assert total == 1
    assert [clip.clip_id for clip in clips] == ["test-list-filters-a"]


@pytest.mark.asyncio
async def test_list_clips_applies_risk_and_activity_filters(
    state_store: PostgresStateStore,
) -> None:
    """list_clips should support risk_level and activity_type filters."""
    # Given: Clips with different analysis results and alert decisions
    await state_store.upsert(
        "test-list-analysis-a",
        ClipStateData(
            camera_name="front_door",
            status=ClipStatus.ANALYZED,
            local_path="/tmp/analysis-a.mp4",
            analysis_result=AnalysisResult(
                risk_level="high",
                activity_type="person",
                summary="person detected",
            ),
            alert_decision=AlertDecision(notify=True, notify_reason="risk_level=high"),
        ),
    )
    await state_store.upsert(
        "test-list-analysis-b",
        ClipStateData(
            camera_name="front_door",
            status=ClipStatus.ANALYZED,
            local_path="/tmp/analysis-b.mp4",
            analysis_result=AnalysisResult(
                risk_level="low",
                activity_type="person",
                summary="low-risk person",
            ),
            alert_decision=AlertDecision(notify=False, notify_reason="risk_level=low"),
        ),
    )
    await state_store.upsert(
        "test-list-analysis-c",
        ClipStateData(
            camera_name="front_door",
            status=ClipStatus.ANALYZED,
            local_path="/tmp/analysis-c.mp4",
            analysis_result=AnalysisResult(
                risk_level="high",
                activity_type="vehicle",
                summary="vehicle detected",
            ),
            alert_decision=AlertDecision(notify=True, notify_reason="risk_level=high"),
        ),
    )

    # When: Filtering by risk, activity type, and alerted state
    clips, total = await state_store.list_clips(
        risk_level="high",
        activity_type="person",
        alerted=True,
        offset=0,
        limit=10,
    )

    # Then: Only the matching clip is returned
    assert total == 1
    assert [clip.clip_id for clip in clips] == ["test-list-analysis-a"]


@pytest.mark.asyncio
async def test_count_alerts_since_counts_notification_sent_events(
    state_store: PostgresStateStore,
) -> None:
    """count_alerts_since should count notification_sent events only."""
    # Given: A stored clip and one notification_sent event
    clip_id = "test-alert-count-001"
    now = datetime.now(timezone.utc)
    await state_store.upsert(
        clip_id,
        ClipStateData(
            camera_name="front_door",
            status="analyzed",
            local_path="/tmp/alert-count.mp4",
        ),
    )
    event_store = state_store.create_event_store()
    await event_store.append(
        NotificationSentEvent(
            clip_id=clip_id,
            timestamp=now,
            notifier_name="mqtt",
            dedupe_key=clip_id,
            attempt=1,
        )
    )

    # When: Counting alerts over a past and future window
    recent_count = await state_store.count_alerts_since(now - timedelta(minutes=1))
    future_count = await state_store.count_alerts_since(now + timedelta(minutes=1))

    # Then: The event is counted in the past window only
    assert recent_count == 1
    assert future_count == 0
