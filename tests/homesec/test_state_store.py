"""Tests for PostgresStateStore."""

import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, select, text, update

import homesec.state.postgres as state_postgres
from homesec.models.alert import AlertDecision
from homesec.models.clip import ClipListCursor, ClipStateData
from homesec.models.enums import ClipStatus
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
async def test_get_many_with_created_at_roundtrip(state_store: PostgresStateStore) -> None:
    """Test that get_many_with_created_at returns a mapping for existing ids."""
    # Given: Two persisted clip states and one missing id
    clip_a = "test_many_created_at_a"
    clip_b = "test_many_created_at_b"
    await state_store.upsert(clip_a, sample_state(clip_a))
    await state_store.upsert(clip_b, sample_state(clip_b))

    # When: Batch retrieving by clip ids
    results = await state_store.get_many_with_created_at([clip_a, clip_b, "test_many_missing"])

    # Then: Existing ids are returned with state and created_at values
    assert set(results.keys()) == {clip_a, clip_b}
    state_a, created_at_a = results[clip_a]
    state_b, created_at_b = results[clip_b]
    assert state_a.camera_name == "front_door"
    assert state_b.camera_name == "front_door"
    assert state_a.clip_id == clip_a
    assert state_b.clip_id == clip_b
    assert state_a.created_at == created_at_a
    assert state_b.created_at == created_at_b
    assert created_at_a.tzinfo is not None
    assert created_at_b.tzinfo is not None


@pytest.mark.asyncio
async def test_get_many_with_created_at_chunks_large_id_sets(
    state_store: PostgresStateStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that get_many_with_created_at works correctly across multiple chunks."""
    # Given: More clip ids than the configured batch size
    monkeypatch.setattr(state_postgres, "_BATCH_STATE_LOOKUP_SIZE", 2)
    clip_ids = [f"test_many_chunk_{idx}" for idx in range(5)]
    for clip_id in clip_ids:
        await state_store.upsert(clip_id, sample_state(clip_id))

    # When: Batch retrieving all ids at once
    results = await state_store.get_many_with_created_at(clip_ids)

    # Then: All ids are returned even when query execution is chunked
    assert set(results.keys()) == set(clip_ids)
    for clip_id in clip_ids:
        state, created_at = results[clip_id]
        assert state.camera_name == "front_door"
        assert state.clip_id == clip_id
        assert state.created_at == created_at
        assert created_at.tzinfo is not None


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
async def test_mark_clip_deleted_preserves_existing_jsonb_fields(
    state_store: PostgresStateStore,
) -> None:
    """mark_clip_deleted should mutate status without clobbering unrelated JSONB fields."""
    # Given: A clip row with extra JSONB fields outside ClipStateData schema
    clip_id = "test-mark-deleted-preserve-001"
    await state_store.upsert(
        clip_id,
        ClipStateData(
            camera_name="front_door",
            status=ClipStatus.UPLOADED,
            local_path="/tmp/deleted-preserve.mp4",
        ),
    )
    assert state_store._engine is not None
    async with state_store._engine.begin() as conn:
        await conn.execute(
            text(
                """
                UPDATE clip_states
                SET data = data || jsonb_build_object(
                    'worker_metadata',
                    jsonb_build_object('pid', CAST(:worker_pid AS integer))
                )
                WHERE clip_id = :clip_id
                """
            ),
            {"clip_id": clip_id, "worker_pid": 42},
        )

    # When: Marking the clip as deleted
    deleted = await state_store.mark_clip_deleted(clip_id)

    # Then: Status is deleted and unrelated JSONB fields remain present
    assert deleted.status == ClipStatus.DELETED
    async with state_store._engine.connect() as conn:
        raw_data = (
            await conn.execute(select(ClipState.data).where(ClipState.clip_id == clip_id))
        ).scalar_one()
    assert isinstance(raw_data, dict)
    assert raw_data["status"] == ClipStatus.DELETED.value
    assert raw_data["worker_metadata"]["pid"] == 42


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
    page = await state_store.list_clips(
        camera="front_door",
        status=ClipStatus.ANALYZED,
        alerted=True,
        limit=10,
    )

    # Then: Only the matching clip is returned with hydrated clip_id metadata
    assert page.has_more is False
    assert page.next_cursor is None
    assert [clip.clip_id for clip in page.clips] == ["test-list-filters-a"]


@pytest.mark.asyncio
async def test_list_clips_alerted_false_includes_false_and_missing(
    state_store: PostgresStateStore,
) -> None:
    """list_clips should treat missing alert_decision as alerted=false."""
    # Given: Clips with true, false, and missing alert decisions
    await state_store.upsert(
        "test-alerted-false",
        ClipStateData(
            camera_name="front_door",
            status=ClipStatus.ANALYZED,
            local_path="/tmp/alerted-false.mp4",
            alert_decision=AlertDecision(notify=False, notify_reason="risk_level=low"),
        ),
    )
    await state_store.upsert(
        "test-alerted-missing",
        ClipStateData(
            camera_name="front_door",
            status=ClipStatus.ANALYZED,
            local_path="/tmp/alerted-missing.mp4",
        ),
    )
    await state_store.upsert(
        "test-alerted-true",
        ClipStateData(
            camera_name="front_door",
            status=ClipStatus.ANALYZED,
            local_path="/tmp/alerted-true.mp4",
            alert_decision=AlertDecision(notify=True, notify_reason="risk_level=high"),
        ),
    )

    # When: Listing with alerted=false
    page = await state_store.list_clips(alerted=False, limit=10)

    # Then: Both explicit false and missing alert decision clips are returned
    clip_ids = {clip.clip_id for clip in page.clips}
    assert clip_ids == {"test-alerted-false", "test-alerted-missing"}


@pytest.mark.asyncio
async def test_list_clips_detected_filter_distinguishes_present_and_missing(
    state_store: PostgresStateStore,
) -> None:
    """list_clips should support detected=true/false filtering via filter_result classes."""
    # Given: Clips with detected classes, empty classes, and no filter_result
    await state_store.upsert(
        "test-detected-true",
        ClipStateData(
            camera_name="front_door",
            status=ClipStatus.ANALYZED,
            local_path="/tmp/detected-true.mp4",
            filter_result=FilterResult(
                detected_classes=["person"],
                confidence=0.99,
                model="yolo",
                sampled_frames=1,
            ),
        ),
    )
    await state_store.upsert(
        "test-detected-empty",
        ClipStateData(
            camera_name="front_door",
            status=ClipStatus.ANALYZED,
            local_path="/tmp/detected-empty.mp4",
            filter_result=FilterResult(
                detected_classes=[],
                confidence=0.0,
                model="yolo",
                sampled_frames=1,
            ),
        ),
    )
    await state_store.upsert(
        "test-detected-missing",
        ClipStateData(
            camera_name="front_door",
            status=ClipStatus.ANALYZED,
            local_path="/tmp/detected-missing.mp4",
        ),
    )

    # When: Listing with detected=true
    detected_page = await state_store.list_clips(detected=True, limit=10)

    # Then: Only clips with at least one detected class are returned
    assert [clip.clip_id for clip in detected_page.clips] == ["test-detected-true"]

    # When: Listing with detected=false
    not_detected_page = await state_store.list_clips(detected=False, limit=10)

    # Then: Empty and missing detection payloads are both included
    assert {clip.clip_id for clip in not_detected_page.clips} == {
        "test-detected-empty",
        "test-detected-missing",
    }


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
    page = await state_store.list_clips(
        risk_level="high",
        activity_type="PERSON",
        alerted=True,
        limit=10,
    )

    # Then: Only the matching clip is returned with case-insensitive filter behavior
    assert page.has_more is False
    assert page.next_cursor is None
    assert [clip.clip_id for clip in page.clips] == ["test-list-analysis-a"]


@pytest.mark.asyncio
async def test_list_clips_excludes_deleted_by_default(
    state_store: PostgresStateStore,
) -> None:
    """list_clips should exclude deleted records unless status filter requests them."""
    # Given: One deleted clip and one active clip
    await state_store.upsert(
        "test-list-deleted-hidden",
        ClipStateData(
            camera_name="front_door",
            status=ClipStatus.DELETED,
            local_path="/tmp/deleted.mp4",
        ),
    )
    await state_store.upsert(
        "test-list-deleted-visible",
        ClipStateData(
            camera_name="front_door",
            status=ClipStatus.UPLOADED,
            local_path="/tmp/visible.mp4",
        ),
    )

    # When: Listing without a status filter
    default_page = await state_store.list_clips(limit=10)

    # Then: Deleted clip is excluded by default
    assert [clip.clip_id for clip in default_page.clips] == ["test-list-deleted-visible"]

    # When: Listing with explicit deleted status
    deleted_page = await state_store.list_clips(status=ClipStatus.DELETED, limit=10)

    # Then: Deleted clip is returned
    assert [clip.clip_id for clip in deleted_page.clips] == ["test-list-deleted-hidden"]


@pytest.mark.asyncio
async def test_list_clips_keyset_cursor_uses_created_at_and_clip_id(
    state_store: PostgresStateStore,
) -> None:
    """list_clips should page deterministically with keyset cursor and tie-break clip_id."""
    # Given: Three clips with deterministic timestamps including a tie
    tied_ts = datetime(2026, 2, 14, 3, 0, tzinfo=timezone.utc)
    older_ts = datetime(2026, 2, 14, 2, 59, tzinfo=timezone.utc)
    await state_store.upsert(
        "test-keyset-b",
        ClipStateData(
            camera_name="front_door",
            status=ClipStatus.UPLOADED,
            local_path="/tmp/keyset-b.mp4",
        ),
    )
    await state_store.upsert(
        "test-keyset-a",
        ClipStateData(
            camera_name="front_door",
            status=ClipStatus.UPLOADED,
            local_path="/tmp/keyset-a.mp4",
        ),
    )
    await state_store.upsert(
        "test-keyset-c",
        ClipStateData(
            camera_name="front_door",
            status=ClipStatus.UPLOADED,
            local_path="/tmp/keyset-c.mp4",
        ),
    )
    assert state_store._engine is not None
    async with state_store._engine.begin() as conn:
        await conn.execute(
            update(ClipState).where(ClipState.clip_id == "test-keyset-b").values(created_at=tied_ts)
        )
        await conn.execute(
            update(ClipState).where(ClipState.clip_id == "test-keyset-a").values(created_at=tied_ts)
        )
        await conn.execute(
            update(ClipState)
            .where(ClipState.clip_id == "test-keyset-c")
            .values(created_at=older_ts)
        )

    # When: Reading the first page with limit 2
    first_page = await state_store.list_clips(limit=2)

    # Then: Tie-break ordering is deterministic and cursor points to last row
    assert [clip.clip_id for clip in first_page.clips] == ["test-keyset-b", "test-keyset-a"]
    assert first_page.has_more is True
    assert first_page.next_cursor is not None
    assert first_page.next_cursor == ClipListCursor(created_at=tied_ts, clip_id="test-keyset-a")

    # When: Reading the next page with returned cursor
    second_page = await state_store.list_clips(limit=2, cursor=first_page.next_cursor)

    # Then: Remaining rows are returned and pagination ends
    assert [clip.clip_id for clip in second_page.clips] == ["test-keyset-c"]
    assert second_page.has_more is False
    assert second_page.next_cursor is None


@pytest.mark.asyncio
async def test_count_alerts_since_counts_alert_decision_at() -> None:
    """count_alerts_since should filter on alert_decision_at, not notification events."""

    class _FakeResult:
        def __init__(self, value: int) -> None:
            self._value = value

        def scalar(self) -> int:
            return self._value

    class _FakeConnection:
        def __init__(self) -> None:
            self.queries: list[object] = []

        async def __aenter__(self) -> "_FakeConnection":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = exc_type
            _ = exc
            _ = tb
            return None

        async def execute(self, query: object) -> _FakeResult:
            self.queries.append(query)
            return _FakeResult(1)

    class _FakeEngine:
        def __init__(self) -> None:
            self.connection = _FakeConnection()

        def connect(self) -> _FakeConnection:
            return self.connection

    # Given: A store backed by a fake engine
    store = PostgresStateStore("postgresql://unused")
    fake_engine = _FakeEngine()
    store._engine = fake_engine  # type: ignore[assignment]
    since = datetime.now(timezone.utc) - timedelta(minutes=5)

    # When: Counting alerts
    count = await store.count_alerts_since(since)

    # Then: The query is based on alert_decision_at and returns the database result
    assert count == 1
    assert len(fake_engine.connection.queries) == 1
    sql = str(fake_engine.connection.queries[0].compile(compile_kwargs={"literal_binds": True}))
    assert "alert_decision_at" in sql
    assert "alert_decision" in sql
    assert "notify" in sql
    assert "notification_sent" not in sql
