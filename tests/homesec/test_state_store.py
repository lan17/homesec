"""Tests for SQLAlchemyStateStore.

These tests run against both SQLite and PostgreSQL backends via parametrized
fixtures. Use SKIP_POSTGRES_TESTS=1 to run only SQLite tests locally.
"""

from __future__ import annotations

import pytest

from homesec.db import DialectHelper
from homesec.models.clip import ClipStateData
from homesec.models.filter import FilterResult
from homesec.state import SQLAlchemyStateStore
from homesec.state.postgres import _normalize_async_dsn, _parse_json_payload

# =============================================================================
# Unit Tests (no database required)
# =============================================================================


def test_parse_json_payload_accepts_dict() -> None:
    """Test JSON parsing handles dicts."""
    # Given: A dict payload
    payload = {"status": "uploaded", "camera_name": "front_door"}

    # When: Parsing the payload
    result = _parse_json_payload(payload)

    # Then: Dict is returned as-is
    assert result == payload


def test_parse_json_payload_accepts_str() -> None:
    """Test JSON parsing handles JSON strings."""
    # Given: A JSON string
    json_str = '{"status": "uploaded", "camera_name": "front_door"}'

    # When: Parsing the JSON string
    result = _parse_json_payload(json_str)

    # Then: String is parsed to dict
    assert result == {"status": "uploaded", "camera_name": "front_door"}


def test_parse_json_payload_accepts_bytes() -> None:
    """Test JSON parsing handles bytes payloads."""
    # Given: A JSON payload as bytes
    json_bytes = b'{"status": "uploaded", "camera_name": "front_door"}'

    # When: Parsing bytes
    result = _parse_json_payload(json_bytes)

    # Then: Bytes are parsed to dict
    assert result == {"status": "uploaded", "camera_name": "front_door"}


def test_normalize_async_dsn_postgresql() -> None:
    """Test DSN normalization adds asyncpg driver for PostgreSQL."""
    # Given: Different Postgres DSN formats
    dsn_plain = "postgresql://user:pass@localhost/db"
    dsn_short = "postgres://user:pass@localhost/db"
    dsn_async = "postgresql+asyncpg://user:pass@localhost/db"

    # When: Normalizing DSNs
    norm_plain = _normalize_async_dsn(dsn_plain)
    norm_short = _normalize_async_dsn(dsn_short)
    norm_async = _normalize_async_dsn(dsn_async)

    # Then: asyncpg is used
    assert norm_plain == "postgresql+asyncpg://user:pass@localhost/db"
    assert norm_short == "postgresql+asyncpg://user:pass@localhost/db"
    assert norm_async == dsn_async


def test_normalize_async_dsn_sqlite() -> None:
    """Test DSN normalization adds aiosqlite driver for SQLite."""
    # Given: Different SQLite DSN formats
    dsn_plain = "sqlite:///test.db"
    dsn_memory = "sqlite:///:memory:"
    dsn_async = "sqlite+aiosqlite:///:memory:"

    # When: Normalizing DSNs
    norm_plain = _normalize_async_dsn(dsn_plain)
    norm_memory = _normalize_async_dsn(dsn_memory)
    norm_async = _normalize_async_dsn(dsn_async)

    # Then: aiosqlite is used
    assert norm_plain == "sqlite+aiosqlite:///test.db"
    assert norm_memory == "sqlite+aiosqlite:///:memory:"
    assert norm_async == dsn_async


def test_dialect_helper_detects_postgresql() -> None:
    """Test DialectHelper correctly detects PostgreSQL dialect."""
    # Given: A PostgreSQL DSN
    dsn = "postgresql://user:pass@localhost/db"

    # When: Creating DialectHelper
    helper = DialectHelper.from_dsn(dsn)

    # Then: Dialect is PostgreSQL
    assert helper.dialect_name == "postgresql"
    assert helper.is_postgres is True
    assert helper.is_sqlite is False


def test_dialect_helper_detects_sqlite() -> None:
    """Test DialectHelper correctly detects SQLite dialect."""
    # Given: A SQLite DSN
    dsn = "sqlite:///:memory:"

    # When: Creating DialectHelper
    helper = DialectHelper.from_dsn(dsn)

    # Then: Dialect is SQLite
    assert helper.dialect_name == "sqlite"
    assert helper.is_postgres is False
    assert helper.is_sqlite is True


# =============================================================================
# Integration Tests (parametrized for both backends)
# =============================================================================


def sample_state(clip_id: str = "test_clip_001") -> ClipStateData:
    """Create a sample ClipStateData for testing."""
    return ClipStateData(
        camera_name="front_door",
        status="queued_local",
        local_path="/tmp/test.mp4",
    )


@pytest.mark.asyncio
async def test_upsert_and_get_roundtrip(state_store: SQLAlchemyStateStore) -> None:
    """Test that upsert and get work correctly."""
    # Given: A clip state
    clip_id = "test_roundtrip_001"
    state = sample_state(clip_id)

    # When: Upserting and fetching
    await state_store.upsert(clip_id, state)
    retrieved = await state_store.get(clip_id)

    # Then: The roundtrip returns the state
    assert retrieved is not None
    assert retrieved.camera_name == "front_door"
    assert retrieved.status == "queued_local"


@pytest.mark.asyncio
async def test_upsert_updates_existing(state_store: SQLAlchemyStateStore) -> None:
    """Test that upsert updates existing records."""
    # Given: An existing clip state
    clip_id = "test_update_001"
    state = sample_state(clip_id)

    # When: Inserting and updating
    await state_store.upsert(clip_id, state)

    state.status = "uploaded"
    state.storage_uri = "dropbox:/front_door/test.mp4"
    await state_store.upsert(clip_id, state)

    # Then: The updated fields are persisted
    retrieved = await state_store.get(clip_id)
    assert retrieved is not None
    assert retrieved.status == "uploaded"
    assert retrieved.storage_uri == "dropbox:/front_door/test.mp4"


@pytest.mark.asyncio
async def test_get_returns_none_for_missing(state_store: SQLAlchemyStateStore) -> None:
    """Test that get returns None for non-existent clip_id."""
    # Given: A missing clip id
    # When: Retrieving a missing clip id
    result = await state_store.get("test_nonexistent_999")

    # Then: None is returned
    assert result is None


@pytest.mark.asyncio
async def test_ping_returns_true(state_store: SQLAlchemyStateStore) -> None:
    """Test that ping returns True when connected."""
    # Given: An initialized store
    # When: Ping is called
    result = await state_store.ping()

    # Then: Ping is True
    assert result is True


@pytest.mark.asyncio
async def test_graceful_degradation_uninitialized(db_backend: str) -> None:
    """Test graceful degradation when store is not initialized."""
    # Given: An uninitialized store with invalid DSN
    if db_backend == "postgresql":
        store = SQLAlchemyStateStore("postgresql://invalid:5432/nonexistent")
    else:
        # For SQLite, use a path that doesn't exist and can't be created
        store = SQLAlchemyStateStore("sqlite:////nonexistent/path/db.sqlite")

    # When: Operations are called without initialization
    await store.upsert("test_fail", sample_state())
    result = await store.get("test_fail")
    ping = await store.ping()

    # Then: Operations degrade gracefully
    assert result is None
    assert ping is False

    await store.shutdown()  # Should not raise


@pytest.mark.asyncio
async def test_initialize_returns_false_on_bad_dsn(db_backend: str) -> None:
    """Test that initialize returns False for invalid DSN."""
    # Given: An invalid DSN
    if db_backend == "postgresql":
        store = SQLAlchemyStateStore("postgresql://invalid:5432/nonexistent")
    else:
        store = SQLAlchemyStateStore("sqlite:////nonexistent/path/db.sqlite")

    # When: Initializing
    result = await store.initialize()

    # Then: Initialization fails
    assert result is False
    await store.shutdown()


@pytest.mark.asyncio
async def test_list_candidate_clips_for_cleanup_skips_deleted_and_filters_camera(
    state_store: SQLAlchemyStateStore,
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
async def test_event_store_append_and_get(state_store: SQLAlchemyStateStore) -> None:
    """Test that events can be appended and retrieved."""
    from datetime import datetime, timezone

    from homesec.models.events import ClipRecordedEvent

    # Given: A clip state and an event
    clip_id = "test_events_001"
    state = sample_state(clip_id)
    await state_store.upsert(clip_id, state)

    event_store = state_store.create_event_store()
    event = ClipRecordedEvent(
        clip_id=clip_id,
        timestamp=datetime.now(timezone.utc),
        camera_name="front_door",
        local_path="/tmp/test.mp4",
        duration_s=10.5,
        source_type="rtsp",
    )

    # When: Appending and retrieving the event
    await event_store.append(event)
    events = await event_store.get_events(clip_id)

    # Then: The event is retrieved
    assert len(events) == 1
    assert events[0].clip_id == clip_id
    assert events[0].event_type == "clip_recorded"


@pytest.mark.asyncio
async def test_dialect_is_set_after_initialization(state_store: SQLAlchemyStateStore) -> None:
    """Test that dialect helper is properly set after initialization."""
    # Given: An initialized state store
    # When: Checking the dialect
    dialect = state_store.dialect

    # Then: Dialect is set and valid
    assert dialect is not None
    assert dialect.dialect_name in ("postgresql", "sqlite")
