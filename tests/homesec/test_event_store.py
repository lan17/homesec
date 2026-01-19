"""Tests for PostgresEventStore."""

from __future__ import annotations

from datetime import datetime

import pytest

from homesec.models.clip import ClipStateData
from homesec.models.events import (
    ClipDeletedEvent,
    ClipRecordedEvent,
    FilterCompletedEvent,
    UploadCompletedEvent,
)
from homesec.state.postgres import PostgresEventStore, PostgresStateStore


@pytest.mark.asyncio
async def test_append_and_get_events(db_dsn_for_tests: str, clean_test_db: None) -> None:
    # Given: A state store and event store with initialized tables
    state_store = PostgresStateStore(db_dsn_for_tests)
    await state_store.initialize()
    event_store = state_store.create_event_store()
    assert isinstance(event_store, PostgresEventStore)

    clip_id = "test-clip-001"

    # Create state first (foreign key requirement)
    state = ClipStateData(
        camera_name="front_door",
        status="queued_local",
        local_path="/tmp/test.mp4",
    )
    await state_store.upsert(clip_id, state)

    # When: We append multiple events
    events = [
        ClipRecordedEvent(
            clip_id=clip_id,
            timestamp=datetime.now(),
            camera_name="front_door",
            duration_s=10.0,
            source_type="test",
        ),
        UploadCompletedEvent(
            clip_id=clip_id,
            timestamp=datetime.now(),
            storage_uri="dropbox://test.mp4",
            view_url="https://dropbox.com/test.mp4",
            attempt=1,
            duration_ms=5000,
        ),
        FilterCompletedEvent(
            clip_id=clip_id,
            timestamp=datetime.now(),
            detected_classes=["person"],
            confidence=0.95,
            model="yolov8n",
            sampled_frames=10,
            attempt=1,
            duration_ms=2000,
        ),
    ]
    for event in events:
        await event_store.append(event)

    # Then: We can retrieve all events in order
    retrieved = await event_store.get_events(clip_id)
    assert len(retrieved) == 3
    assert retrieved[0].event_type == "clip_recorded"
    assert retrieved[1].event_type == "upload_completed"
    assert retrieved[2].event_type == "filter_completed"
    assert retrieved[0].id is not None
    assert retrieved[1].id is not None
    assert retrieved[2].id is not None
    assert retrieved[0].id < retrieved[1].id < retrieved[2].id

    # Cleanup
    await state_store.shutdown()


@pytest.mark.asyncio
async def test_append_and_get_clip_deleted_event(
    db_dsn_for_tests: str,
    clean_test_db: None,
) -> None:
    # Given: A state store and event store with initialized tables
    state_store = PostgresStateStore(db_dsn_for_tests)
    await state_store.initialize()
    event_store = state_store.create_event_store()
    assert isinstance(event_store, PostgresEventStore)

    clip_id = "test-clip-delete-001"

    # Create state first (foreign key requirement)
    state = ClipStateData(
        camera_name="front_door",
        status="queued_local",
        local_path="/tmp/test.mp4",
    )
    await state_store.upsert(clip_id, state)

    # When: We append a clip_deleted event
    event = ClipDeletedEvent(
        clip_id=clip_id,
        timestamp=datetime.now(),
        camera_name="front_door",
        reason="cleanup_cli",
        run_id="run-123",
        local_path="/tmp/test.mp4",
        storage_uri="dropbox:/homecam/front_door/test.mp4",
        deleted_local=True,
        deleted_storage=True,
    )
    await event_store.append(event)

    # Then: We can retrieve it and fields round-trip
    retrieved = await event_store.get_events(clip_id)
    assert len(retrieved) == 1
    assert retrieved[0].event_type == "clip_deleted"
    assert isinstance(retrieved[0], ClipDeletedEvent)
    assert retrieved[0].reason == "cleanup_cli"
    assert retrieved[0].deleted_local is True
    assert retrieved[0].deleted_storage is True

    # Cleanup
    await state_store.shutdown()


@pytest.mark.asyncio
async def test_get_events_after_id(db_dsn_for_tests: str, clean_test_db: None) -> None:
    # Given: A clip with multiple events
    state_store = PostgresStateStore(db_dsn_for_tests)
    await state_store.initialize()
    event_store = state_store.create_event_store()
    assert isinstance(event_store, PostgresEventStore)

    clip_id = "test-clip-002"

    # Create state first (foreign key requirement)
    state = ClipStateData(
        camera_name="front_door",
        status="queued_local",
        local_path="/tmp/test.mp4",
    )
    await state_store.upsert(clip_id, state)

    events = [
        ClipRecordedEvent(
            clip_id=clip_id,
            timestamp=datetime.now(),
            camera_name="front_door",
            duration_s=10.0,
            source_type="test",
        ),
        UploadCompletedEvent(
            clip_id=clip_id,
            timestamp=datetime.now(),
            storage_uri="dropbox://test.mp4",
            view_url=None,
            attempt=1,
            duration_ms=5000,
        ),
    ]
    for event in events:
        await event_store.append(event)

    # When: We query events after the first id
    all_events = await event_store.get_events(clip_id)
    assert all_events[0].id is not None
    retrieved = await event_store.get_events(clip_id, after_id=all_events[0].id)

    # Then: We only get events after that id
    assert len(retrieved) == 1
    assert retrieved[0].event_type == "upload_completed"

    # Cleanup
    await state_store.shutdown()


@pytest.mark.asyncio
async def test_events_for_nonexistent_clip(db_dsn_for_tests: str) -> None:
    # Given: An initialized event store
    state_store = PostgresStateStore(db_dsn_for_tests)
    await state_store.initialize()
    event_store = state_store.create_event_store()
    assert isinstance(event_store, PostgresEventStore)

    # When: We query events for a nonexistent clip
    retrieved = await event_store.get_events("nonexistent-clip")

    # Then: We get an empty list
    assert retrieved == []

    # Cleanup
    await state_store.shutdown()
