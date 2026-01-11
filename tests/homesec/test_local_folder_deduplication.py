"""Comprehensive tests for LocalFolderSource duplicate prevention via clip_states."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from homesec.models.clip import ClipStateData
from homesec.models.filter import FilterResult
from homesec.models.source import LocalFolderSourceConfig
from homesec.sources.local_folder import LocalFolderSource
from homesec.state.postgres import PostgresStateStore


@pytest.mark.asyncio
async def test_new_file_is_emitted(tmp_path: Path, postgres_dsn: str, clean_test_db: None) -> None:
    """Test that a new file is detected and emitted."""
    # Given: LocalFolderSource with state_store
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()

    config = LocalFolderSourceConfig(
        watch_dir=str(tmp_path),
        poll_interval=0.1,
        stability_threshold_s=0.0,
    )
    source = LocalFolderSource(config, camera_name="test", state_store=state_store)

    detected_clips = []
    source.register_callback(lambda clip: detected_clips.append(clip))


    # When: Starting source and adding a new file
    await source.start()
    try:
        clip_file = tmp_path / "new_clip.mp4"
        clip_file.write_text("video data")

        # Wait for detection
        await asyncio.sleep(0.3)

        # Then: File should be detected
        assert len(detected_clips) == 1
        assert detected_clips[0].clip_id == "new_clip"

    finally:
        await source.shutdown()
        await state_store.shutdown()


@pytest.mark.asyncio
async def test_file_already_in_cache_not_emitted(
    tmp_path: Path, postgres_dsn: str, clean_test_db: None
) -> None:
    """Test that a file already in the in-memory cache is not reprocessed."""
    # Given: LocalFolderSource with a file already in cache
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()

    config = LocalFolderSourceConfig(
        watch_dir=str(tmp_path),
        poll_interval=0.1,
        stability_threshold_s=0.0,
    )
    source = LocalFolderSource(config, camera_name="test", state_store=state_store)

    detected_clips = []
    source.register_callback(lambda clip: detected_clips.append(clip))

    # Create file before starting
    clip_file = tmp_path / "cached_clip.mp4"
    clip_file.write_text("video data")

    # When: Starting source (file detected on first scan)
    await source.start()
    try:
        await asyncio.sleep(0.3)
        assert len(detected_clips) == 1

        # Clear detected clips
        detected_clips.clear()

        # Wait for another scan
        await asyncio.sleep(0.3)

        # Then: File should NOT be detected again (in cache)
        assert len(detected_clips) == 0

    finally:
        await source.shutdown()
        await state_store.shutdown()


@pytest.mark.asyncio
async def test_file_already_processed_not_emitted(
    tmp_path: Path, postgres_dsn: str, clean_test_db: None
) -> None:
    """Test that a file with existing clip_state is not reprocessed."""
    # Given: LocalFolderSource with state_store
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()

    # Create a clip_state for a file BEFORE starting source
    clip_file = tmp_path / "processed_clip.mp4"
    clip_file.write_text("video data")

    state = ClipStateData(
        camera_name="test",
        status="done",
        local_path=str(clip_file),
        filter_result=FilterResult(
            detected_classes=["person"],
            confidence=0.9,
            model="yolo",
            sampled_frames=10,
        ),
    )
    await state_store.upsert("processed_clip", state)

    # When: Starting source
    config = LocalFolderSourceConfig(
        watch_dir=str(tmp_path),
        poll_interval=0.1,
        stability_threshold_s=0.0,
    )
    source = LocalFolderSource(config, camera_name="test", state_store=state_store)

    detected_clips = []
    source.register_callback(lambda clip: detected_clips.append(clip))

    await source.start()
    try:
        # Wait for scan
        await asyncio.sleep(0.3)

        # Then: File should NOT be detected (clip_state exists)
        assert len(detected_clips) == 0

    finally:
        await source.shutdown()
        await state_store.shutdown()


@pytest.mark.asyncio
async def test_deleted_file_tombstone_prevents_reprocessing(
    tmp_path: Path, postgres_dsn: str, clean_test_db: None
) -> None:
    """Test that a file with status='deleted' (tombstone) is not reprocessed."""
    # Given: LocalFolderSource with state_store
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()

    # Create a tombstone (status='deleted') for a file
    clip_file = tmp_path / "deleted_clip.mp4"
    clip_file.write_text("video data")

    state = ClipStateData(
        camera_name="test",
        status="deleted",
        local_path=str(clip_file),
        filter_result=None,
    )
    await state_store.upsert("deleted_clip", state)

    # When: Starting source
    config = LocalFolderSourceConfig(
        watch_dir=str(tmp_path),
        poll_interval=0.1,
        stability_threshold_s=0.0,
    )
    source = LocalFolderSource(config, camera_name="test", state_store=state_store)

    detected_clips = []
    source.register_callback(lambda clip: detected_clips.append(clip))

    await source.start()
    try:
        # Wait for scan
        await asyncio.sleep(0.3)

        # Then: File should NOT be detected (tombstone prevents reprocessing)
        assert len(detected_clips) == 0

    finally:
        await source.shutdown()
        await state_store.shutdown()


@pytest.mark.asyncio
async def test_old_mtime_new_file_is_emitted(
    tmp_path: Path, postgres_dsn: str, clean_test_db: None
) -> None:
    """Test that a new file with an old mtime IS emitted (no watermark bug)."""
    # Given: LocalFolderSource with state_store
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()

    config = LocalFolderSourceConfig(
        watch_dir=str(tmp_path),
        poll_interval=0.1,
        stability_threshold_s=0.0,
    )
    source = LocalFolderSource(config, camera_name="test", state_store=state_store)

    detected_clips = []
    source.register_callback(lambda clip: detected_clips.append(clip))

    # When: Creating a file with an old mtime (simulating copied old footage)
    await source.start()
    try:
        clip_file = tmp_path / "old_mtime_clip.mp4"
        clip_file.write_text("video data")

        # Set mtime to 1 hour ago
        old_time = time.time() - 3600
        clip_file.touch()
        import os

        os.utime(clip_file, (old_time, old_time))

        # Wait for detection
        await asyncio.sleep(0.3)

        # Then: File SHOULD be detected (no clip_state exists, despite old mtime)
        assert len(detected_clips) == 1
        assert detected_clips[0].clip_id == "old_mtime_clip"

    finally:
        await source.shutdown()
        await state_store.shutdown()


@pytest.mark.asyncio
async def test_cache_eviction_with_db_check_prevents_reprocessing(
    tmp_path: Path, postgres_dsn: str, clean_test_db: None
) -> None:
    """Test that evicted files from cache are not reprocessed (DB check prevents it)."""
    # Given: LocalFolderSource with state_store and small cache size
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()

    config = LocalFolderSourceConfig(
        watch_dir=str(tmp_path),
        poll_interval=5.0,  # Long interval to ensure only one scan during test
        stability_threshold_s=0.0,
    )
    source = LocalFolderSource(config, camera_name="test", state_store=state_store)

    # Set small cache size to force eviction
    source._max_seen_files = 6  # Small cache for testing eviction

    detected_clips = []
    source.register_callback(lambda clip: detected_clips.append(clip))

    # Create 10 files BEFORE starting (so they're all scanned in one pass)
    for i in range(10):
        clip_file = tmp_path / f"file_{i:02d}.mp4"
        clip_file.write_text(f"video data {i}")

    # When: Starting source (will detect all 10 files in first scan)
    await source.start()
    try:
        await asyncio.sleep(0.5)  # Wait for first scan to complete

        # Then: All 10 files detected
        assert len(detected_clips) == 10

        # Cache should have evicted some entries (we processed 10 files but cache has max=6)
        # After evictions, we end up with fewer files than processed
        assert len(source._seen_files) < 10  # Eviction happened
        assert len(source._seen_files) <= source._max_seen_files  # Respects max

        # Shutdown source to simulate restart
        await source.shutdown()

        # Create clip_states for some files (simulating they were processed by pipeline)
        # Files 0-3 will have clip_states, files 4-9 won't
        for i in range(4):
            state = ClipStateData(
                camera_name="test",
                status="done",
                local_path=str(tmp_path / f"file_{i:02d}.mp4"),
                filter_result=None,
            )
            await state_store.upsert(f"file_{i:02d}", state)

        # Restart source with fresh cache (simulating app restart)
        source2 = LocalFolderSource(config, camera_name="test", state_store=state_store)
        detected_clips2 = []
        source2.register_callback(lambda clip: detected_clips2.append(clip))
        await source2.start()

        try:
            # Wait for first scan after restart
            await asyncio.sleep(0.5)

            # Then: Only files 4-9 should be detected (0-3 have clip_states, so skipped)
            # This proves DB check prevents reprocessing, even with empty cache
            assert len(detected_clips2) == 6
            clip_ids = sorted(clip.clip_id for clip in detected_clips2)
            assert clip_ids == [f"file_{i:02d}" for i in range(4, 10)]

        finally:
            await source2.shutdown()

    finally:
        await state_store.shutdown()


@pytest.mark.asyncio
async def test_no_state_store_falls_back_to_cache_only(tmp_path: Path) -> None:
    """Test that source works without state_store (cache-only mode)."""
    # Given: LocalFolderSource WITHOUT state_store
    config = LocalFolderSourceConfig(
        watch_dir=str(tmp_path),
        poll_interval=0.1,
        stability_threshold_s=0.0,
    )
    source = LocalFolderSource(config, camera_name="test", state_store=None)

    detected_clips = []
    source.register_callback(lambda clip: detected_clips.append(clip))

    await source.start()
    try:
        # When: Creating a new file
        clip_file = tmp_path / "no_db_clip.mp4"
        clip_file.write_text("video data")

        # Wait for detection
        await asyncio.sleep(0.3)

        # Then: File should be detected (cache allows it)
        assert len(detected_clips) == 1

        # Clear detected clips
        detected_clips.clear()

        # Wait for another scan
        await asyncio.sleep(0.3)

        # Then: File should NOT be detected again (in cache)
        assert len(detected_clips) == 0

    finally:
        await source.shutdown()


@pytest.mark.asyncio
async def test_multiple_files_some_seen_some_new(
    tmp_path: Path, postgres_dsn: str, clean_test_db: None
) -> None:
    """Test mixed scenario: some files new, some already processed."""
    # Given: LocalFolderSource with state_store
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()

    # Create clip_states for some files
    for i in [0, 2, 4]:
        state = ClipStateData(
            camera_name="test",
            status="done",
            local_path=str(tmp_path / f"mixed_{i}.mp4"),
            filter_result=None,
        )
        await state_store.upsert(f"mixed_{i}", state)

    # Create all files
    for i in range(5):
        clip_file = tmp_path / f"mixed_{i}.mp4"
        clip_file.write_text(f"video data {i}")

    # When: Starting source
    config = LocalFolderSourceConfig(
        watch_dir=str(tmp_path),
        poll_interval=0.1,
        stability_threshold_s=0.0,
    )
    source = LocalFolderSource(config, camera_name="test", state_store=state_store)

    detected_clips = []
    source.register_callback(lambda clip: detected_clips.append(clip))

    await source.start()
    try:
        # Wait for scan
        await asyncio.sleep(0.3)

        # Then: Only new files (1, 3) should be detected
        assert len(detected_clips) == 2
        clip_ids = sorted(clip.clip_id for clip in detected_clips)
        assert clip_ids == ["mixed_1", "mixed_3"]

    finally:
        await source.shutdown()
        await state_store.shutdown()


@pytest.mark.asyncio
async def test_db_unavailable_graceful_degradation(tmp_path: Path) -> None:
    """Test that source continues working when DB queries fail."""
    # Given: LocalFolderSource with state_store that will become unavailable
    from homesec.state.postgres import NoopStateStore

    state_store = NoopStateStore()  # Always returns None

    config = LocalFolderSourceConfig(
        watch_dir=str(tmp_path),
        poll_interval=0.1,
        stability_threshold_s=0.0,
    )
    source = LocalFolderSource(config, camera_name="test", state_store=state_store)

    detected_clips = []
    source.register_callback(lambda clip: detected_clips.append(clip))

    await source.start()
    try:
        # When: Creating a new file (DB unavailable, but cache works)
        clip_file = tmp_path / "degraded_clip.mp4"
        clip_file.write_text("video data")

        # Wait for detection
        await asyncio.sleep(0.3)

        # Then: File should be detected (fallback to cache-only mode)
        assert len(detected_clips) == 1

    finally:
        await source.shutdown()
