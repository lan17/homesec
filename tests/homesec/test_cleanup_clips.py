"""Tests for cleanup workflow CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from homesec.interfaces import ObjectFilter
from homesec.maintenance.cleanup_clips import CleanupOptions, run_cleanup
from homesec.models.clip import ClipStateData
from homesec.models.filter import FilterOverrides, FilterResult
from homesec.plugins.storage.local import LocalStorage, LocalStorageConfig
from homesec.state.postgres import (
    NoopEventStore,
    PostgresStateStore,
    create_event_store_for_postgres_state_store,
)


class _TestFilter(ObjectFilter):
    """Deterministic filter for cleanup tests."""

    def __init__(self, *, detect_on: set[str]) -> None:
        self._detect_on = detect_on
        self.shutdown_called = False

    async def detect(
        self, video_path: Path, overrides: FilterOverrides | None = None
    ) -> FilterResult:
        _ = overrides
        for token in self._detect_on:
            if token in video_path.name:
                return FilterResult(
                    detected_classes=["person"],
                    confidence=0.9,
                    model="test_filter",
                    sampled_frames=1,
                )
        return FilterResult(
            detected_classes=[],
            confidence=0.0,
            model="test_filter",
            sampled_frames=1,
        )

    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout
        self.shutdown_called = True

    async def ping(self) -> bool:
        """Health check - test filter is always healthy."""
        return not self.shutdown_called


class _CleanupStorage:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout
        self._calls.append("storage")


class _CleanupStateStore:
    _engine = object()

    def __init__(self, calls: list[str], *, initialize_ok: bool) -> None:
        self._calls = calls
        self._initialize_ok = initialize_ok

    async def initialize(self) -> bool:
        return self._initialize_ok

    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout
        self._calls.append("state")


class _CleanupEventStore(NoopEventStore):
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout
        self._calls.append("event")


def _write_cleanup_config(path: Path, *, dsn: str, storage_root: Path) -> None:
    config = {
        "version": 1,
        "cameras": [
            {
                "name": "front",
                "source": {
                    "backend": "local_folder",
                    "config": {"watch_dir": "recordings", "poll_interval": 1.0},
                },
            }
        ],
        "storage": {
            "backend": "local",
            "config": {"root": str(storage_root)},
        },
        "state_store": {"dsn": dsn},
        "notifiers": [
            {
                "backend": "mqtt",
                "config": {"host": "localhost"},
            }
        ],
        "filter": {
            "backend": "yolo",
            "config": {},
        },
        "vlm": {
            "backend": "openai",
            "trigger_classes": ["person"],
            "config": {"api_key_env": "OPENAI_API_KEY", "model": "gpt-4o"},
        },
        "alert_policy": {
            "backend": "default",
            "config": {"min_risk_level": "low"},
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))


@pytest.mark.asyncio
async def test_cleanup_deletes_empty_clips(
    postgres_dsn: str, tmp_path: Path, clean_test_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup removes clips that still have no detections."""
    # Given a clip state with empty filter result and local + storage copies
    storage_root = tmp_path / "storage"
    storage = LocalStorage(LocalStorageConfig(root=str(storage_root)))
    clip_id = "test-empty-001"
    local_path = tmp_path / f"{clip_id}.mp4"
    local_path.write_bytes(b"video")
    dest_path = f"clips/front/{clip_id}.mp4"
    upload = await storage.put_file(local_path, dest_path)

    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    empty_filter = FilterResult(
        detected_classes=[],
        confidence=0.0,
        model="mock",
        sampled_frames=1,
    )
    state = ClipStateData(
        camera_name="front",
        status="done",
        local_path=str(local_path),
        storage_uri=upload.storage_uri,
        view_url=upload.view_url,
        filter_result=empty_filter,
    )
    await state_store.upsert(clip_id, state)

    config_path = tmp_path / "config.yaml"
    _write_cleanup_config(config_path, dsn=postgres_dsn, storage_root=storage_root)

    filter_plugin = _TestFilter(detect_on=set())
    monkeypatch.setattr(
        "homesec.maintenance.cleanup_clips.load_filter",
        lambda *_: filter_plugin,
    )

    # When cleanup runs without dry_run
    opts = CleanupOptions(
        config_path=config_path,
        batch_size=10,
        workers=1,
        dry_run=False,
    )
    await run_cleanup(opts)

    # Then the clip is deleted from local and storage, and state is updated
    updated = await state_store.get(clip_id)
    assert updated is not None
    assert updated.status == "deleted"
    assert not local_path.exists()

    storage_path = Path(upload.storage_uri.split("local:", 1)[1])
    assert not storage_path.exists()

    event_store = create_event_store_for_postgres_state_store(state_store)
    events = await event_store.get_events(clip_id)
    assert any(event.event_type == "clip_deleted" for event in events)

    await storage.shutdown()
    await state_store.shutdown()


@pytest.mark.asyncio
async def test_cleanup_releases_storage_and_state_when_postgres_init_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup should not leak resources when fail-fast Postgres init fails."""
    # Given: Cleanup storage is created but Postgres initialization degrades
    config_path = tmp_path / "config.yaml"
    _write_cleanup_config(
        config_path,
        dsn="postgresql://user:pass@localhost/db",
        storage_root=tmp_path / "storage",
    )
    calls: list[str] = []
    storage = _CleanupStorage(calls)
    state_store = _CleanupStateStore(calls, initialize_ok=False)
    monkeypatch.setattr("homesec.maintenance.cleanup_clips.load_storage_plugin", lambda _: storage)
    monkeypatch.setattr(
        "homesec.maintenance.cleanup_clips.PostgresStateStore",
        lambda _dsn: state_store,
    )

    def _fail_if_filter_loads(_: object) -> object:
        raise AssertionError("filter should not load after Postgres init failure")

    monkeypatch.setattr("homesec.maintenance.cleanup_clips.load_filter", _fail_if_filter_loads)

    # When/Then: Cleanup fails fast and releases resources acquired before failure
    with pytest.raises(RuntimeError, match="Failed to initialize Postgres state store"):
        await run_cleanup(CleanupOptions(config_path=config_path))

    assert calls == ["storage", "state"]


@pytest.mark.asyncio
async def test_cleanup_releases_persistence_when_filter_composition_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup should unwind explicit persistence wiring if later composition fails."""
    # Given: Cleanup reaches event-store wiring before filter creation fails
    config_path = tmp_path / "config.yaml"
    _write_cleanup_config(
        config_path,
        dsn="postgresql://user:pass@localhost/db",
        storage_root=tmp_path / "storage",
    )
    calls: list[str] = []
    storage = _CleanupStorage(calls)
    state_store = _CleanupStateStore(calls, initialize_ok=True)
    event_store = _CleanupEventStore(calls)
    monkeypatch.setattr("homesec.maintenance.cleanup_clips.load_storage_plugin", lambda _: storage)
    monkeypatch.setattr(
        "homesec.maintenance.cleanup_clips.PostgresStateStore",
        lambda _dsn: state_store,
    )
    monkeypatch.setattr(
        "homesec.maintenance.cleanup_clips.create_event_store_for_postgres_state_store",
        lambda _state_store: event_store,
    )

    def _raise_filter_failure(_: object) -> object:
        raise RuntimeError("filter setup failed")

    monkeypatch.setattr("homesec.maintenance.cleanup_clips.load_filter", _raise_filter_failure)

    # When/Then: Later composition failure unwinds event store, storage, and state store
    with pytest.raises(RuntimeError, match="filter setup failed"):
        await run_cleanup(CleanupOptions(config_path=config_path))

    assert calls == ["event", "storage", "state"]


@pytest.mark.asyncio
async def test_cleanup_marks_false_negatives(
    postgres_dsn: str, tmp_path: Path, clean_test_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup should recheck clips and skip delete when detections appear."""
    # Given a clip state that is empty on first pass
    storage_root = tmp_path / "storage"
    storage = LocalStorage(LocalStorageConfig(root=str(storage_root)))
    clip_id = "test-detect-001"
    local_path = tmp_path / f"{clip_id}.mp4"
    local_path.write_bytes(b"video")
    dest_path = f"clips/front/{clip_id}.mp4"
    upload = await storage.put_file(local_path, dest_path)

    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    empty_filter = FilterResult(
        detected_classes=[],
        confidence=0.0,
        model="mock",
        sampled_frames=1,
    )
    state = ClipStateData(
        camera_name="front",
        status="done",
        local_path=str(local_path),
        storage_uri=upload.storage_uri,
        view_url=upload.view_url,
        filter_result=empty_filter,
    )
    await state_store.upsert(clip_id, state)

    config_path = tmp_path / "config.yaml"
    _write_cleanup_config(config_path, dsn=postgres_dsn, storage_root=storage_root)

    filter_plugin = _TestFilter(detect_on={"detect"})
    monkeypatch.setattr(
        "homesec.maintenance.cleanup_clips.load_filter",
        lambda *_: filter_plugin,
    )

    # When cleanup runs without dry_run
    opts = CleanupOptions(
        config_path=config_path,
        batch_size=10,
        workers=1,
        dry_run=False,
    )
    await run_cleanup(opts)

    # Then the clip remains and state is updated with detections
    updated = await state_store.get(clip_id)
    assert updated is not None
    assert updated.status != "deleted"
    assert updated.filter_result is not None
    assert updated.filter_result.detected_classes == ["person"]
    assert local_path.exists()

    storage_path = Path(upload.storage_uri.split("local:", 1)[1])
    assert storage_path.exists()

    event_store = create_event_store_for_postgres_state_store(state_store)
    events = await event_store.get_events(clip_id)
    assert any(event.event_type == "clip_rechecked" for event in events)
    assert not any(event.event_type == "clip_deleted" for event in events)

    await storage.shutdown()
    await state_store.shutdown()
