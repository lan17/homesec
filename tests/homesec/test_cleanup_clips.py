"""Tests for cleanup workflow CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from homesec.interfaces import ObjectFilter
from homesec.maintenance.cleanup_clips import CleanupOptions, run_cleanup
from homesec.models.clip import ClipStateData
from homesec.models.config import LocalStorageConfig
from homesec.models.filter import FilterOverrides, FilterResult
from homesec.plugins.storage.local import LocalStorage
from homesec.state.postgres import PostgresStateStore


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


def _write_cleanup_config(path: Path, *, dsn: str, storage_root: Path) -> None:
    config = {
        "version": 1,
        "cameras": [
            {
                "name": "front",
                "source": {
                    "type": "local_folder",
                    "config": {"watch_dir": "recordings", "poll_interval": 1.0},
                },
            }
        ],
        "storage": {
            "backend": "local",
            "local": {"root": str(storage_root)},
        },
        "state_store": {"dsn": dsn},
        "notifiers": [
            {
                "backend": "mqtt",
                "config": {"host": "localhost"},
            }
        ],
        "filter": {
            "plugin": "yolo",
            "config": {},
        },
        "vlm": {
            "backend": "openai",
            "trigger_classes": ["person"],
            "llm": {"api_key_env": "OPENAI_API_KEY", "model": "gpt-4o"},
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

    events = await state_store.create_event_store().get_events(clip_id)
    assert any(event.event_type == "clip_deleted" for event in events)

    await storage.shutdown()
    await state_store.shutdown()


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

    events = await state_store.create_event_store().get_events(clip_id)
    assert any(event.event_type == "clip_rechecked" for event in events)
    assert not any(event.event_type == "clip_deleted" for event in events)

    await storage.shutdown()
    await state_store.shutdown()
