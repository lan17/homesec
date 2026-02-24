"""Tests for local retention pruning."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from homesec.models.clip import ClipStateData
from homesec.models.enums import ClipStatus
from homesec.repository import ClipRepository
from homesec.retention.pruner import LocalRetentionPruner
from tests.homesec.mocks import MockEventStore, MockStateStore


async def _seed_state(
    *,
    state_store: MockStateStore,
    clip_id: str,
    local_path: Path,
    created_at: datetime,
    storage_uri: str | None,
) -> None:
    state = ClipStateData(
        camera_name="front_door",
        status=ClipStatus.DONE,
        local_path=str(local_path),
        storage_uri=storage_uri,
    )
    await state_store.upsert(clip_id, state)
    state_store.created_at[clip_id] = created_at


@pytest.mark.asyncio
async def test_prune_deletes_oldest_until_under_limit(tmp_path: Path) -> None:
    # Given: Three uploaded local clips exceeding the retention byte limit
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_old = clips_dir / "clip-old.mp4"
    clip_mid = clips_dir / "clip-mid.mp4"
    clip_new = clips_dir / "clip-new.mp4"
    clip_old.write_bytes(b"a" * 60)
    clip_mid.write_bytes(b"b" * 50)
    clip_new.write_bytes(b"c" * 40)

    state_store = MockStateStore()
    event_store = MockEventStore()
    repository = ClipRepository(state_store, event_store)
    now = datetime.now()
    await _seed_state(
        state_store=state_store,
        clip_id="clip-old",
        local_path=clip_old,
        created_at=now - timedelta(minutes=2),
        storage_uri="mock://clip-old",
    )
    await _seed_state(
        state_store=state_store,
        clip_id="clip-mid",
        local_path=clip_mid,
        created_at=now - timedelta(minutes=1),
        storage_uri="mock://clip-mid",
    )
    await _seed_state(
        state_store=state_store,
        clip_id="clip-new",
        local_path=clip_new,
        created_at=now,
        storage_uri="mock://clip-new",
    )
    pruner = LocalRetentionPruner(
        repository=repository,
        local_clip_dirs=[clips_dir],
        max_local_size_bytes=90,
    )

    # When: A prune pass runs
    summary = await pruner.prune_once(reason="test")

    # Then: Oldest clip is deleted first and retention bytes are under the limit
    assert not clip_old.exists()
    assert clip_mid.exists()
    assert clip_new.exists()
    assert summary.deleted_files == 1
    assert summary.measured_local_files == 3
    assert summary.unmeasured_local_files == 0
    assert summary.measured_local_bytes == 150
    assert summary.non_eligible_local_bytes == 0
    assert summary.measurement_incomplete is False
    assert summary.blocked_over_limit is False
    assert summary.eligible_bytes_before == 150
    assert summary.eligible_bytes_after == 90
    assert summary.reclaimed_bytes == 60
    assert state_store.get_many_count == 1
    assert state_store.get_count == 0


@pytest.mark.asyncio
async def test_prune_respects_upload_and_inflight_gates(tmp_path: Path) -> None:
    # Given: One uploaded clip, one in-flight uploaded clip, and one local-only clip
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_uploaded = clips_dir / "clip-uploaded.mp4"
    clip_inflight = clips_dir / "clip-inflight.mp4"
    clip_local_only = clips_dir / "clip-local-only.mp4"
    clip_uploaded.write_bytes(b"a" * 20)
    clip_inflight.write_bytes(b"b" * 30)
    clip_local_only.write_bytes(b"c" * 40)

    state_store = MockStateStore()
    event_store = MockEventStore()
    repository = ClipRepository(state_store, event_store)
    now = datetime.now()
    await _seed_state(
        state_store=state_store,
        clip_id="clip-uploaded",
        local_path=clip_uploaded,
        created_at=now - timedelta(minutes=2),
        storage_uri="mock://clip-uploaded",
    )
    await _seed_state(
        state_store=state_store,
        clip_id="clip-inflight",
        local_path=clip_inflight,
        created_at=now - timedelta(minutes=1),
        storage_uri="mock://clip-inflight",
    )
    await _seed_state(
        state_store=state_store,
        clip_id="clip-local-only",
        local_path=clip_local_only,
        created_at=now,
        storage_uri=None,
    )
    pruner = LocalRetentionPruner(
        repository=repository,
        local_clip_dirs=[clips_dir],
        max_local_size_bytes=0,
    )
    before_upsert = state_store.upsert_count
    before_event_append = event_store.append_count

    # When: A prune pass runs with one clip marked in-flight
    summary = await pruner.prune_once(reason="test", in_flight_clip_ids={"clip-inflight"})

    # Then: Only uploaded non-in-flight clip is deleted and no repository writes occur
    assert not clip_uploaded.exists()
    assert clip_inflight.exists()
    assert clip_local_only.exists()
    assert summary.deleted_files == 1
    assert summary.skipped_in_flight == 1
    assert summary.skipped_not_uploaded == 1
    assert summary.measured_local_files == 3
    assert summary.unmeasured_local_files == 0
    assert summary.measured_local_bytes == 90
    assert summary.non_eligible_local_bytes == 70
    assert summary.measurement_incomplete is False
    assert summary.blocked_over_limit is True
    assert state_store.upsert_count == before_upsert
    assert event_store.append_count == before_event_append


@pytest.mark.asyncio
async def test_prune_uses_local_files_as_source_of_candidates(tmp_path: Path) -> None:
    # Given: A local file with mismatched state path and a DB-only uploaded state with no local file
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    discovered = clips_dir / "clip-a.mp4"
    discovered.write_bytes(b"a" * 10)
    db_only_path = clips_dir / "clip-db-only.mp4"
    mismatched_state_path = clips_dir / "different-name.mp4"

    state_store = MockStateStore()
    event_store = MockEventStore()
    repository = ClipRepository(state_store, event_store)
    now = datetime.now()
    await _seed_state(
        state_store=state_store,
        clip_id="clip-a",
        local_path=mismatched_state_path,
        created_at=now,
        storage_uri="mock://clip-a",
    )
    await _seed_state(
        state_store=state_store,
        clip_id="clip-db-only",
        local_path=db_only_path,
        created_at=now,
        storage_uri="mock://clip-db-only",
    )
    pruner = LocalRetentionPruner(
        repository=repository,
        local_clip_dirs=[clips_dir],
        max_local_size_bytes=0,
    )

    # When: A prune pass runs
    summary = await pruner.prune_once(reason="test")

    # Then: Only discovered local files are considered and path mismatches are skipped
    assert discovered.exists()
    assert summary.discovered_local_files == 1
    assert summary.measured_local_files == 1
    assert summary.unmeasured_local_files == 0
    assert summary.measured_local_bytes == 10
    assert summary.non_eligible_local_bytes == 10
    assert summary.measurement_incomplete is False
    assert summary.blocked_over_limit is True
    assert summary.eligible_candidates == 0
    assert summary.skipped_path_mismatch == 1
    assert summary.skipped_no_state == 0


@pytest.mark.asyncio
async def test_prune_tie_breaks_by_clip_id_for_equal_created_at(tmp_path: Path) -> None:
    # Given: Two equally old uploaded clips with equal created_at timestamps
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_a = clips_dir / "clip-a.mp4"
    clip_b = clips_dir / "clip-b.mp4"
    clip_a.write_bytes(b"a" * 40)
    clip_b.write_bytes(b"b" * 40)

    state_store = MockStateStore()
    repository = ClipRepository(state_store, MockEventStore())
    created_at = datetime.now()
    await _seed_state(
        state_store=state_store,
        clip_id="clip-a",
        local_path=clip_a,
        created_at=created_at,
        storage_uri="mock://clip-a",
    )
    await _seed_state(
        state_store=state_store,
        clip_id="clip-b",
        local_path=clip_b,
        created_at=created_at,
        storage_uri="mock://clip-b",
    )
    pruner = LocalRetentionPruner(
        repository=repository,
        local_clip_dirs=[clips_dir],
        max_local_size_bytes=40,
    )

    # When: A prune pass runs
    summary = await pruner.prune_once(reason="test")

    # Then: clip-a is removed first due to deterministic clip_id tie-break
    assert not clip_a.exists()
    assert clip_b.exists()
    assert summary.deleted_files == 1
    assert summary.measured_local_bytes == 80
    assert summary.eligible_bytes_after == 40


@pytest.mark.asyncio
async def test_prune_reports_incomplete_measurement_on_stat_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: Two local files where stat fails for one file during size collection
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_ok = clips_dir / "clip-ok.mp4"
    clip_broken = clips_dir / "clip-broken.mp4"
    clip_ok.write_bytes(b"a" * 50)
    clip_broken.write_bytes(b"b" * 60)

    state_store = MockStateStore()
    repository = ClipRepository(state_store, MockEventStore())
    now = datetime.now()
    await _seed_state(
        state_store=state_store,
        clip_id="clip-ok",
        local_path=clip_ok,
        created_at=now - timedelta(minutes=1),
        storage_uri="mock://clip-ok",
    )
    await _seed_state(
        state_store=state_store,
        clip_id="clip-broken",
        local_path=clip_broken,
        created_at=now,
        storage_uri="mock://clip-broken",
    )

    original_stat = Path.stat

    def _patched_stat(path_obj: Path, *args: object, **kwargs: object) -> object:
        if path_obj == clip_broken:
            raise OSError("simulated stat failure")
        return original_stat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _patched_stat)
    pruner = LocalRetentionPruner(
        repository=repository,
        local_clip_dirs=[clips_dir],
        max_local_size_bytes=0,
    )

    # When: A prune pass runs
    summary = await pruner.prune_once(reason="test")

    # Then: Summary marks measurement as incomplete and counts unmeasured files
    assert summary.discovered_local_files == 2
    assert summary.measured_local_files == 1
    assert summary.unmeasured_local_files == 1
    assert summary.measurement_incomplete is True
    assert summary.skipped_stat_error == 1
