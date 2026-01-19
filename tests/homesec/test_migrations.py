"""Smoke tests for Alembic migrations on SQLite and PostgreSQL."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import MetaData

from homesec.db import create_async_engine_for_dsn
from homesec.models.clip import ClipStateData
from homesec.models.events import ClipRecordedEvent
from homesec.state.postgres import Base as StateBase
from homesec.state.postgres import SQLAlchemyStateStore
from homesec.telemetry.db.log_table import metadata as telemetry_metadata

_TARGET_METADATA = MetaData()
for table in telemetry_metadata.tables.values():
    table.to_metadata(_TARGET_METADATA)
for table in StateBase.metadata.tables.values():
    table.to_metadata(_TARGET_METADATA)

_SKIP_INDEX_NAMES = {"idx_clip_states_camera", "idx_clip_states_status"}


def _is_ignored_diff(diff: object) -> bool:
    candidate = diff
    if isinstance(diff, list) and len(diff) == 1 and isinstance(diff[0], tuple):
        candidate = diff[0]
    if not isinstance(candidate, tuple) or not candidate:
        return False
    if candidate[0] != "modify_default":
        return False
    _, _, table_name, column_name, *_ = candidate
    return table_name == "clip_events" and column_name == "id"


def _include_object(
    obj: object, name: str | None, type_: str, reflected: bool, compare_to: object
) -> bool:
    _ = obj
    _ = reflected
    _ = compare_to
    if type_ == "table" and name == "alembic_version":
        return False
    if type_ == "index" and name is not None and name.startswith("sqlite_autoindex"):
        return False
    if type_ == "index" and name in _SKIP_INDEX_NAMES:
        return False
    return True


@pytest.fixture
def migration_db_dsn(db_backend: str, db_dsn: str, tmp_path: Path) -> str:
    """Return a DSN suitable for Alembic migrations."""
    if db_backend == "sqlite":
        return f"sqlite+aiosqlite:///{tmp_path / 'alembic_test.db'}"
    return db_dsn


def _load_alembic_config() -> Config:
    root = Path(__file__).resolve().parents[2]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    return config


async def _reset_database(dsn: str) -> None:
    engine = create_async_engine_for_dsn(dsn)
    try:
        async with engine.begin() as conn:

            def _drop(sync_conn) -> None:
                _TARGET_METADATA.drop_all(bind=sync_conn, checkfirst=True)
                sync_conn.execute(sa.text("DROP TABLE IF EXISTS alembic_version"))

            await conn.run_sync(_drop)
    finally:
        await engine.dispose()


async def _assert_schema_matches_models(dsn: str) -> None:
    engine = create_async_engine_for_dsn(dsn)
    diffs: list[object] = []
    try:
        async with engine.connect() as conn:

            def _compare(sync_conn) -> list[object]:
                context = MigrationContext.configure(
                    connection=sync_conn,
                    opts={
                        "compare_type": True,
                        "compare_server_default": False,
                        "include_object": _include_object,
                    },
                )
                return compare_metadata(context, _TARGET_METADATA)

            diffs = await conn.run_sync(_compare)
    finally:
        await engine.dispose()

    diffs = [diff for diff in diffs if not _is_ignored_diff(diff)]
    assert diffs == []


async def _assert_basic_roundtrip(dsn: str) -> None:
    store = SQLAlchemyStateStore(dsn)
    try:
        initialized = await store.initialize()
        assert initialized is True

        clip_id = "test-migration-001"
        state = ClipStateData(
            camera_name="front_door",
            status="queued_local",
            local_path="/tmp/test.mp4",
        )
        await store.upsert(clip_id, state)

        loaded = await store.get(clip_id)
        assert loaded is not None
        assert loaded.camera_name == "front_door"

        event_store = store.create_event_store()
        event = ClipRecordedEvent(
            clip_id=clip_id,
            timestamp=datetime.now(),
            camera_name="front_door",
            duration_s=10.0,
            source_type="test",
        )
        await event_store.append(event)

        events = await event_store.get_events(clip_id)
        assert len(events) == 1
        assert events[0].event_type == "clip_recorded"
    finally:
        await store.shutdown()


def test_alembic_migrations_match_models(
    migration_db_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run Alembic migrations and verify model compatibility."""
    # Given: A clean database and alembic configuration
    monkeypatch.setenv("DB_DSN", migration_db_dsn)
    config = _load_alembic_config()
    asyncio.run(_reset_database(migration_db_dsn))

    # When: Running migrations to head
    command.upgrade(config, "head")

    # Then: Schema matches models and basic queries succeed
    asyncio.run(_assert_schema_matches_models(migration_db_dsn))
    asyncio.run(_assert_basic_roundtrip(migration_db_dsn))
