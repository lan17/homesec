"""Shared pytest fixtures for HomeSec tests."""

import asyncio
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add src to sys.path for imports
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path.resolve()) not in sys.path:
    sys.path.insert(0, str(src_path.resolve()))

import pytest

from homesec.models.clip import Clip
from homesec.postgres_support import (
    TEST_DB_SCHEMA_ENV,
    create_schema_if_missing,
    drop_schema_cascade,
    resolve_test_db_schema,
)
from homesec.sources.rtsp.capabilities import get_global_rtsp_timeout_capabilities
from tests.homesec.mocks import (
    MockFilter,
    MockNotifier,
    MockStateStore,
    MockStorage,
    MockVLM,
)


def _default_test_dsn() -> str:
    return os.getenv("TEST_DB_DSN", "postgresql://homesec:homesec@localhost:5432/homesec")


def _generate_test_schema_name() -> str:
    token = uuid.uuid4().hex[:8]
    return f"hs_pytest_{os.getpid()}_{token}"


@pytest.fixture(scope="session")
def isolated_postgres_schema() -> str:
    """Provision a per-run schema so parallel test runs can share one Postgres instance."""
    dsn = _default_test_dsn()
    previous_raw = os.environ.get(TEST_DB_SCHEMA_ENV)
    configured_schema = resolve_test_db_schema()
    created_by_fixture = configured_schema is None
    schema = _generate_test_schema_name() if created_by_fixture else configured_schema

    asyncio.run(create_schema_if_missing(dsn, schema))
    os.environ[TEST_DB_SCHEMA_ENV] = schema
    try:
        yield schema
    finally:
        try:
            if created_by_fixture:
                asyncio.run(drop_schema_cascade(dsn, schema))
        finally:
            if previous_raw is None:
                os.environ.pop(TEST_DB_SCHEMA_ENV, None)
            else:
                os.environ[TEST_DB_SCHEMA_ENV] = previous_raw


@pytest.fixture(autouse=True)
def reset_rtsp_timeout_capabilities() -> None:
    """Reset process-wide RTSP timeout capability cache between tests."""
    capabilities = get_global_rtsp_timeout_capabilities()
    capabilities.reset()
    yield
    capabilities.reset()


@pytest.fixture
def mock_filter() -> MockFilter:
    """Return a MockFilter with default config."""
    return MockFilter()


@pytest.fixture
def mock_vlm() -> MockVLM:
    """Return a MockVLM with default config."""
    return MockVLM()


@pytest.fixture
def mock_storage() -> MockStorage:
    """Return a MockStorage with default config."""
    return MockStorage()


@pytest.fixture
def mock_notifier() -> MockNotifier:
    """Return a MockNotifier with default config."""
    return MockNotifier()


@pytest.fixture
def mock_state_store() -> MockStateStore:
    """Return a MockStateStore with default config."""
    return MockStateStore()


@pytest.fixture
def sample_clip() -> Clip:
    """Return a sample Clip for testing."""
    now = datetime.now()
    return Clip(
        clip_id="test_clip_1234567890",
        camera_name="front_door",
        local_path=Path("/tmp/test_clip_1234567890.mp4"),
        start_ts=now,
        end_ts=now,
        duration_s=10.0,
        source_backend="rtsp",
    )


@pytest.fixture
def postgres_dsn(isolated_postgres_schema: str) -> str:
    """Return test Postgres DSN (requires local DB running)."""
    _ = isolated_postgres_schema
    return _default_test_dsn()


@pytest.fixture
async def clean_test_db(postgres_dsn: str) -> None:
    """Clean up test data after each test."""
    from sqlalchemy import delete

    from homesec.state.postgres import Base, ClipEvent, ClipState, PostgresStateStore

    store = PostgresStateStore(postgres_dsn)
    await store.initialize()
    if store._engine is not None:
        async with store._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)

    yield  # Run the test first

    # Cleanup after test
    if store._engine:
        async with store._engine.begin() as conn:
            # Delete in correct order due to foreign key
            await conn.execute(delete(ClipEvent).where(ClipEvent.clip_id.like("test-%")))
            await conn.execute(delete(ClipState).where(ClipState.clip_id.like("test-%")))
    await store.shutdown()
