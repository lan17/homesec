"""Shared pytest fixtures for HomeSec tests."""

import sys
from datetime import datetime
from pathlib import Path

# Add src to sys.path for imports
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path.resolve()) not in sys.path:
    sys.path.insert(0, str(src_path.resolve()))

import pytest

from homesec.models.clip import Clip
from tests.homesec.mocks import (
    MockFilter,
    MockNotifier,
    MockStateStore,
    MockStorage,
    MockVLM,
)


@pytest.fixture(autouse=True)
def _enable_socket(socket_enabled) -> None:
    """Allow socket usage for tests that hit local Postgres."""
    _ = socket_enabled


@pytest.fixture
def enable_event_loop_debug() -> None:
    """Override HA plugin fixture to avoid HA event loop policy in core tests."""
    return None


@pytest.fixture
def verify_cleanup() -> None:
    """Override HA plugin cleanup verification for core tests."""
    return None


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
def postgres_dsn() -> str:
    """Return test Postgres DSN (requires local DB running)."""
    import os

    return os.getenv("TEST_DB_DSN", "postgresql://homesec:homesec@127.0.0.1:5432/homesec")


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
