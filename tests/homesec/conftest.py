"""Shared pytest fixtures for HomeSec tests.

Provides parametrized database fixtures that run tests against both
SQLite and PostgreSQL backends.
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from homesec.state.postgres import SQLAlchemyStateStore

# =============================================================================
# Database Backend Configuration
# =============================================================================

# Default PostgreSQL DSN for local Docker (matches docker-compose.postgres.yml)
DEFAULT_PG_DSN = "postgresql://homesec:homesec@localhost:5432/homesec"


def _should_skip_postgres() -> bool:
    """Check if PostgreSQL tests should be skipped."""
    return os.environ.get("SKIP_POSTGRES_TESTS", "0") == "1"


def _get_postgres_dsn() -> str | None:
    """Get PostgreSQL DSN from environment, or None if unavailable."""
    if _should_skip_postgres():
        return None
    return os.environ.get("TEST_DB_DSN", DEFAULT_PG_DSN)


# =============================================================================
# Parametrized Database Fixtures
# =============================================================================


@pytest.fixture(params=["sqlite", "postgresql"])
def db_backend(request: pytest.FixtureRequest) -> str:
    """Parametrize tests to run against both database backends.

    Tests are run against both SQLite (in-memory) and PostgreSQL.
    PostgreSQL tests can be skipped by setting SKIP_POSTGRES_TESTS=1.

    Returns:
        Database backend name ("sqlite" or "postgresql")
    """
    backend = request.param

    if backend == "postgresql":
        if _should_skip_postgres():
            pytest.skip("PostgreSQL tests disabled via SKIP_POSTGRES_TESTS=1")

        pg_dsn = _get_postgres_dsn()
        if pg_dsn is None:
            pytest.skip("TEST_DB_DSN not set, skipping PostgreSQL tests")

    return backend


@pytest.fixture
def db_dsn(db_backend: str) -> str:
    """Return appropriate DSN for the database backend.

    Args:
        db_backend: Either "sqlite" or "postgresql"

    Returns:
        Database connection string
    """
    if db_backend == "sqlite":
        # In-memory SQLite for fast testing
        return "sqlite+aiosqlite:///:memory:"
    else:
        pg_dsn = _get_postgres_dsn()
        assert pg_dsn is not None, "PostgreSQL DSN should be available"
    return pg_dsn


@pytest.fixture
def db_dsn_for_tests(db_backend: str, db_dsn: str, tmp_path: Path) -> str:
    """Return the DSN used by fixtures that need a consistent database."""
    if db_backend == "sqlite":
        return f"sqlite+aiosqlite:///{tmp_path / 'tests.db'}"
    return db_dsn


@pytest.fixture
async def state_store(db_dsn: str) -> AsyncGenerator[SQLAlchemyStateStore, None]:
    """Create and initialize a state store for testing.

    Works with both SQLite and PostgreSQL via the parametrized db_dsn fixture.
    Creates fresh tables for each test and cleans up afterward.

    Yields:
        Initialized SQLAlchemyStateStore instance
    """
    from sqlalchemy import delete

    from homesec.state.postgres import Base, ClipEvent, ClipState, SQLAlchemyStateStore

    store = SQLAlchemyStateStore(db_dsn)
    initialized = await store.initialize()
    assert initialized, f"Failed to initialize state store with {db_dsn}"

    # Create fresh tables for the test
    if store._engine is not None:
        async with store._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)

    yield store

    # Cleanup: delete test data
    if store._engine is not None:
        async with store._engine.begin() as conn:
            # Delete in correct order due to foreign key constraint
            await conn.execute(delete(ClipEvent).where(ClipEvent.clip_id.like("test%")))
            await conn.execute(delete(ClipState).where(ClipState.clip_id.like("test%")))

    await store.shutdown()


# =============================================================================
# Legacy Fixtures (for backwards compatibility)
# =============================================================================


@pytest.fixture
def postgres_dsn() -> str:
    """Return test Postgres DSN (requires local DB running).

    This fixture is provided for backwards compatibility.
    New tests should use the parametrized db_dsn fixture instead.
    """
    dsn = _get_postgres_dsn()
    if dsn is None:
        pytest.skip("PostgreSQL not available")
    return dsn


@pytest.fixture
async def clean_test_db(db_dsn_for_tests: str) -> AsyncGenerator[None, None]:
    """Clean up test data after each test.

    This fixture is provided for backwards compatibility.
    New tests should use the parametrized state_store fixture instead.
    """
    from sqlalchemy import delete

    from homesec.state.postgres import Base, ClipEvent, ClipState, SQLAlchemyStateStore

    store = SQLAlchemyStateStore(db_dsn_for_tests)
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


# =============================================================================
# Mock Fixtures
# =============================================================================


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
        source_type="rtsp",
    )
