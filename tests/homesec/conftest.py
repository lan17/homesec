"""Shared pytest fixtures for HomeSec tests."""

import os
import socket
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

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
    dsn = os.getenv(
        "TEST_DB_DSN",
        os.getenv("DB_DSN", "postgresql://homesec:homesec@127.0.0.1:5432/homesec"),
    )

    if os.getenv("SKIP_POSTGRES_TESTS") == "1":
        pytest.skip("Skipping Postgres tests (SKIP_POSTGRES_TESTS=1)")

    if not _is_ci():
        host, port = _dsn_host_port(dsn)
        if host and not _can_connect(host, port):
            pytest.skip(f"Postgres not available at {host}:{port}")

    return dsn


@pytest.fixture
async def clean_test_db(postgres_dsn: str) -> None:
    """Clean up test data after each test."""
    from sqlalchemy import delete

    from homesec.state.postgres import Base, ClipEvent, ClipState, PostgresStateStore

    store = PostgresStateStore(postgres_dsn)
    try:
        initialized = await store.initialize()
    except Exception as exc:  # pragma: no cover - defensive
        if _is_ci():
            raise
        pytest.skip(f"Postgres not available: {exc}")
        return

    if not initialized:
        if _is_ci():
            raise AssertionError("Failed to initialize PostgresStateStore")
        pytest.skip("Postgres not available")
        return
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


def _is_ci() -> bool:
    return os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"


def _dsn_host_port(dsn: str) -> tuple[str | None, int]:
    parsed = urlsplit(dsn)
    host = parsed.hostname
    port = parsed.port or 5432
    return host, port


def _can_connect(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False
