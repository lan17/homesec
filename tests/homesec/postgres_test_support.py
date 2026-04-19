"""Shared Postgres helpers for tests."""

from __future__ import annotations

import os

from homesec.state.postgres import Base, PostgresStateStore

DEFAULT_TEST_DSN = "postgresql://homesec:homesec@localhost:5432/homesec"


def default_test_dsn() -> str:
    """Return the Docker-backed Postgres DSN used by server-facing tests."""
    return os.getenv("TEST_DB_DSN", DEFAULT_TEST_DSN)


async def reset_store_tables(store: PostgresStateStore) -> None:
    """Drop and recreate Postgres state tables for a clean test schema."""
    assert store._engine is not None
    async with store._engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
