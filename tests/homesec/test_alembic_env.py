"""Tests for Alembic environment configuration."""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text

from homesec.postgres_support import (
    TEST_DB_SCHEMA_ENABLE_ENV,
    TEST_DB_SCHEMA_ENV,
    create_scoped_async_engine,
    drop_schema_cascade,
)


def _plain_postgres_dsn(dsn: str) -> str:
    """Convert an asyncpg SQLAlchemy DSN into the plain Postgres form."""
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
    return dsn


@pytest.mark.asyncio
async def test_alembic_upgrade_accepts_plain_postgres_dsn(postgres_dsn: str) -> None:
    """Alembic should accept the repo's documented plain Postgres DSN form."""
    # Given: A unique isolated schema and a plain Postgres DSN
    repo_root = Path(__file__).resolve().parents[2]
    schema = f"hs_alembic_{uuid.uuid4().hex[:8]}"
    env = os.environ.copy()
    env["DB_DSN"] = _plain_postgres_dsn(postgres_dsn)
    env[TEST_DB_SCHEMA_ENV] = schema
    env[TEST_DB_SCHEMA_ENABLE_ENV] = "1"

    try:
        # When: Running alembic upgrade head in that schema
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "-c", "alembic.ini", "upgrade", "head"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        # Then: Migration succeeds and creates the alembic version table in that schema
        assert result.returncode == 0, result.stderr

        engine = create_scoped_async_engine(postgres_dsn, schema=schema)
        try:
            async with engine.connect() as conn:
                version = await conn.scalar(text("SELECT version_num FROM alembic_version"))
            assert version is not None
        finally:
            await engine.dispose()
    finally:
        await drop_schema_cascade(postgres_dsn, schema)
