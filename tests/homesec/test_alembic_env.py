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
from homesec.repository.mobile_device_repository import MobileDeviceRepository, hash_apns_token


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


@pytest.mark.asyncio
async def test_mobile_revision_migration_preserves_existing_registrations(
    postgres_dsn: str,
) -> None:
    # Given: The already-applied registry migration with enabled and disabled device history
    repo_root = Path(__file__).resolve().parents[2]
    schema = f"hs_mobile_migration_{uuid.uuid4().hex[:8]}"
    env = os.environ.copy()
    env["DB_DSN"] = _plain_postgres_dsn(postgres_dsn)
    env[TEST_DB_SCHEMA_ENV] = schema
    env[TEST_DB_SCHEMA_ENABLE_ENV] = "1"
    engine = create_scoped_async_engine(postgres_dsn, schema=schema)

    def run_migration(command: str, revision: str) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "-c", "alembic.ini", command, revision],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr

    try:
        run_migration("upgrade", "7b0b1fbfc69b")
        async with engine.begin() as conn:
            for enabled in (True, False):
                token = f"synthetic-legacy-{enabled}"
                await conn.execute(
                    text(
                        "INSERT INTO mobile_devices "
                        "(id, platform, apns_token_hash, apns_token, apns_environment, bundle_id, "
                        "device_name, enabled) VALUES "
                        "(:id, 'ios', :hash, :token, 'sandbox', 'test.homesec', :name, :enabled)"
                    ),
                    {
                        "id": f"legacy-{enabled}",
                        "hash": hash_apns_token(token),
                        "token": token,
                        "name": "Existing iPhone",
                        "enabled": enabled,
                    },
                )

        # When: Applying the successor migration to existing registrations
        run_migration("upgrade", "head")

        # Then: History and enabled state survive, with a usable internal revision on each row
        repository = MobileDeviceRepository(engine)
        records = await repository.list_devices(include_disabled=True)
        assert {record.id: record.enabled for record in records} == {
            "legacy-True": True,
            "legacy-False": False,
        }
        assert all(record.device_name == "Existing iPhone" for record in records)
        assert all("registration_revision" not in record.model_dump() for record in records)
        targets = await repository.list_enabled_apns_targets(
            environment="sandbox", bundle_id="test.homesec"
        )
        assert [(target.id, target.revision) for target in targets] == [("legacy-True", 1)]
        async with engine.connect() as conn:
            revisions = (
                (await conn.execute(text("SELECT registration_revision FROM mobile_devices")))
                .scalars()
                .all()
            )
        assert revisions == [1, 1]

        # When: Rolling back only the new revision column
        run_migration("downgrade", "7b0b1fbfc69b")

        # Then: The prior registry schema remains usable and neither registration is deleted
        async with engine.connect() as conn:
            rows = (
                (await conn.execute(text("SELECT id, enabled FROM mobile_devices")))
                .mappings()
                .all()
            )
            revision_columns = await conn.scalar(
                text(
                    "SELECT count(*) FROM information_schema.columns "
                    "WHERE table_schema = current_schema() AND table_name = 'mobile_devices' "
                    "AND column_name = 'registration_revision'"
                )
            )
        assert {row["id"]: row["enabled"] for row in rows} == {
            "legacy-True": True,
            "legacy-False": False,
        }
        assert revision_columns == 0
    finally:
        await engine.dispose()
        await drop_schema_cascade(postgres_dsn, schema)
