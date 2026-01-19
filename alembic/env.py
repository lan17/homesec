"""Alembic migration environment configuration.

Supports both PostgreSQL and SQLite through the database abstraction layer.
"""

from __future__ import annotations

import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from alembic import context
from sqlalchemy import MetaData, pool
from sqlalchemy.ext.asyncio import async_engine_from_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from homesec.db import DialectHelper  # noqa: E402
from homesec.state.postgres import Base as StateBase  # noqa: E402
from homesec.telemetry.db.log_table import metadata as telemetry_metadata  # noqa: E402

# Combine all metadata into one for alembic
target_metadata = MetaData()
for table in telemetry_metadata.tables.values():
    table.to_metadata(target_metadata)
for table in StateBase.metadata.tables.values():
    table.to_metadata(target_metadata)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)


def _get_url() -> str:
    """Get database URL from environment or config."""
    url = os.getenv("DB_DSN") or os.getenv("DATABASE_URL") or config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError("Missing DB_DSN (or DATABASE_URL) for alembic migration.")
    return url


def _normalize_url(url: str) -> str:
    """Normalize URL to use appropriate async driver."""
    return DialectHelper.normalize_dsn(url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.
    """
    url = _normalize_url(_get_url())

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        # Enable batch mode for SQLite ALTER TABLE support
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Execute migrations with the given connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        # Enable batch mode for SQLite ALTER TABLE support
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate
    a connection with the context.
    """
    url = _normalize_url(_get_url())
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = url

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
