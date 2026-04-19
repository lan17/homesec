from __future__ import annotations

import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from alembic import context
from sqlalchemy import pool, text
from sqlalchemy.ext.asyncio import async_engine_from_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sqlalchemy import MetaData  # noqa: E402

from homesec.postgres_support import (  # noqa: E402
    build_async_engine_kwargs,
    is_test_db_schema_enabled,
    normalize_async_dsn,
    resolve_test_db_schema,
    schema_ddl_identifier,
)
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
    fileConfig(config.config_file_name)


def _get_url() -> str:
    url = os.getenv("DB_DSN") or os.getenv("DATABASE_URL") or config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError("Missing DB_DSN (or DATABASE_URL) for alembic migration.")
    return normalize_async_dsn(url)


def run_migrations_offline() -> None:
    context.configure(
        url=_get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection, schema: str | None) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        version_table_schema=schema,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _get_url()
    schema = resolve_test_db_schema() if is_test_db_schema_enabled() else None
    engine_kwargs = build_async_engine_kwargs(schema=schema)

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        **engine_kwargs,
    )

    async with connectable.begin() as connection:
        if schema is not None:
            await connection.execute(
                text(f"CREATE SCHEMA IF NOT EXISTS {schema_ddl_identifier(schema)}")
            )
            await connection.execute(text(f"SET search_path TO {schema_ddl_identifier(schema)}"))
        await connection.run_sync(do_run_migrations, schema)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
