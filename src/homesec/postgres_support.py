"""Shared Postgres helpers for async engines and test schema isolation."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

TEST_DB_SCHEMA_ENV = "HOMESEC_TEST_DB_SCHEMA"
_SCHEMA_PATTERN = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def normalize_async_dsn(dsn: str) -> str:
    """Normalize Postgres DSNs to the asyncpg SQLAlchemy dialect."""
    if "+asyncpg" in dsn:
        return dsn
    if dsn.startswith("postgresql://"):
        return dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
    if dsn.startswith("postgres://"):
        return dsn.replace("postgres://", "postgresql+asyncpg://", 1)
    return dsn


def resolve_test_db_schema(env: Mapping[str, str] | None = None) -> str | None:
    """Return the configured test schema, if any."""
    configured = (os.environ if env is None else env).get(TEST_DB_SCHEMA_ENV)
    if configured is None or configured == "":
        return None
    return validate_schema_name(configured)


def validate_schema_name(schema: str) -> str:
    """Validate a Postgres schema identifier used for test isolation."""
    if not _SCHEMA_PATTERN.fullmatch(schema):
        raise ValueError(
            f"Invalid Postgres schema {schema!r}; expected /{_SCHEMA_PATTERN.pattern}/"
        )
    return schema


def build_async_engine_kwargs(
    *,
    schema: str | None = None,
    engine_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build SQLAlchemy engine kwargs with an optional schema search path."""
    kwargs = dict(engine_kwargs or {})
    target_schema = resolve_test_db_schema() if schema is None else validate_schema_name(schema)
    if target_schema is None:
        return kwargs

    connect_args = dict(kwargs.get("connect_args", {}))
    server_settings = dict(connect_args.get("server_settings", {}))
    existing_search_path = server_settings.get("search_path")
    if existing_search_path is not None and existing_search_path != target_schema:
        raise ValueError(
            "engine_kwargs.connect_args.server_settings.search_path conflicts with schema"
        )
    server_settings["search_path"] = target_schema
    connect_args["server_settings"] = server_settings
    kwargs["connect_args"] = connect_args
    return kwargs


def create_scoped_async_engine(
    dsn: str,
    *,
    schema: str | None = None,
    **engine_kwargs: Any,
) -> AsyncEngine:
    """Create an async engine scoped to the configured schema when present."""
    return create_async_engine(
        normalize_async_dsn(dsn),
        **build_async_engine_kwargs(schema=schema, engine_kwargs=engine_kwargs),
    )


def schema_ddl_identifier(schema: str) -> str:
    """Return a quoted schema identifier for DDL statements."""
    return f'"{validate_schema_name(schema)}"'


async def create_schema_if_missing(dsn: str, schema: str) -> None:
    """Create a schema for an isolated test run if it does not exist."""
    engine = create_async_engine(normalize_async_dsn(dsn))
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_ddl_identifier(schema)}"))
    finally:
        await engine.dispose()


async def drop_schema_cascade(dsn: str, schema: str) -> None:
    """Drop an isolated test schema and everything inside it."""
    engine = create_async_engine(normalize_async_dsn(dsn))
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(f"DROP SCHEMA IF EXISTS {schema_ddl_identifier(schema)} CASCADE")
            )
    finally:
        await engine.dispose()
