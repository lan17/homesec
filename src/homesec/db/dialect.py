"""Dialect-specific database operations.

This module encapsulates ALL database dialect differences in a single place,
providing a clean abstraction for the rest of the codebase. Instead of
scattering if/else dialect checks throughout the code, all dialect-specific
logic is centralized here.

The DialectHelper class provides methods for:
- JSON path extraction (JSONB operators vs json_extract)
- Date/time arithmetic (make_interval vs datetime modifiers)
- Upsert statements (dialect-specific INSERT ... ON CONFLICT)
- Retryable error detection (PostgreSQL vs SQLite error types)
- DSN normalization (adding appropriate async drivers)
- Engine configuration (pool settings per dialect)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy import ColumnElement, func
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.pool import StaticPool

if TYPE_CHECKING:
    from sqlalchemy import Table
    from sqlalchemy.dialects.postgresql import Insert as PgInsert
    from sqlalchemy.dialects.sqlite import Insert as SqliteInsert

logger = logging.getLogger(__name__)

# PostgreSQL SQLSTATE codes that indicate transient/retryable errors
_RETRYABLE_PG_SQLSTATES = frozenset(
    {
        "08000",  # connection_exception
        "08003",  # connection_does_not_exist
        "08006",  # connection_failure
        "08007",  # transaction_resolution_unknown
        "08001",  # sqlclient_unable_to_establish_sqlconnection
        "08004",  # sqlserver_rejected_establishment_of_sqlconnection
        "40P01",  # deadlock_detected
        "40001",  # serialization_failure
        "53300",  # too_many_connections
        "57P01",  # admin_shutdown
        "57P02",  # crash_shutdown
        "57P03",  # cannot_connect_now
    }
)

# SQLite error messages that indicate transient/retryable errors
_RETRYABLE_SQLITE_MESSAGES = frozenset(
    {
        "database is locked",
        "database is busy",
    }
)


class DialectHelper:
    """Encapsulates all database dialect-specific operations.

    This class provides a unified interface for operations that differ
    between PostgreSQL and SQLite, allowing the rest of the codebase
    to remain dialect-agnostic.

    Create an instance via the factory methods:
        dialect = DialectHelper.from_engine(engine)
        dialect = DialectHelper.from_dsn(dsn)
        dialect = DialectHelper(dialect_name)  # Direct construction

    Example usage:
        # JSON path extraction
        status_expr = dialect.json_extract_text(ClipState.data, "status")
        query = select(ClipState).where(status_expr == "uploaded")

        # Upsert
        stmt = dialect.insert(table).values(...)
        stmt = dialect.on_conflict_do_update(stmt, ["id"], {"data": new_data})
    """

    def __init__(self, dialect_name: str) -> None:
        """Initialize with dialect name.

        Args:
            dialect_name: Either "postgresql" or "sqlite"

        Raises:
            ValueError: If dialect_name is not supported
        """
        if dialect_name not in ("postgresql", "sqlite"):
            raise ValueError(f"Unsupported dialect: {dialect_name}")

        self._dialect_name = dialect_name
        self._is_postgres = dialect_name == "postgresql"
        self._is_sqlite = dialect_name == "sqlite"

    @classmethod
    def from_engine(cls, engine: AsyncEngine) -> DialectHelper:
        """Create DialectHelper from an async engine.

        Args:
            engine: SQLAlchemy async engine

        Returns:
            DialectHelper configured for the engine's dialect
        """
        return cls(engine.dialect.name)

    @classmethod
    def from_dsn(cls, dsn: str) -> DialectHelper:
        """Create DialectHelper by detecting dialect from DSN.

        Args:
            dsn: Database connection string

        Returns:
            DialectHelper configured for the DSN's dialect

        Raises:
            ValueError: If DSN dialect cannot be detected
        """
        dialect_name = detect_dialect_from_dsn(dsn)
        return cls(dialect_name)

    @property
    def dialect_name(self) -> str:
        """Return the dialect name ("postgresql" or "sqlite")."""
        return self._dialect_name

    @property
    def is_postgres(self) -> bool:
        """Return True if using PostgreSQL."""
        return self._is_postgres

    @property
    def is_sqlite(self) -> bool:
        """Return True if using SQLite."""
        return self._is_sqlite

    # -------------------------------------------------------------------------
    # JSON Operations
    # -------------------------------------------------------------------------

    def json_extract_text(
        self,
        column: ColumnElement[Any] | Any,  # Accept InstrumentedAttribute too
        *path: str,
    ) -> ColumnElement[str]:
        """Extract a text value from a JSON column.

        Creates a SQL expression that extracts a value from a JSON column
        as text, using the appropriate syntax for the dialect.

        Args:
            column: The JSON/JSONB column to extract from
            *path: Path components to the desired value (e.g., "status" or "nested", "key")

        Returns:
            SQL expression that evaluates to the extracted text value

        Example:
            # PostgreSQL: jsonb_extract_path_text(data, 'status')
            # SQLite: json_extract(data, '$.status')
            status = dialect.json_extract_text(ClipState.data, "status")
        """
        if not path:
            raise ValueError("At least one path component is required")

        if self._is_postgres:
            return func.jsonb_extract_path_text(column, *path)
        else:
            # SQLite uses JSON path notation: $.key.nested
            json_path = "$." + ".".join(path)
            return func.json_extract(column, json_path)

    # -------------------------------------------------------------------------
    # Date/Time Operations
    # -------------------------------------------------------------------------

    def timestamp_older_than(
        self,
        column: ColumnElement[Any] | Any,  # Accept InstrumentedAttribute too
        days: int,
    ) -> ColumnElement[bool]:
        """Create expression for 'column < now() - N days'.

        Args:
            column: Timestamp column to compare
            days: Number of days in the past

        Returns:
            SQL expression that is True if column is older than N days

        Example:
            # PostgreSQL: created_at < now() - make_interval(days => 7)
            # SQLite: created_at < datetime('now', '-7 days')
            old_clips = dialect.timestamp_older_than(ClipState.created_at, 7)
        """
        if days < 0:
            raise ValueError("days must be non-negative")

        if self._is_postgres:
            return column < func.now() - func.make_interval(days=days)
        else:
            # SQLite datetime modifier
            return column < func.datetime("now", f"-{days} days")

    def current_timestamp(self) -> ColumnElement[Any]:
        """Return expression for current timestamp.

        Returns:
            SQL expression for the current timestamp
        """
        if self._is_postgres:
            return func.now()
        else:
            return func.datetime("now")

    # -------------------------------------------------------------------------
    # Upsert Operations
    # -------------------------------------------------------------------------

    def insert(self, table: Table) -> PgInsert | SqliteInsert:
        """Create a dialect-specific INSERT statement.

        Args:
            table: The table to insert into

        Returns:
            Dialect-specific insert statement that supports on_conflict_do_update
        """
        if self._is_postgres:
            return postgresql.insert(table)
        else:
            return sqlite.insert(table)

    def on_conflict_do_update(
        self,
        stmt: PgInsert | SqliteInsert,
        index_elements: list[str],
        set_: dict[str, Any],
    ) -> PgInsert | SqliteInsert:
        """Add ON CONFLICT DO UPDATE clause to an insert statement.

        Args:
            stmt: Insert statement from dialect.insert()
            index_elements: Columns that form the unique constraint
            set_: Dictionary of column -> value for the UPDATE

        Returns:
            Insert statement with conflict handling added

        Example:
            stmt = dialect.insert(table).values(id=1, data={"key": "value"})
            stmt = dialect.on_conflict_do_update(
                stmt,
                index_elements=["id"],
                set_={"data": stmt.excluded.data, "updated_at": func.now()}
            )
        """
        return stmt.on_conflict_do_update(
            index_elements=index_elements,
            set_=set_,
        )

    # -------------------------------------------------------------------------
    # Error Classification
    # -------------------------------------------------------------------------

    def is_retryable_error(self, exc: Exception) -> bool:
        """Determine if an exception represents a transient, retryable error.

        Checks for connection errors, deadlocks, and other conditions that
        might succeed on retry.

        Args:
            exc: The exception to classify

        Returns:
            True if the error is likely transient and worth retrying
        """
        # SQLAlchemy OperationalError is generally retryable
        if isinstance(exc, OperationalError):
            return True

        # Connection invalidated
        if isinstance(exc, DBAPIError) and exc.connection_invalidated:
            return True

        if self._is_postgres:
            return self._is_retryable_postgres_error(exc)
        else:
            return self._is_retryable_sqlite_error(exc)

    def _is_retryable_postgres_error(self, exc: Exception) -> bool:
        """Check PostgreSQL-specific error codes."""
        sqlstate = _extract_sqlstate(exc)
        return sqlstate in _RETRYABLE_PG_SQLSTATES

    def _is_retryable_sqlite_error(self, exc: Exception) -> bool:
        """Check SQLite-specific error messages."""
        # Check the exception chain for SQLite errors
        current: BaseException | None = exc
        while current is not None:
            msg = str(current).lower()
            for retryable_msg in _RETRYABLE_SQLITE_MESSAGES:
                if retryable_msg in msg:
                    return True
            current = current.__cause__
        return False

    # -------------------------------------------------------------------------
    # Engine Configuration
    # -------------------------------------------------------------------------

    def get_engine_kwargs(self, dsn: str | None = None) -> dict[str, Any]:
        """Get dialect-appropriate engine configuration.

        Args:
            dsn: Optional database connection string. Used to detect
                in-memory SQLite for special pooling needs.

        Returns:
            Dictionary of kwargs for create_async_engine()
        """
        if self._is_postgres:
            return {
                "pool_size": 5,
                "max_overflow": 0,
                "pool_pre_ping": True,
            }
        # SQLite: simpler pooling, but in-memory needs a single shared connection.
        engine_kwargs: dict[str, Any] = {
            "pool_pre_ping": True,
        }
        if dsn is not None and self._is_sqlite and self._is_sqlite_memory_dsn(dsn):
            engine_kwargs.update(
                {
                    "poolclass": StaticPool,
                    "connect_args": {"check_same_thread": False},
                }
            )
        return engine_kwargs

    @staticmethod
    def _is_sqlite_memory_dsn(dsn: str) -> bool:
        """Return True if the DSN points at an in-memory SQLite database."""
        return ":memory:" in dsn

    # -------------------------------------------------------------------------
    # DSN Normalization
    # -------------------------------------------------------------------------

    @staticmethod
    def normalize_dsn(dsn: str) -> str:
        """Normalize DSN to include the appropriate async driver.

        Ensures the DSN uses the correct async driver:
        - PostgreSQL: asyncpg
        - SQLite: aiosqlite

        Args:
            dsn: Original database connection string

        Returns:
            DSN with async driver specified
        """
        # PostgreSQL normalization
        if dsn.startswith("postgresql://") and "+asyncpg" not in dsn:
            return dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
        if dsn.startswith("postgres://") and "+asyncpg" not in dsn:
            return dsn.replace("postgres://", "postgresql+asyncpg://", 1)

        # SQLite normalization
        if dsn.startswith("sqlite://") and "+aiosqlite" not in dsn:
            return dsn.replace("sqlite://", "sqlite+aiosqlite://", 1)

        return dsn


def detect_dialect_from_dsn(dsn: str) -> str:
    """Detect database dialect from a DSN string.

    Args:
        dsn: Database connection string

    Returns:
        Dialect name ("postgresql" or "sqlite")

    Raises:
        ValueError: If dialect cannot be detected from DSN
    """
    dsn_lower = dsn.lower()
    if dsn_lower.startswith(("postgresql://", "postgres://", "postgresql+asyncpg://")):
        return "postgresql"
    if dsn_lower.startswith(("sqlite://", "sqlite+aiosqlite://")):
        return "sqlite"
    raise ValueError(f"Cannot detect dialect from DSN: {dsn}")


def _extract_sqlstate(exc: BaseException) -> str | None:
    """Extract PostgreSQL SQLSTATE code from an exception."""
    for candidate in (exc, getattr(exc, "orig", None), getattr(exc, "__cause__", None)):
        if candidate is None:
            continue
        # Try different attribute names used by different drivers
        sqlstate = getattr(candidate, "sqlstate", None) or getattr(candidate, "pgcode", None)
        if sqlstate:
            return str(sqlstate)
    return None
