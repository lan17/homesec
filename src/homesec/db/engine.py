"""Database engine factory and utilities.

This module provides factory functions for creating properly configured
SQLAlchemy async engines for both PostgreSQL and SQLite.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from homesec.db.dialect import DialectHelper, detect_dialect_from_dsn


def create_async_engine_for_dsn(dsn: str, **extra_kwargs: object) -> AsyncEngine:
    """Create an async SQLAlchemy engine with dialect-appropriate configuration.

    This factory function:
    1. Normalizes the DSN to use the correct async driver
    2. Applies dialect-specific engine configuration (pool size, etc.)
    3. Allows overriding configuration via extra_kwargs

    Args:
        dsn: Database connection string (PostgreSQL or SQLite)
        **extra_kwargs: Additional kwargs passed to create_async_engine,
                       these override the dialect defaults

    Returns:
        Configured AsyncEngine instance

    Raises:
        ValueError: If DSN dialect is not supported

    Example:
        # PostgreSQL
        engine = create_async_engine_for_dsn(
            "postgresql://user:pass@localhost/db"
        )

        # SQLite in-memory
        engine = create_async_engine_for_dsn("sqlite:///:memory:")

        # Override defaults
        engine = create_async_engine_for_dsn(
            "postgresql://...",
            pool_size=10,
            echo=True,
        )
    """
    # Detect dialect and get appropriate configuration
    dialect_name = detect_dialect_from_dsn(dsn)
    dialect = DialectHelper(dialect_name)

    # Normalize DSN to include async driver
    normalized_dsn = dialect.normalize_dsn(dsn)

    # Get dialect-specific engine kwargs
    engine_kwargs = dialect.get_engine_kwargs(normalized_dsn)

    # Allow caller to override defaults
    engine_kwargs.update(extra_kwargs)

    return create_async_engine(normalized_dsn, **engine_kwargs)


def detect_dialect(dsn: str) -> str:
    """Detect database dialect from a DSN string.

    Convenience re-export of detect_dialect_from_dsn for cleaner imports.

    Args:
        dsn: Database connection string

    Returns:
        Dialect name ("postgresql" or "sqlite")

    Raises:
        ValueError: If dialect cannot be detected
    """
    return detect_dialect_from_dsn(dsn)
