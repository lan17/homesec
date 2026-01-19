"""Database-agnostic type definitions.

This module provides custom SQLAlchemy types that automatically adapt
to the underlying database dialect, enabling the same model definitions
to work with both PostgreSQL and SQLite.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator, TypeEngine


class JSONType(TypeDecorator[dict[str, Any]]):
    """Database-agnostic JSON column type.

    Automatically uses the optimal JSON storage for each database:
    - PostgreSQL: JSONB (binary format, indexable, efficient queries)
    - SQLite: JSON (stored as TEXT, parsed by SQLAlchemy)

    This allows models to be defined once and work correctly with both
    databases without any code changes.

    Example:
        class ClipState(Base):
            data: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)

    The type is determined at connection time based on the dialect,
    so the same model class can be used with different databases.
    """

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[dict[str, Any]]:
        """Return the appropriate type implementation for the dialect.

        Called by SQLAlchemy when compiling SQL statements. Returns
        JSONB for PostgreSQL (with its superior indexing and operators)
        or standard JSON for other databases.

        Args:
            dialect: The SQLAlchemy dialect being used.

        Returns:
            TypeEngine appropriate for the dialect.
        """
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())
