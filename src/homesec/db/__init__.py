"""Database abstraction layer for SQLite and PostgreSQL support.

This package provides database-agnostic types and utilities that allow
the same SQLAlchemy models to work with both PostgreSQL and SQLite.

Key components:
- JSONType: Custom type that uses JSONB for PostgreSQL, JSON for SQLite
- DialectHelper: Encapsulates all dialect-specific SQL operations
- create_async_engine_for_dsn: Factory for creating properly configured engines

Example usage:
    from homesec.db import JSONType, DialectHelper, create_async_engine_for_dsn

    # In models - JSONType auto-adapts to dialect
    class MyModel(Base):
        data: Mapped[dict] = mapped_column(JSONType, nullable=False)

    # In stores - use DialectHelper for dialect-specific operations
    engine = create_async_engine_for_dsn(dsn)
    dialect = DialectHelper.from_engine(engine)
    json_expr = dialect.json_extract_text(MyModel.data, "status")
"""

from homesec.db.dialect import DialectHelper
from homesec.db.engine import create_async_engine_for_dsn, detect_dialect
from homesec.db.types import JSONType

__all__ = [
    "JSONType",
    "DialectHelper",
    "create_async_engine_for_dsn",
    "detect_dialect",
]
