"""State store implementations.

Provides StateStore and EventStore implementations that work with both
PostgreSQL and SQLite through the database abstraction layer.

For backwards compatibility, PostgresStateStore and PostgresEventStore
are still exported and work as aliases to the unified implementations.
"""

from homesec.state.postgres import (
    NoopEventStore,
    NoopStateStore,
    # Backwards compatibility aliases
    PostgresEventStore,
    PostgresStateStore,
    # New unified implementations
    SQLAlchemyEventStore,
    SQLAlchemyStateStore,
)

__all__ = [
    # New unified implementations (preferred)
    "SQLAlchemyStateStore",
    "SQLAlchemyEventStore",
    # Backwards compatibility (aliases to unified implementations)
    "PostgresStateStore",
    "PostgresEventStore",
    # No-op implementations for graceful degradation
    "NoopStateStore",
    "NoopEventStore",
]
