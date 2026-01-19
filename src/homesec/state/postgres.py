"""SQLAlchemy implementation of StateStore and EventStore.

Supports both PostgreSQL and SQLite through the database abstraction layer.
For backwards compatibility, PostgresStateStore and PostgresEventStore are
provided as aliases to the new unified implementations.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Table,
    Text,
    and_,
    func,
    insert,
    or_,
    select,
)
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from homesec.db import DialectHelper, JSONType, create_async_engine_for_dsn
from homesec.interfaces import EventStore, StateStore
from homesec.models.clip import ClipStateData
from homesec.models.events import (
    AlertDecisionMadeEvent,
    ClipDeletedEvent,
    ClipLifecycleEvent,
    ClipRecheckedEvent,
    ClipRecordedEvent,
    FilterCompletedEvent,
    FilterFailedEvent,
    FilterStartedEvent,
    NotificationFailedEvent,
    NotificationSentEvent,
    UploadCompletedEvent,
    UploadFailedEvent,
    UploadStartedEvent,
    VLMCompletedEvent,
    VLMFailedEvent,
    VLMSkippedEvent,
    VLMStartedEvent,
)
from homesec.models.events import (
    ClipEvent as ClipEventModel,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Event type name -> Pydantic model class mapping
_EVENT_TYPE_MAP: dict[str, type[ClipEventModel]] = {
    "clip_recorded": ClipRecordedEvent,
    "clip_deleted": ClipDeletedEvent,
    "clip_rechecked": ClipRecheckedEvent,
    "upload_started": UploadStartedEvent,
    "upload_completed": UploadCompletedEvent,
    "upload_failed": UploadFailedEvent,
    "filter_started": FilterStartedEvent,
    "filter_completed": FilterCompletedEvent,
    "filter_failed": FilterFailedEvent,
    "vlm_started": VLMStartedEvent,
    "vlm_completed": VLMCompletedEvent,
    "vlm_failed": VLMFailedEvent,
    "vlm_skipped": VLMSkippedEvent,
    "alert_decision_made": AlertDecisionMadeEvent,
    "notification_sent": NotificationSentEvent,
    "notification_failed": NotificationFailedEvent,
}


# =============================================================================
# SQLAlchemy Models
# =============================================================================


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class ClipState(Base):
    """Current state snapshot (lightweight, fast queries).

    Uses JSONType which automatically adapts to JSONB for PostgreSQL
    and JSON for SQLite.
    """

    __tablename__ = "clip_states"

    clip_id: Mapped[str] = mapped_column(Text, primary_key=True)
    data: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Note: Functional indexes on JSON fields are created via Alembic migrations
    # using dialect-specific syntax (see alembic/versions/ for details).
    # Simple column indexes can still be defined here if needed.


class ClipEvent(Base):
    """Event history (append-only audit log).

    Uses JSONType which automatically adapts to JSONB for PostgreSQL
    and JSON for SQLite.
    """

    __tablename__ = "clip_events"

    # Use Integer with Identity() for cross-database autoincrement support.
    # SQLite requires INTEGER PRIMARY KEY for autoincrement; BigInteger doesn't work.
    # For most use cases, 32-bit Integer is sufficient for event IDs.
    id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    clip_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("clip_states.clip_id", ondelete="CASCADE"),
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    event_data: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)

    __table_args__ = (
        Index("idx_clip_events_clip_id", "clip_id"),
        Index("idx_clip_events_clip_id_id", "clip_id", "id"),
        Index("idx_clip_events_timestamp", "timestamp"),
        Index("idx_clip_events_type", "event_type"),
    )


# =============================================================================
# JSON Parsing Utilities
# =============================================================================


def _parse_json_payload(raw: object) -> dict[str, Any]:
    """Parse JSON payload from SQLAlchemy into a dict.

    Handles different representations that may come from different
    database drivers and configurations.

    Args:
        raw: Raw value from database (dict, str, or bytes)

    Returns:
        Parsed dictionary

    Raises:
        TypeError: If raw is not a supported type
    """
    match raw:
        case dict():
            return cast(dict[str, Any], raw)
        case str():
            return cast(dict[str, Any], json.loads(raw))
        case bytes() | bytearray():
            return cast(dict[str, Any], json.loads(raw.decode("utf-8")))
        case _:
            raise TypeError(f"Unsupported JSON payload type: {type(raw).__name__}")


# =============================================================================
# State Store Implementation
# =============================================================================


class SQLAlchemyStateStore(StateStore):
    """SQLAlchemy implementation of StateStore interface.

    Supports both PostgreSQL and SQLite through the DialectHelper abstraction.
    Implements graceful degradation: operations return None/False instead of
    raising when DB is unavailable.

    Example:
        # PostgreSQL
        store = SQLAlchemyStateStore("postgresql://user:pass@localhost/db")

        # SQLite (file-based)
        store = SQLAlchemyStateStore("sqlite:///data/homesec.db")

        # SQLite (in-memory, for testing)
        store = SQLAlchemyStateStore("sqlite:///:memory:")
    """

    def __init__(self, dsn: str) -> None:
        """Initialize state store.

        Args:
            dsn: Database connection string. Supported formats:
                 - PostgreSQL: postgresql://user:pass@host/db
                 - SQLite: sqlite:///path/to/db.sqlite or sqlite:///:memory:
        """
        self._dsn = dsn
        self._engine: AsyncEngine | None = None
        self._dialect: DialectHelper | None = None

    @property
    def dialect(self) -> DialectHelper | None:
        """Return the dialect helper, or None if not initialized."""
        return self._dialect

    async def initialize(self) -> bool:
        """Initialize connection pool and dialect helper.

        Note: Tables are created via Alembic migrations, not here.

        Returns:
            True if initialization succeeded, False otherwise
        """
        try:
            self._dialect = DialectHelper.from_dsn(self._dsn)
            self._engine = create_async_engine_for_dsn(self._dsn)

            # Verify connection works
            async with self._engine.connect() as conn:
                await conn.execute(select(1))

            logger.info(
                "SQLAlchemyStateStore initialized successfully (dialect=%s)",
                self._dialect.dialect_name,
            )
            return True
        except Exception as e:
            logger.error("Failed to initialize SQLAlchemyStateStore: %s", e, exc_info=True)
            if self._engine is not None:
                await self._engine.dispose()
            self._engine = None
            self._dialect = None
            return False

    async def upsert(self, clip_id: str, data: ClipStateData) -> None:
        """Insert or update clip state.

        Uses dialect-specific upsert (INSERT ... ON CONFLICT DO UPDATE).
        Raises on execution errors so callers can retry/log appropriately.
        """
        if self._engine is None or self._dialect is None:
            logger.warning("StateStore not initialized, skipping upsert for %s", clip_id)
            return

        json_data = data.model_dump(mode="json")
        table = cast(Table, ClipState.__table__)

        # Use dialect-specific insert for upsert support
        stmt = self._dialect.insert(table).values(
            clip_id=clip_id,
            data=json_data,
            updated_at=func.now(),
        )
        stmt = self._dialect.on_conflict_do_update(
            stmt,
            index_elements=["clip_id"],
            set_={"data": stmt.excluded.data, "updated_at": func.now()},
        )

        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def get(self, clip_id: str) -> ClipStateData | None:
        """Retrieve clip state.

        Graceful degradation: returns None if DB unavailable or error occurs.
        """
        if self._engine is None:
            logger.warning("StateStore not initialized, returning None for %s", clip_id)
            return None

        try:
            async with self._engine.connect() as conn:
                result = await conn.execute(
                    select(ClipState.data).where(ClipState.clip_id == clip_id)
                )
                raw = result.scalar_one_or_none()

            if raw is None:
                return None

            # Parse JSON and validate with Pydantic
            data_dict = _parse_json_payload(raw)
            return ClipStateData.model_validate(data_dict)
        except Exception as e:
            logger.error(
                "Failed to get clip state for %s: %s",
                clip_id,
                e,
                exc_info=True,
            )
            return None

    async def list_candidate_clips_for_cleanup(
        self,
        *,
        older_than_days: int | None,
        camera_name: str | None,
        batch_size: int,
        cursor: tuple[datetime, str] | None = None,
    ) -> list[tuple[str, ClipStateData, datetime]]:
        """List clip states to scan for cleanup.

        Uses keyset pagination (cursor) instead of OFFSET so that the caller can
        safely update rows (e.g., mark clips deleted) without skipping entries.

        Cursor is `(created_at, clip_id)` from the last row of the previous page.
        """
        if self._engine is None or self._dialect is None:
            logger.warning("StateStore not initialized, returning empty cleanup candidate list")
            return []

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if older_than_days is not None and older_than_days < 0:
            raise ValueError("older_than_days must be >= 0")

        # Use dialect helper for JSON field extraction
        status_expr = self._dialect.json_extract_text(ClipState.data, "status")
        camera_expr = self._dialect.json_extract_text(ClipState.data, "camera_name")

        conditions = [or_(status_expr.is_(None), status_expr != "deleted")]

        if camera_name is not None:
            conditions.append(camera_expr == camera_name)

        if older_than_days is not None:
            conditions.append(
                self._dialect.timestamp_older_than(ClipState.created_at, older_than_days)
            )

        if cursor is not None:
            after_created_at, after_clip_id = cursor
            conditions.append(
                or_(
                    ClipState.created_at > after_created_at,
                    and_(
                        ClipState.created_at == after_created_at,
                        ClipState.clip_id > after_clip_id,
                    ),
                )
            )

        query = (
            select(ClipState.clip_id, ClipState.data, ClipState.created_at)
            .where(and_(*conditions))
            .order_by(ClipState.created_at.asc(), ClipState.clip_id.asc())
            .limit(int(batch_size))
        )

        async with self._engine.connect() as conn:
            result = await conn.execute(query)
            rows = result.all()

        items: list[tuple[str, ClipStateData, datetime]] = []
        for row_clip_id, raw, created_at in rows:
            try:
                data_dict = _parse_json_payload(raw)
                state = ClipStateData.model_validate(data_dict)
            except Exception as exc:
                logger.warning(
                    "Failed parsing clip state for cleanup: %s error=%s",
                    row_clip_id,
                    exc,
                    exc_info=True,
                )
                continue

            items.append((row_clip_id, state, created_at))

        return items

    async def ping(self) -> bool:
        """Health check.

        Returns True if database is reachable, False otherwise.
        """
        if self._engine is None:
            return False

        try:
            async with self._engine.connect() as conn:
                await conn.execute(select(1))
            return True
        except Exception as e:
            logger.warning("Database ping failed: %s", e, exc_info=True)
            return False

    async def shutdown(self, timeout: float | None = None) -> None:
        """Close connection pool."""
        _ = timeout
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._dialect = None
            logger.info("SQLAlchemyStateStore closed")

    def is_retryable_error(self, exc: Exception) -> bool:
        """Check if an exception is a retryable database error.

        Delegates to DialectHelper for dialect-specific error classification.
        """
        if self._dialect is None:
            return False
        return self._dialect.is_retryable_error(exc)

    def create_event_store(self) -> SQLAlchemyEventStore | NoopEventStore:
        """Create an event store using the same engine, or a no-op fallback."""
        if self._engine is None or self._dialect is None:
            return NoopEventStore()
        return SQLAlchemyEventStore(self._engine, self._dialect)

    @staticmethod
    def _parse_state_data(raw: object) -> dict[str, Any]:
        """Parse JSON payload from SQLAlchemy into a dict.

        Provided for backwards compatibility with tests.
        """
        return _parse_json_payload(raw)


# =============================================================================
# Event Store Implementation
# =============================================================================


class SQLAlchemyEventStore(EventStore):
    """SQLAlchemy implementation of EventStore interface.

    Supports both PostgreSQL and SQLite through the DialectHelper abstraction.
    """

    def __init__(self, engine: AsyncEngine, dialect: DialectHelper) -> None:
        """Initialize with shared engine and dialect from StateStore.

        Args:
            engine: SQLAlchemy async engine
            dialect: DialectHelper for dialect-specific operations
        """
        self._engine = engine
        self._dialect = dialect

    async def append(self, event: ClipLifecycleEvent) -> None:
        """Append a single event."""
        try:
            async with self._engine.begin() as conn:
                table = cast(Table, ClipEvent.__table__)
                payload = {
                    "clip_id": event.clip_id,
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "event_data": event.model_dump(
                        mode="json",
                        exclude={"id", "event_type"},
                    ),
                }
                # Use standard insert - no conflict handling needed for append-only log
                await conn.execute(insert(table).values(**payload))
        except Exception as e:
            logger.error("Failed to append event: %s", e, exc_info=e)
            raise

    async def get_events(
        self,
        clip_id: str,
        after_id: int | None = None,
    ) -> list[ClipLifecycleEvent]:
        """Get all events for a clip, optionally after an event id."""
        try:
            query = select(ClipEvent.id, ClipEvent.event_type, ClipEvent.event_data).where(
                ClipEvent.clip_id == clip_id
            )
            if after_id is not None:
                query = query.where(ClipEvent.id > after_id)
            query = query.order_by(ClipEvent.id)

            async with self._engine.connect() as conn:
                result = await conn.execute(query)
                rows = result.all()

            events: list[ClipLifecycleEvent] = []
            for event_id, event_type, event_data in rows:
                event_dict = _parse_json_payload(event_data)
                event_dict.setdefault("event_type", event_type)
                event_dict["id"] = event_id
                event_cls = _EVENT_TYPE_MAP.get(event_type)
                if event_cls is None:
                    logger.warning("Unknown event type: %s", event_type)
                    continue
                event = event_cls.model_validate(event_dict)
                events.append(cast(ClipLifecycleEvent, event))

            return events
        except Exception as e:
            logger.error("Failed to get events for %s: %s", clip_id, e, exc_info=e)
            return []


# =============================================================================
# No-op Implementations (for graceful degradation)
# =============================================================================


class NoopEventStore(EventStore):
    """Event store that drops events (used when database is unavailable)."""

    async def append(self, event: ClipLifecycleEvent) -> None:
        """No-op: event is dropped."""
        return

    async def get_events(
        self,
        clip_id: str,
        after_id: int | None = None,
    ) -> list[ClipLifecycleEvent]:
        """No-op: returns empty list."""
        return []


class NoopStateStore(StateStore):
    """State store that drops writes and returns no data."""

    async def upsert(self, clip_id: str, data: ClipStateData) -> None:
        """No-op: data is dropped."""
        return

    async def get(self, clip_id: str) -> ClipStateData | None:
        """No-op: returns None."""
        return None

    async def list_candidate_clips_for_cleanup(
        self,
        *,
        older_than_days: int | None,
        camera_name: str | None,
        batch_size: int,
        cursor: tuple[datetime, str] | None = None,
    ) -> list[tuple[str, ClipStateData, datetime]]:
        """No-op: returns empty list."""
        _ = older_than_days
        _ = camera_name
        _ = batch_size
        _ = cursor
        return []

    async def shutdown(self, timeout: float | None = None) -> None:
        """No-op."""
        return

    async def ping(self) -> bool:
        """No-op: always returns False."""
        return False


# =============================================================================
# Backwards Compatibility
# =============================================================================

# Aliases for backwards compatibility with existing code
PostgresStateStore = SQLAlchemyStateStore
PostgresEventStore = SQLAlchemyEventStore


def is_retryable_pg_error(exc: Exception) -> bool:
    """Return True if the exception is likely a transient database error.

    This function is provided for backwards compatibility. New code should
    use SQLAlchemyStateStore.is_retryable_error() or DialectHelper.is_retryable_error().
    """
    # Create a PostgreSQL dialect helper for backwards-compatible behavior
    dialect = DialectHelper("postgresql")
    return dialect.is_retryable_error(exc)


def _normalize_async_dsn(dsn: str) -> str:
    """Normalize DSN to include async driver.

    This function is provided for backwards compatibility. New code should
    use DialectHelper.normalize_dsn().
    """
    return DialectHelper.normalize_dsn(dsn)
