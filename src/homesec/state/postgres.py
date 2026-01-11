"""Postgres implementation of StateStore and EventStore."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, cast

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Table,
    Text,
    and_,
    func,
    or_,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

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

logger = logging.getLogger(__name__)

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


class Base(DeclarativeBase):
    pass


class ClipState(Base):
    """Current state snapshot (lightweight, fast queries)."""

    __tablename__ = "clip_states"

    clip_id: Mapped[str] = mapped_column(Text, primary_key=True)
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
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

    __table_args__ = (
        Index("idx_clip_states_status", func.jsonb_extract_path_text(data, "status")),
        Index("idx_clip_states_camera", func.jsonb_extract_path_text(data, "camera_name")),
    )


class ClipEvent(Base):
    """Event history (append-only audit log)."""

    __tablename__ = "clip_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
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
    event_data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    __table_args__ = (
        Index("idx_clip_events_clip_id", "clip_id"),
        Index("idx_clip_events_clip_id_id", "clip_id", "id"),
        Index("idx_clip_events_timestamp", "timestamp"),
        Index("idx_clip_events_type", "event_type"),
    )


def _normalize_async_dsn(dsn: str) -> str:
    if "+asyncpg" in dsn:
        return dsn
    if dsn.startswith("postgresql://"):
        return dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
    if dsn.startswith("postgres://"):
        return dsn.replace("postgres://", "postgresql+asyncpg://", 1)
    return dsn


class PostgresStateStore(StateStore):
    """Postgres implementation of StateStore interface.

    Implements graceful degradation: operations return None/False
    instead of raising when DB is unavailable.
    """

    def __init__(self, dsn: str) -> None:
        """Initialize state store.

        Args:
            dsn: Postgres connection string (e.g., "postgresql+asyncpg://user:pass@host/db")
        """
        self._dsn = _normalize_async_dsn(dsn)
        self._engine: AsyncEngine | None = None

    async def initialize(self) -> bool:
        """Initialize connection pool.

        Note: Tables are created via alembic migrations, not here.

        Returns:
            True if initialization succeeded, False otherwise
        """
        try:
            self._engine = create_async_engine(
                self._dsn,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=0,
            )
            # Verify connection works
            async with self._engine.connect() as conn:
                await conn.execute(select(1))
            logger.info("PostgresStateStore initialized successfully")
            return True
        except Exception as e:
            logger.error("Failed to initialize PostgresStateStore: %s", e, exc_info=True)
            if self._engine is not None:
                await self._engine.dispose()
            self._engine = None
            return False

    async def upsert(self, clip_id: str, data: ClipStateData) -> None:
        """Insert or update clip state.

        Raises on execution errors so callers can retry/log appropriately.
        """
        if self._engine is None:
            logger.warning("StateStore not initialized, skipping upsert for %s", clip_id)
            return

        json_data = data.model_dump(mode="json")
        table = cast(Table, ClipState.__table__)
        stmt = pg_insert(table).values(
            clip_id=clip_id,
            data=json_data,
            updated_at=func.now(),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[table.c.clip_id],
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
            data_dict = self._parse_state_data(raw)
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
        if self._engine is None:
            logger.warning("StateStore not initialized, returning empty cleanup candidate list")
            return []

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if older_than_days is not None and older_than_days < 0:
            raise ValueError("older_than_days must be >= 0")

        status_expr = func.jsonb_extract_path_text(ClipState.data, "status")
        camera_expr = func.jsonb_extract_path_text(ClipState.data, "camera_name")

        conditions = [or_(status_expr.is_(None), status_expr != "deleted")]
        if camera_name is not None:
            conditions.append(camera_expr == camera_name)
        if older_than_days is not None:
            conditions.append(
                ClipState.created_at < func.now() - func.make_interval(days=int(older_than_days))
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
        for clip_id, raw, created_at in rows:
            try:
                data_dict = self._parse_state_data(raw)
                state = ClipStateData.model_validate(data_dict)
            except Exception as exc:
                logger.warning(
                    "Failed parsing clip state for cleanup: %s error=%s",
                    clip_id,
                    exc,
                    exc_info=True,
                )
                continue

            items.append((clip_id, state, created_at))

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
            logger.info("PostgresStateStore closed")

    @staticmethod
    def _parse_state_data(raw: object) -> dict[str, Any]:
        """Parse JSONB payload from SQLAlchemy into a dict."""
        return _parse_jsonb_payload(raw)

    def create_event_store(self) -> PostgresEventStore | NoopEventStore:
        """Create a Postgres-backed event store or a no-op fallback."""
        if self._engine is None:
            return NoopEventStore()
        return PostgresEventStore(self._engine)


def _parse_jsonb_payload(raw: object) -> dict[str, Any]:
    """Parse JSONB payload from SQLAlchemy into a dict."""
    match raw:
        case dict():
            return cast(dict[str, Any], raw)
        case str():
            return cast(dict[str, Any], json.loads(raw))
        case bytes() | bytearray():
            return cast(dict[str, Any], json.loads(raw.decode("utf-8")))
        case _:
            raise TypeError(f"Unsupported JSONB payload type: {type(raw).__name__}")


_RETRYABLE_SQLSTATES = {
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


def _extract_sqlstate(exc: BaseException) -> str | None:
    for candidate in (exc, getattr(exc, "orig", None)):
        if candidate is None:
            continue
        sqlstate = getattr(candidate, "sqlstate", None) or getattr(candidate, "pgcode", None)
        if sqlstate:
            return str(sqlstate)
    return None


def is_retryable_pg_error(exc: Exception) -> bool:
    """Return True if the exception is likely a transient Postgres error."""
    if isinstance(exc, OperationalError):
        return True
    if isinstance(exc, DBAPIError) and exc.connection_invalidated:
        return True
    sqlstate = _extract_sqlstate(exc)
    return sqlstate in _RETRYABLE_SQLSTATES


class PostgresEventStore(EventStore):
    """Postgres implementation of EventStore interface."""

    def __init__(self, engine: AsyncEngine) -> None:
        """Initialize with shared engine from StateStore."""
        self._engine = engine

    async def append(self, event: ClipLifecycleEvent) -> None:
        """Append a single event."""
        try:
            async with self._engine.begin() as conn:
                table = cast(Any, ClipEvent.__table__)
                payload = {
                    "clip_id": event.clip_id,
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "event_data": event.model_dump(
                        mode="json",
                        exclude={"id", "event_type"},
                    ),
                }
                await conn.execute(pg_insert(table), [payload])
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
                event_dict = _parse_jsonb_payload(event_data)
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


class NoopEventStore(EventStore):
    """Event store that drops events (used when Postgres is unavailable)."""

    async def append(self, event: ClipLifecycleEvent) -> None:
        return

    async def get_events(
        self,
        clip_id: str,
        after_id: int | None = None,
    ) -> list[ClipLifecycleEvent]:
        return []


class NoopStateStore(StateStore):
    """State store that drops writes and returns no data."""

    async def upsert(self, clip_id: str, data: ClipStateData) -> None:
        return

    async def get(self, clip_id: str) -> ClipStateData | None:
        return None

    async def list_candidate_clips_for_cleanup(
        self,
        *,
        older_than_days: int | None,
        camera_name: str | None,
        batch_size: int,
        cursor: tuple[datetime, str] | None = None,
    ) -> list[tuple[str, ClipStateData, datetime]]:
        _ = older_than_days
        _ = camera_name
        _ = batch_size
        _ = cursor
        return []

    async def shutdown(self, timeout: float | None = None) -> None:
        return

    async def ping(self) -> bool:
        return False
