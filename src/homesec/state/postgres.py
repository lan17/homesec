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
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from homesec.interfaces import EventStore, StateStore
from homesec.models.clip import ClipListCursor, ClipListPage, ClipStateData
from homesec.models.enums import ClipStatus, EventType
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

# Map EventType enum to event model classes
# Using enum values ensures consistency with event models
_EVENT_TYPE_MAP: dict[str, type[ClipEventModel]] = {
    EventType.CLIP_RECORDED: ClipRecordedEvent,
    EventType.CLIP_DELETED: ClipDeletedEvent,
    EventType.CLIP_RECHECKED: ClipRecheckedEvent,
    EventType.UPLOAD_STARTED: UploadStartedEvent,
    EventType.UPLOAD_COMPLETED: UploadCompletedEvent,
    EventType.UPLOAD_FAILED: UploadFailedEvent,
    EventType.FILTER_STARTED: FilterStartedEvent,
    EventType.FILTER_COMPLETED: FilterCompletedEvent,
    EventType.FILTER_FAILED: FilterFailedEvent,
    EventType.VLM_STARTED: VLMStartedEvent,
    EventType.VLM_COMPLETED: VLMCompletedEvent,
    EventType.VLM_FAILED: VLMFailedEvent,
    EventType.VLM_SKIPPED: VLMSkippedEvent,
    EventType.ALERT_DECISION_MADE: AlertDecisionMadeEvent,
    EventType.NOTIFICATION_SENT: NotificationSentEvent,
    EventType.NOTIFICATION_FAILED: NotificationFailedEvent,
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
        Index("idx_clip_states_created_at_desc", text("created_at DESC")),
        Index(
            "idx_clip_states_alerted",
            func.jsonb_extract_path_text(data, "alert_decision", "notify"),
        ),
        Index(
            "idx_clip_states_risk_level",
            func.jsonb_extract_path_text(data, "analysis_result", "risk_level"),
        ),
        Index(
            "idx_clip_states_activity_type",
            func.jsonb_extract_path_text(data, "analysis_result", "activity_type"),
        ),
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
        return await self.get_clip(clip_id)

    async def get_clip(self, clip_id: str) -> ClipStateData | None:
        """Get clip state by ID."""
        if self._engine is None:
            logger.warning("StateStore not initialized, returning None for %s", clip_id)
            return None

        try:
            async with self._engine.connect() as conn:
                result = await conn.execute(
                    select(ClipState.data, ClipState.created_at).where(ClipState.clip_id == clip_id)
                )
                row = result.one_or_none()

            if row is None:
                return None

            raw, created_at = row
            data_dict = self._parse_state_data(raw)
            data_dict["clip_id"] = clip_id
            data_dict["created_at"] = created_at
            return ClipStateData.model_validate(data_dict)
        except Exception as e:
            logger.error(
                "Failed to get clip state for %s: %s",
                clip_id,
                e,
                exc_info=True,
            )
            return None

    async def list_clips(
        self,
        *,
        camera: str | None = None,
        status: ClipStatus | None = None,
        alerted: bool | None = None,
        detected: bool | None = None,
        risk_level: str | None = None,
        activity_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        cursor: ClipListCursor | None = None,
        limit: int = 50,
    ) -> ClipListPage:
        """List clips with filtering and keyset pagination."""
        if self._engine is None:
            logger.warning("StateStore not initialized, returning empty clip list")
            return ClipListPage(clips=[], next_cursor=None, has_more=False)

        status_expr = func.jsonb_extract_path_text(ClipState.data, "status")
        camera_expr = func.jsonb_extract_path_text(ClipState.data, "camera_name")
        alerted_expr = func.jsonb_extract_path_text(ClipState.data, "alert_decision", "notify")
        detected_count_expr = func.coalesce(
            func.jsonb_array_length(
                func.jsonb_extract_path(ClipState.data, "filter_result", "detected_classes")
            ),
            0,
        )
        risk_expr = func.jsonb_extract_path_text(ClipState.data, "analysis_result", "risk_level")
        activity_expr = func.jsonb_extract_path_text(
            ClipState.data, "analysis_result", "activity_type"
        )

        conditions = []
        if camera is not None:
            conditions.append(camera_expr == camera)
        if status is None:
            conditions.append(or_(status_expr.is_(None), status_expr != ClipStatus.DELETED.value))
        else:
            conditions.append(status_expr == status.value)
        if alerted is True:
            conditions.append(alerted_expr == "true")
        elif alerted is False:
            conditions.append(or_(alerted_expr == "false", alerted_expr.is_(None)))
        if detected is True:
            conditions.append(detected_count_expr > 0)
        elif detected is False:
            conditions.append(detected_count_expr == 0)
        # Keep case-insensitive comparisons in SQL. If expression indexes are added for
        # these filters, index expressions must match the lower(...) form below.
        if risk_level is not None:
            conditions.append(func.lower(risk_expr) == risk_level.lower())
        if activity_type is not None:
            conditions.append(func.lower(activity_expr) == activity_type.lower())
        if since is not None:
            conditions.append(ClipState.created_at >= since)
        if until is not None:
            conditions.append(ClipState.created_at <= until)
        if cursor is not None:
            conditions.append(
                or_(
                    ClipState.created_at < cursor.created_at,
                    and_(
                        ClipState.created_at == cursor.created_at,
                        ClipState.clip_id < cursor.clip_id,
                    ),
                )
            )

        where_clause = and_(*conditions) if conditions else None

        fetch_limit = max(1, int(limit))
        query = select(ClipState.clip_id, ClipState.data, ClipState.created_at)
        if where_clause is not None:
            query = query.where(where_clause)
        query = query.order_by(ClipState.created_at.desc(), ClipState.clip_id.desc())
        query = query.limit(fetch_limit + 1)

        try:
            async with self._engine.connect() as conn:
                result = await conn.execute(query)
                rows = result.all()
        except Exception as e:
            logger.error("Failed to list clips: %s", e, exc_info=True)
            return ClipListPage(clips=[], next_cursor=None, has_more=False)

        has_more = len(rows) > fetch_limit
        visible_rows = rows[:fetch_limit]

        items: list[ClipStateData] = []
        for clip_id, raw, created_at in visible_rows:
            try:
                data_dict = self._parse_state_data(raw)
                data_dict["clip_id"] = clip_id
                data_dict["created_at"] = created_at
                items.append(ClipStateData.model_validate(data_dict))
            except Exception as exc:
                logger.warning("Failed parsing clip state %s: %s", clip_id, exc, exc_info=True)

        next_cursor: ClipListCursor | None = None
        if has_more and visible_rows:
            last_clip_id, _raw, last_created_at = visible_rows[-1]
            next_cursor = ClipListCursor(created_at=last_created_at, clip_id=last_clip_id)

        return ClipListPage(clips=items, next_cursor=next_cursor, has_more=has_more)

    async def mark_clip_deleted(self, clip_id: str) -> ClipStateData:
        """Mark a clip as deleted."""
        state = await self.get_clip(clip_id)
        if state is None:
            raise ValueError(f"Clip not found: {clip_id}")

        state.status = ClipStatus.DELETED
        state.clip_id = clip_id

        if self._engine is None:
            raise RuntimeError("StateStore not initialized")

        payload = state.model_dump(mode="json")
        stmt = (
            update(ClipState)
            .where(ClipState.clip_id == clip_id)
            .values(data=payload, updated_at=func.now())
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

        return state

    async def count_clips_since(self, since: datetime) -> int:
        """Count clips created since the given timestamp."""
        if self._engine is None:
            return 0

        query = select(func.count()).select_from(ClipState).where(ClipState.created_at >= since)
        try:
            async with self._engine.connect() as conn:
                result = await conn.execute(query)
                return int(result.scalar() or 0)
        except Exception as e:
            logger.warning("Failed to count clips: %s", e, exc_info=True)
            return 0

    async def count_alerts_since(self, since: datetime) -> int:
        """Count alert events (notification_sent) since the given timestamp."""
        if self._engine is None:
            return 0

        query = (
            select(func.count())
            .select_from(ClipEvent)
            .where(
                and_(
                    ClipEvent.event_type == EventType.NOTIFICATION_SENT,
                    ClipEvent.timestamp >= since,
                )
            )
        )
        try:
            async with self._engine.connect() as conn:
                result = await conn.execute(query)
                return int(result.scalar() or 0)
        except Exception as e:
            logger.warning("Failed to count alerts: %s", e, exc_info=True)
            return 0

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
                data_dict["clip_id"] = clip_id
                data_dict["created_at"] = created_at
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

    def create_event_store(self) -> EventStore:
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

    async def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown is handled by PostgresStateStore which owns the engine."""
        _ = timeout

    async def ping(self) -> bool:
        """Health check - verify database is reachable."""
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


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

    async def shutdown(self, timeout: float | None = None) -> None:
        """No resources to clean up."""
        _ = timeout

    async def ping(self) -> bool:
        """Noop store is always 'unhealthy' - indicates no real backend."""
        return False


class NoopStateStore(StateStore):
    """State store that drops writes and returns no data."""

    async def upsert(self, clip_id: str, data: ClipStateData) -> None:
        return

    async def get(self, clip_id: str) -> ClipStateData | None:
        return None

    async def get_clip(self, clip_id: str) -> ClipStateData | None:
        return None

    async def list_clips(
        self,
        *,
        camera: str | None = None,
        status: ClipStatus | None = None,
        alerted: bool | None = None,
        detected: bool | None = None,
        risk_level: str | None = None,
        activity_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        cursor: ClipListCursor | None = None,
        limit: int = 50,
    ) -> ClipListPage:
        _ = camera
        _ = status
        _ = alerted
        _ = detected
        _ = risk_level
        _ = activity_type
        _ = since
        _ = until
        _ = cursor
        _ = limit
        return ClipListPage(clips=[], next_cursor=None, has_more=False)

    async def mark_clip_deleted(self, clip_id: str) -> ClipStateData:
        raise ValueError(f"Clip not found: {clip_id}")

    async def count_clips_since(self, since: datetime) -> int:
        _ = since
        return 0

    async def count_alerts_since(self, since: datetime) -> int:
        _ = since
        return 0

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

    def create_event_store(self) -> EventStore:
        """Return NoopEventStore."""
        return NoopEventStore()
