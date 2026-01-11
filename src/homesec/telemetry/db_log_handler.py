from __future__ import annotations

import asyncio
import json
import logging
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from homesec.telemetry.db.log_table import logs
from homesec.telemetry.db.log_table import metadata as db_metadata
from homesec.telemetry.postgres_settings import PostgresConfig


_STANDARD_LOGRECORD_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
}


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _record_to_payload(record: logging.LogRecord) -> dict[str, Any]:
    camera_name = getattr(record, "camera_name", None) or None
    recording_id = getattr(record, "recording_id", None)
    if recording_id in ("", "-"):
        recording_id = None

    event_type = getattr(record, "event_type", None)
    kind = getattr(record, "kind", None) or ("event" if event_type else "log")

    msg_obj: Any
    if isinstance(record.msg, str):
        msg_obj = record.getMessage()
    else:
        msg_obj = record.msg

    fields: dict[str, Any] = {}
    for k, v in record.__dict__.items():
        if k in _STANDARD_LOGRECORD_ATTRS:
            continue
        if k in {"camera_name", "recording_id", "event_type", "kind"}:
            continue
        try:
            json.dumps(v, default=str)
            fields[k] = v
        except Exception:
            fields[k] = str(v)

    payload: dict[str, Any] = {
        "ts": _utc_iso(record.created),
        "created": record.created,
        "level": record.levelname,
        "logger": record.name,
        "module": record.module,
        "lineno": record.lineno,
        "pathname": record.pathname,
        "camera_name": camera_name,
        "recording_id": recording_id,
        "kind": kind,
        "event_type": event_type,
        "message": msg_obj,
        "fields": fields,
    }

    if record.exc_info:
        payload["exception"] = "".join(traceback.format_exception(*record.exc_info))
    elif record.exc_text:
        payload["exception"] = record.exc_text

    return payload


@dataclass(frozen=True)
class _DbRow:
    created_ts: float
    payload: dict[str, Any]


class AsyncPostgresJsonLogHandler(logging.Handler):
    """Best-effort DB log handler using async SQLAlchemy in a worker thread.

    - `emit()` must never block the caller.
    - When DB is down or queue is full, logs are dropped (with a stderr note).
    """

    def __init__(self, config: PostgresConfig) -> None:
        super().__init__()
        self.config = config
        self._queue: queue.Queue[_DbRow] = queue.Queue(maxsize=int(config.db_log_queue_size))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run_worker, name="db-log-writer", daemon=True)
        self._started = False
        self._drop_count = 0
        self._schema_ensured = False

        self.setLevel(getattr(logging, config.db_log_level, logging.INFO))

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def close(self) -> None:
        try:
            self._stop.set()
        finally:
            super().close()

    def emit(self, record: logging.LogRecord) -> None:
        if not self.config.enabled or not self.config.db_dsn:
            return
        if not self._started:
            self.start()

        try:
            payload = _record_to_payload(record)
            row = _DbRow(
                created_ts=float(record.created),
                payload=payload,
            )
        except Exception as exc:
            sys.stderr.write(f"[db-log] failed to serialize record: {exc}\n")
            return

        try:
            self._queue.put_nowait(row)
        except queue.Full:
            if self.config.db_log_drop_policy == "drop_oldest":
                try:
                    _ = self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(row)
                    return
                except queue.Full:
                    pass
            self._drop_count += 1
            if self._drop_count % 100 == 1:
                sys.stderr.write(f"[db-log] queue full; dropping logs (dropped={self._drop_count})\n")

    def _drain_batch(self) -> list[_DbRow]:
        batch: list[_DbRow] = []
        deadline = time.monotonic() + float(self.config.db_log_flush_s)
        while len(batch) < int(self.config.db_log_batch_size):
            timeout = max(0.0, deadline - time.monotonic())
            try:
                row = self._queue.get(timeout=timeout if batch else timeout)
            except queue.Empty:
                break
            batch.append(row)
        return batch

    def _run_worker(self) -> None:
        if not self.config.db_dsn:
            return

        backoff = float(self.config.db_log_backoff_initial_s)
        backoff_max = float(self.config.db_log_backoff_max_s)

        engine = create_async_engine(self.config.db_dsn, pool_pre_ping=True)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            try:
                loop.run_until_complete(self._ensure_schema(engine))
                self._schema_ensured = True
            except Exception as exc:
                sys.stderr.write(f"[db-log] failed ensuring schema (will retry on flush): {exc}\n")

            while True:
                if self._stop.is_set() and self._queue.empty():
                    break

                batch = self._drain_batch()
                if not batch:
                    continue

                try:
                    loop.run_until_complete(self._flush(engine, batch))
                    backoff = float(self.config.db_log_backoff_initial_s)
                except Exception as exc:
                    sys.stderr.write(f"[db-log] flush failed: {exc}; backing off {backoff:.1f}s\n")
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, backoff_max)
        finally:
            try:
                loop.run_until_complete(engine.dispose())
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass

    async def _flush(self, engine: AsyncEngine, batch: list[_DbRow]) -> None:
        if not self._schema_ensured:
            await self._ensure_schema(engine)
            self._schema_ensured = True

        rows = []
        for row in batch:
            rows.append(
                {
                    "ts": datetime.fromtimestamp(row.created_ts, tz=timezone.utc),
                    "payload": row.payload,
                }
            )
        async with engine.begin() as conn:
            await conn.execute(insert(logs), rows)

    async def _ensure_schema(self, engine: AsyncEngine) -> None:
        async with engine.begin() as conn:
            await conn.run_sync(db_metadata.create_all)
