"""Tests for DB log handler lifecycle behavior."""

from __future__ import annotations

import logging
import threading
from typing import cast

from homesec.telemetry.db_log_handler import AsyncPostgresJsonLogHandler, _record_to_payload
from homesec.telemetry.postgres_settings import PostgresConfig


class _FakeThread:
    def __init__(self) -> None:
        self.join_calls: list[float] = []
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        assert timeout is not None
        self.join_calls.append(timeout)
        self._alive = False


def test_record_to_payload_excludes_formatter_created_standard_fields() -> None:
    """DB payload fields should only contain caller custom extras."""
    # Given: A log record with custom extras that has already been formatted
    record = logging.getLogger("homesec.test").makeRecord(
        "homesec.test",
        logging.INFO,
        "/tmp/source.py",
        42,
        "hello %s",
        ("world",),
        None,
        extra={
            "camera_name": "front",
            "recording_id": "clip-1.mp4",
            "event_type": "recording_start",
            "kind": "event",
            "custom": "value",
        },
    )
    logging.Formatter("%(asctime)s %(levelname)s %(message)s").format(record)

    # When: Converting the record to a DB telemetry payload
    payload = _record_to_payload(record)

    # Then: Formatter-created and promoted fields do not leak into nested fields
    assert payload["message"] == "hello world"
    assert payload["camera_name"] == "front"
    assert payload["recording_id"] == "clip-1.mp4"
    assert payload["kind"] == "event"
    assert payload["event_type"] == "recording_start"
    assert payload["fields"] == {"custom": "value"}


def test_record_to_payload_treats_event_type_as_event_kind() -> None:
    """DB payload should classify event_type-only records as event telemetry."""
    # Given: A log record with event_type but no explicit kind
    record = logging.getLogger("homesec.test").makeRecord(
        "homesec.test",
        logging.INFO,
        "/tmp/source.py",
        42,
        "usage",
        (),
        None,
        extra={"event_type": "vlm_usage", "total_tokens": 12},
    )

    # When: Converting the record to a DB telemetry payload
    payload = _record_to_payload(record)

    # Then: The payload is classified as an event and custom fields are preserved
    assert payload["kind"] == "event"
    assert payload["event_type"] == "vlm_usage"
    assert payload["fields"] == {"total_tokens": 12}


def test_db_log_handler_close_joins_writer_thread_when_started() -> None:
    """Close should wait briefly for writer thread to flush before exit."""
    # Given: A started handler with a live writer thread
    config = PostgresConfig(
        db_dsn="postgresql+asyncpg://user:pass@localhost/homesec",
        db_log_flush_s=0.4,
    )
    handler = AsyncPostgresJsonLogHandler(config)
    fake_thread = _FakeThread()
    handler._started = True
    handler._thread = cast(threading.Thread, fake_thread)

    # When: Closing the handler
    handler.close()

    # Then: Close signals stop and joins the writer thread with bounded timeout
    assert handler._stop.is_set() is True
    assert fake_thread.join_calls == [1.0]


def test_db_log_handler_close_skips_join_on_writer_thread_itself() -> None:
    """Close should not attempt to join when called from writer thread context."""
    # Given: Handler marked started where writer thread is current thread
    config = PostgresConfig(db_dsn="postgresql+asyncpg://user:pass@localhost/homesec")
    handler = AsyncPostgresJsonLogHandler(config)
    handler._started = True
    handler._thread = cast(threading.Thread, threading.current_thread())

    # When: Closing handler from current thread context
    handler.close()

    # Then: Close still signals stop without deadlocking on self-join
    assert handler._stop.is_set() is True
