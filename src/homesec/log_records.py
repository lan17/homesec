from __future__ import annotations

import logging
from collections.abc import Collection

STANDARD_LOGRECORD_ATTRS = frozenset(
    {
        "name",
        "msg",
        "message",
        "asctime",
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
)


def extract_log_record_extras(
    record: logging.LogRecord,
    *,
    exclude: Collection[str] = (),
) -> dict[str, object]:
    """Return caller-provided LogRecord extras after standard/promoted fields are removed."""
    excluded = STANDARD_LOGRECORD_ATTRS | frozenset(exclude)
    return {key: value for key, value in record.__dict__.items() if key not in excluded}


def is_event_log_record(record: logging.LogRecord) -> bool:
    """Return true when a log record should be treated as structured event telemetry."""
    return getattr(record, "kind", None) == "event" or bool(getattr(record, "event_type", None))
