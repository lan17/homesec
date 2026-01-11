from __future__ import annotations

import json
import logging
import logging.config
import os

from homesec.telemetry.postgres_settings import PostgresConfig

_CURRENT_CAMERA_NAME = "-"
_CURRENT_RECORDING_ID: str | None = None
_STANDARD_LOGRECORD_ATTRS = {
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

class _CameraNameFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "camera_name") or getattr(record, "camera_name") in (None, ""):
            record.camera_name = _CURRENT_CAMERA_NAME
        return True


class _RecordingIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "recording_id") or getattr(record, "recording_id") in (None, ""):
            record.recording_id = _CURRENT_RECORDING_ID
        return True


class _DbLevelFilter(logging.Filter):
    def __init__(self, *, min_level: int) -> None:
        super().__init__()
        self._min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "kind", None) == "event":
            return True
        return record.levelno >= logging.WARNING or record.levelno >= self._min_level


class _JsonExtraFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = _extract_extras(record)
        if not extras:
            return base
        extras_json = json.dumps(extras, indent=2, default=str, sort_keys=True)
        return f"{base}\n{extras_json}"


def _extract_extras(record: logging.LogRecord) -> dict[str, object]:
    extras: dict[str, object] = {}
    for key, value in record.__dict__.items():
        if key in _STANDARD_LOGRECORD_ATTRS:
            continue
        if key in {"camera_name", "recording_id"}:
            continue
        extras[key] = value
    return extras


def set_camera_name(name: str | None) -> None:
    """Set the `camera_name` value injected into log records."""
    global _CURRENT_CAMERA_NAME
    _CURRENT_CAMERA_NAME = name or "-"

def set_recording_id(recording_id: str | None) -> None:
    """Set the `recording_id` value injected into log records."""
    global _CURRENT_RECORDING_ID
    _CURRENT_RECORDING_ID = recording_id or None


def _install_camera_filter() -> None:
    root = logging.getLogger()
    for handler in root.handlers:
        if any(isinstance(f, _CameraNameFilter) for f in handler.filters):
            continue
        handler.addFilter(_CameraNameFilter())

def _install_recording_filter() -> None:
    root = logging.getLogger()
    for handler in root.handlers:
        if any(isinstance(f, _RecordingIdFilter) for f in handler.filters):
            continue
        handler.addFilter(_RecordingIdFilter())


def configure_logging(*, log_level: str = "INFO", camera_name: str | None = None) -> None:
    """Configure root logging with a consistent format.

    Format includes `camera_name` plus `module:lineno` for easier multi-process debugging.
    If `DB_DSN` is configured (via env or `.env`), also emits logs to Postgres as JSON.
    """
    console_level_name = str(log_level).upper()
    default_console_fmt = (
        "%(asctime)s %(levelname)s [%(camera_name)s] "
        "%(module)s %(pathname)s:%(lineno)d %(message)s"
    )
    console_fmt = os.getenv("CONSOLE_LOG_FORMAT", default_console_fmt)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "homesec.logging_setup._JsonExtraFormatter",
                    "format": console_fmt,
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": console_level_name,
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"level": "DEBUG", "handlers": ["console"]},
        }
    )

    _install_camera_filter()
    _install_recording_filter()
    set_camera_name(camera_name)
    logging.captureWarnings(True)

    # Reduce noisy third-party request logs by default.
    logging.getLogger("dropbox").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    config = PostgresConfig()
    if not config.enabled:
        return

    root = logging.getLogger()
    try:
        from homesec.telemetry.db_log_handler import AsyncPostgresJsonLogHandler
    except Exception as exc:
        # Keep the app running even if DB logging deps aren't installed.
        logging.getLogger(__name__).warning("DB_DSN is set but DB log handler failed to import: %s", exc)
        return

    for handler in list(root.handlers):
        if isinstance(handler, AsyncPostgresJsonLogHandler):
            return

    db_handler = AsyncPostgresJsonLogHandler(config)
    min_level = getattr(logging, config.db_log_level, logging.INFO)
    db_handler.setLevel(logging.DEBUG)
    db_handler.addFilter(_DbLevelFilter(min_level=min_level))
    root.addHandler(db_handler)
    _install_camera_filter()
    _install_recording_filter()
