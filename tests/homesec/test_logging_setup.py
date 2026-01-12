"""Tests for logging setup module."""

from __future__ import annotations

import json
import logging
import sys
import types
from unittest.mock import patch

import pytest

from homesec.logging_setup import configure_logging, set_camera_name, set_recording_id


@pytest.fixture(autouse=True)
def reset_logging_state() -> None:
    """Reset global logging state before each test."""
    import homesec.logging_setup as module

    original_camera = module._CURRENT_CAMERA_NAME
    original_recording = module._CURRENT_RECORDING_ID

    yield

    module._CURRENT_CAMERA_NAME = original_camera
    module._CURRENT_RECORDING_ID = original_recording


@pytest.fixture(autouse=True)
def reset_logging_root() -> None:
    """Restore root logger handlers/levels after each test."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_filters = list(root.filters)
    original_level = root.level
    original_disabled = root.disabled
    original_dropbox_level = logging.getLogger("dropbox").level
    original_urllib3_level = logging.getLogger("urllib3").level

    yield

    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in original_handlers:
        root.addHandler(handler)

    for filt in list(root.filters):
        root.removeFilter(filt)
    for filt in original_filters:
        root.addFilter(filt)

    root.setLevel(original_level)
    root.disabled = original_disabled
    logging.captureWarnings(False)
    logging.getLogger("dropbox").setLevel(original_dropbox_level)
    logging.getLogger("urllib3").setLevel(original_urllib3_level)


class TestLoggingInjection:
    """Tests for camera/recording injection via configure_logging."""

    def test_injects_camera_name(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Camera name is injected into log records."""
        # Given: Logging configured with a custom formatter
        with patch.dict("os.environ", {"DB_DSN": ""}, clear=False):
            configure_logging(log_level="INFO")
        root = logging.getLogger()
        handler = root.handlers[0]
        handler.setFormatter(logging.Formatter("%(camera_name)s %(message)s"))
        set_camera_name("back_yard")

        # When: Logging a message
        logging.getLogger("test").info("hello")

        # Then: Output includes injected camera name
        captured = capsys.readouterr().out.strip().splitlines()
        assert any("back_yard hello" in line for line in captured)

    def test_injects_recording_id(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Recording ID is injected into log records."""
        # Given: Logging configured with a formatter including recording_id
        with patch.dict("os.environ", {"DB_DSN": ""}, clear=False):
            configure_logging(log_level="INFO")
        root = logging.getLogger()
        handler = root.handlers[0]
        handler.setFormatter(logging.Formatter("%(recording_id)s %(message)s"))
        set_recording_id("rec_123")

        # When: Logging a message
        logging.getLogger("test").info("hello")

        # Then: Output includes injected recording ID
        captured = capsys.readouterr().out.strip().splitlines()
        assert any("rec_123 hello" in line for line in captured)


class TestLoggingExtras:
    """Tests for JSON extras formatting."""

    def test_extras_render_as_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Extra fields are rendered as JSON on a new line."""
        # Given: Logging configured with JSON extra formatter
        with patch.dict("os.environ", {"DB_DSN": ""}, clear=False):
            configure_logging(log_level="INFO")

        # When: Logging with extra fields
        logging.getLogger("test").info(
            "test message",
            extra={"custom_field": "custom_value", "number_field": 42},
        )

        # Then: Extras are appended as JSON
        lines = capsys.readouterr().out.strip().splitlines()
        assert len(lines) >= 2
        json_start = next(index for index, line in enumerate(lines) if line.strip().startswith("{"))
        extras = json.loads("\n".join(lines[json_start:]))
        assert extras["custom_field"] == "custom_value"
        assert extras["number_field"] == 42

    def test_excludes_camera_and_recording_from_extras(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """camera_name and recording_id are excluded from extras."""
        # Given: Logging configured with JSON extra formatter
        with patch.dict("os.environ", {"DB_DSN": ""}, clear=False):
            configure_logging(log_level="INFO")

        # When: Logging with camera/recording extras
        logging.getLogger("test").info(
            "message",
            extra={"camera_name": "front", "recording_id": "rec_1", "custom": "value"},
        )

        # Then: Extras exclude camera/recording fields
        lines = capsys.readouterr().out.strip().splitlines()
        json_start = next(index for index, line in enumerate(lines) if line.strip().startswith("{"))
        extras = json.loads("\n".join(lines[json_start:]))
        assert "camera_name" not in extras
        assert "recording_id" not in extras
        assert extras.get("custom") == "value"


class TestDbHandlerFilter:
    """Tests for DB handler filtering via configure_logging."""

    def test_db_handler_filters_events_and_levels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DB handler allows event logs and blocks low-level non-events."""

        # Given: DB logging enabled with a fake handler
        class _FakeDbHandler(logging.Handler):
            instances: list[_FakeDbHandler] = []

            def __init__(self, _config: object) -> None:
                super().__init__()
                self.records: list[logging.LogRecord] = []
                _FakeDbHandler.instances.append(self)

            def emit(self, record: logging.LogRecord) -> None:
                self.records.append(record)

        fake_module = types.ModuleType("homesec.telemetry.db_log_handler")
        fake_module.AsyncPostgresJsonLogHandler = _FakeDbHandler
        monkeypatch.setitem(sys.modules, "homesec.telemetry.db_log_handler", fake_module)
        monkeypatch.setenv("DB_DSN", "postgresql://user:pass@localhost/db")
        monkeypatch.setenv("DB_LOG_LEVEL", "INFO")

        configure_logging(log_level="DEBUG")
        handler = _FakeDbHandler.instances[-1]

        # When: Logging an event and a debug message
        logger = logging.getLogger()
        logger.debug("event", extra={"kind": "event"})
        logger.debug("debug")
        logger.info("info")

        # Then: Event debug passes, non-event debug is filtered
        messages = [record.getMessage() for record in handler.records]
        assert "event" in messages
        assert "debug" not in messages
        assert "info" in messages


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configures_console_handler(self) -> None:
        """Configures console handler on root logger."""
        # When: Configuring logging
        with patch.dict("os.environ", {"DB_DSN": ""}, clear=False):
            configure_logging(log_level="DEBUG")

        # Then: Root logger has handlers
        root = logging.getLogger()
        assert len(root.handlers) > 0

    def test_suppresses_third_party_loggers(self) -> None:
        """Sets third-party loggers to WARNING level."""
        # When: Configuring logging
        with patch.dict("os.environ", {"DB_DSN": ""}, clear=False):
            configure_logging()

        # Then: Third-party loggers are suppressed
        assert logging.getLogger("dropbox").level == logging.WARNING
        assert logging.getLogger("urllib3").level == logging.WARNING

    def test_skips_db_handler_when_disabled(self) -> None:
        """Doesn't add DB handler when DB_DSN not set."""
        # Given: No DB_DSN environment variable

        # When: Configuring logging
        with patch.dict("os.environ", {"DB_DSN": ""}, clear=False):
            configure_logging()

        # Then: No AsyncPostgresJsonLogHandler added
        from homesec.telemetry.db_log_handler import AsyncPostgresJsonLogHandler

        root = logging.getLogger()
        has_db_handler = any(isinstance(h, AsyncPostgresJsonLogHandler) for h in root.handlers)
        assert not has_db_handler
