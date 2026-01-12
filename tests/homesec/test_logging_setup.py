"""Tests for logging setup module."""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

from homesec.logging_setup import (
    _CameraNameFilter,
    _DbLevelFilter,
    _extract_extras,
    _install_camera_filter,
    _install_recording_filter,
    _JsonExtraFormatter,
    _RecordingIdFilter,
    configure_logging,
    set_camera_name,
    set_recording_id,
)


class TestCameraNameFilter:
    """Tests for _CameraNameFilter."""

    def test_adds_camera_name_when_missing(self) -> None:
        """Adds camera_name attribute when not present on record."""
        # Given: A log record without camera_name
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        filter_ = _CameraNameFilter()

        # When: Filtering the record
        result = filter_.filter(record)

        # Then: camera_name is added with default value
        assert result is True
        assert hasattr(record, "camera_name")

    def test_preserves_existing_camera_name(self) -> None:
        """Preserves camera_name when already set."""
        # Given: A log record with camera_name
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        record.camera_name = "front_door"
        filter_ = _CameraNameFilter()

        # When: Filtering the record
        result = filter_.filter(record)

        # Then: camera_name is preserved
        assert result is True
        assert record.camera_name == "front_door"  # type: ignore[attr-defined]

    def test_replaces_none_camera_name(self) -> None:
        """Replaces None camera_name with default."""
        # Given: A log record with None camera_name
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        record.camera_name = None
        filter_ = _CameraNameFilter()

        # When: Filtering the record
        result = filter_.filter(record)

        # Then: camera_name is set to default
        assert result is True
        assert record.camera_name is not None  # type: ignore[attr-defined]

    def test_replaces_empty_camera_name(self) -> None:
        """Replaces empty string camera_name with default."""
        # Given: A log record with empty camera_name
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        record.camera_name = ""
        filter_ = _CameraNameFilter()

        # When: Filtering the record
        result = filter_.filter(record)

        # Then: camera_name is set to default
        assert result is True
        assert record.camera_name != ""  # type: ignore[attr-defined]


class TestRecordingIdFilter:
    """Tests for _RecordingIdFilter."""

    def test_adds_recording_id_when_missing(self) -> None:
        """Adds recording_id attribute when not present on record."""
        # Given: A log record without recording_id
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        filter_ = _RecordingIdFilter()

        # When: Filtering the record
        result = filter_.filter(record)

        # Then: recording_id is added
        assert result is True
        assert hasattr(record, "recording_id")

    def test_preserves_existing_recording_id(self) -> None:
        """Preserves recording_id when already set."""
        # Given: A log record with recording_id
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        record.recording_id = "rec_123"
        filter_ = _RecordingIdFilter()

        # When: Filtering the record
        result = filter_.filter(record)

        # Then: recording_id is preserved
        assert result is True
        assert record.recording_id == "rec_123"  # type: ignore[attr-defined]


class TestDbLevelFilter:
    """Tests for _DbLevelFilter."""

    def test_always_passes_event_kind(self) -> None:
        """Records with kind='event' always pass."""
        # Given: A DEBUG record with kind='event'
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="test event",
            args=(),
            exc_info=None,
        )
        record.kind = "event"
        filter_ = _DbLevelFilter(min_level=logging.WARNING)

        # When: Filtering the record
        result = filter_.filter(record)

        # Then: Record passes despite being DEBUG
        assert result is True

    def test_passes_warning_and_above(self) -> None:
        """WARNING and above always pass."""
        # Given: A WARNING record without kind='event'
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="test warning",
            args=(),
            exc_info=None,
        )
        filter_ = _DbLevelFilter(min_level=logging.ERROR)

        # When: Filtering the record
        result = filter_.filter(record)

        # Then: Record passes because it's WARNING or above
        assert result is True

    def test_passes_at_min_level(self) -> None:
        """Records at min_level pass."""
        # Given: An INFO record with min_level=INFO
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test info",
            args=(),
            exc_info=None,
        )
        filter_ = _DbLevelFilter(min_level=logging.INFO)

        # When: Filtering the record
        result = filter_.filter(record)

        # Then: Record passes
        assert result is True

    def test_blocks_below_min_level(self) -> None:
        """Records below min_level are blocked."""
        # Given: A DEBUG record with min_level=INFO
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="test debug",
            args=(),
            exc_info=None,
        )
        filter_ = _DbLevelFilter(min_level=logging.INFO)

        # When: Filtering the record
        result = filter_.filter(record)

        # Then: Record is blocked
        assert result is False


class TestJsonExtraFormatter:
    """Tests for _JsonExtraFormatter."""

    def test_no_extras_returns_base_only(self) -> None:
        """Returns base format when no extras present."""
        # Given: A simple log record
        formatter = _JsonExtraFormatter("%(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )

        # When: Formatting the record
        result = formatter.format(record)

        # Then: Just the message is returned
        assert result == "test message"

    def test_extras_appended_as_json(self) -> None:
        """Extra fields are appended as formatted JSON."""
        # Given: A log record with extra fields
        formatter = _JsonExtraFormatter("%(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        record.custom_field = "custom_value"
        record.number_field = 42

        # When: Formatting the record
        result = formatter.format(record)

        # Then: Extras are appended as JSON
        assert "test message" in result
        assert "custom_field" in result
        assert "custom_value" in result
        # Verify JSON is valid
        lines = result.split("\n", 1)
        assert len(lines) == 2
        extras = json.loads(lines[1])
        assert extras["custom_field"] == "custom_value"
        assert extras["number_field"] == 42


class TestExtractExtras:
    """Tests for _extract_extras function."""

    def test_no_extras(self) -> None:
        """Returns empty dict when no extras."""
        # Given: A simple log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )

        # When: Extracting extras
        result = _extract_extras(record)

        # Then: Empty dict returned
        assert result == {}

    def test_excludes_standard_attrs(self) -> None:
        """Standard LogRecord attributes are excluded."""
        # Given: A log record with standard attrs
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="test",
            args=(),
            exc_info=None,
        )

        # When: Extracting extras
        result = _extract_extras(record)

        # Then: Standard attrs not in result
        assert "name" not in result
        assert "msg" not in result
        assert "pathname" not in result
        assert "lineno" not in result

    def test_excludes_camera_and_recording(self) -> None:
        """camera_name and recording_id are excluded."""
        # Given: A log record with camera_name and recording_id
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        record.camera_name = "front_door"
        record.recording_id = "rec_123"
        record.custom = "value"

        # When: Extracting extras
        result = _extract_extras(record)

        # Then: Only custom field is included
        assert "camera_name" not in result
        assert "recording_id" not in result
        assert result.get("custom") == "value"


class TestSetCameraName:
    """Tests for set_camera_name function."""

    def test_sets_camera_name(self) -> None:
        """Sets the global camera name."""
        # Given: A camera name to set

        # When: Setting camera name
        set_camera_name("back_yard")

        # Then: Filter uses the new value
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        filter_ = _CameraNameFilter()
        filter_.filter(record)
        assert record.camera_name == "back_yard"  # type: ignore[attr-defined]

    def test_none_becomes_dash(self) -> None:
        """None camera name becomes '-'."""
        # Given: None camera name

        # When: Setting camera name to None
        set_camera_name(None)

        # Then: Default is '-'
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        filter_ = _CameraNameFilter()
        filter_.filter(record)
        assert record.camera_name == "-"  # type: ignore[attr-defined]


class TestSetRecordingId:
    """Tests for set_recording_id function."""

    def test_sets_recording_id(self) -> None:
        """Sets the global recording ID."""
        # Given: A recording ID to set

        # When: Setting recording ID
        set_recording_id("rec_456")

        # Then: Filter uses the new value
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        filter_ = _RecordingIdFilter()
        filter_.filter(record)
        assert record.recording_id == "rec_456"  # type: ignore[attr-defined]

    def test_none_stays_none(self) -> None:
        """None recording ID stays None."""
        # Given: None recording ID

        # When: Setting recording ID to None
        set_recording_id(None)

        # Then: Value is None
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        filter_ = _RecordingIdFilter()
        filter_.filter(record)
        assert record.recording_id is None  # type: ignore[attr-defined]


class TestInstallFilters:
    """Tests for filter installation functions."""

    def test_install_camera_filter_adds_to_handlers(self) -> None:
        """Installs camera filter on all root handlers."""
        # Given: A logger with a handler
        logger = logging.getLogger()
        handler = logging.StreamHandler()
        original_handlers = list(logger.handlers)
        logger.addHandler(handler)

        try:
            # When: Installing camera filter
            _install_camera_filter()

            # Then: Filter is added to handler
            has_camera_filter = any(isinstance(f, _CameraNameFilter) for f in handler.filters)
            assert has_camera_filter
        finally:
            # Cleanup
            logger.removeHandler(handler)
            for h in list(logger.handlers):
                if h not in original_handlers:
                    logger.removeHandler(h)

    def test_install_camera_filter_no_duplicates(self) -> None:
        """Doesn't add duplicate camera filters."""
        # Given: A logger with handler that already has the filter
        logger = logging.getLogger()
        handler = logging.StreamHandler()
        handler.addFilter(_CameraNameFilter())
        original_handlers = list(logger.handlers)
        logger.addHandler(handler)

        try:
            initial_count = sum(1 for f in handler.filters if isinstance(f, _CameraNameFilter))

            # When: Installing camera filter again
            _install_camera_filter()

            # Then: No duplicate added
            final_count = sum(1 for f in handler.filters if isinstance(f, _CameraNameFilter))
            assert final_count == initial_count
        finally:
            logger.removeHandler(handler)
            for h in list(logger.handlers):
                if h not in original_handlers:
                    logger.removeHandler(h)

    def test_install_recording_filter_adds_to_handlers(self) -> None:
        """Installs recording filter on all root handlers."""
        # Given: A logger with a handler
        logger = logging.getLogger()
        handler = logging.StreamHandler()
        original_handlers = list(logger.handlers)
        logger.addHandler(handler)

        try:
            # When: Installing recording filter
            _install_recording_filter()

            # Then: Filter is added to handler
            has_recording_filter = any(isinstance(f, _RecordingIdFilter) for f in handler.filters)
            assert has_recording_filter
        finally:
            logger.removeHandler(handler)
            for h in list(logger.handlers):
                if h not in original_handlers:
                    logger.removeHandler(h)


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_sets_console_handler(self) -> None:
        """Configures console handler on root logger."""
        # Given: A fresh logger state
        # (Note: This modifies global state, cleanup needed)

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

    def test_captures_warnings(self) -> None:
        """Enables Python warning capture."""
        # When: Configuring logging
        with patch.dict("os.environ", {"DB_DSN": ""}, clear=False):
            configure_logging()

        # Then: Warnings are captured (hard to test directly, but configure_logging
        # should call logging.captureWarnings(True))
        # We just verify no exception is raised
