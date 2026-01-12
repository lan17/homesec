"""Tests for SendGrid email notifier."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from homesec.models.alert import Alert
from homesec.models.config import SendGridEmailConfig
from homesec.models.vlm import SequenceAnalysis
from homesec.plugins.notifiers.sendgrid_email import SendGridEmailNotifier


def _make_config(**overrides: Any) -> SendGridEmailConfig:
    """Create a SendGridEmailConfig with defaults."""
    defaults: dict[str, Any] = {
        "from_email": "sender@example.com",
        "to_emails": ["to@example.com"],
        "subject_template": "Alert: {camera_name}",
        "text_template": "Clip {clip_id}",
        "html_template": "<b>{clip_id}</b>",
    }
    defaults.update(overrides)
    return SendGridEmailConfig(**defaults)


def _make_alert(**overrides: Any) -> Alert:
    """Create a sample Alert with defaults."""
    defaults: dict[str, Any] = {
        "clip_id": "clip_123",
        "camera_name": "front_door",
        "storage_uri": "mock://clip_123",
        "view_url": "http://example.com/clip_123",
        "risk_level": "low",
        "activity_type": "delivery",
        "notify_reason": "risk_level=low",
        "summary": "Package delivered",
        "ts": datetime.now(),
        "dedupe_key": "clip_123",
        "upload_failed": False,
    }
    defaults.update(overrides)
    return Alert(**defaults)


def _mock_http_response(status: int) -> AsyncMock:
    """Create a mock aiohttp response."""
    response = AsyncMock()
    response.status = status
    response.text = AsyncMock(return_value="OK" if status < 400 else "Error")
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    return response


class TestSendGridNotifierSend:
    """Tests for the send method - verifying HTTP requests."""

    @pytest.mark.asyncio
    async def test_send_constructs_correct_api_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Send constructs the correct HTTP request to SendGrid API."""
        # Given: A SendGrid notifier with mocked HTTP session
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        captured_request: dict[str, Any] = {}

        mock_response = _mock_http_response(202)

        mock_session = MagicMock()

        def capture_post(url: str, json: dict[str, Any], headers: dict[str, str]) -> AsyncMock:
            captured_request["url"] = url
            captured_request["json"] = json
            captured_request["headers"] = headers
            return mock_response

        mock_session.post = capture_post
        mock_session.close = AsyncMock()
        mock_session.closed = False
        notifier._session = mock_session

        # When: Sending an alert
        await notifier.send(_make_alert(camera_name="back_yard", clip_id="clip_456"))

        # Then: API was called with correct structure
        assert "mail/send" in captured_request["url"]
        assert captured_request["headers"]["Authorization"] == "Bearer test-api-key"

        payload = captured_request["json"]

        # Verify sender
        assert payload["from"]["email"] == "sender@example.com"

        # Verify recipients
        personalizations = payload["personalizations"]
        assert len(personalizations) == 1
        assert personalizations[0]["to"] == [{"email": "to@example.com"}]

        # Verify subject was templated
        assert personalizations[0]["subject"] == "Alert: back_yard"

        # Verify content
        content = payload["content"]
        text_content = next(c for c in content if c["type"] == "text/plain")
        html_content = next(c for c in content if c["type"] == "text/html")
        assert text_content["value"] == "Clip clip_456"
        assert html_content["value"] == "<b>clip_456</b>"

        await notifier.shutdown()

    @pytest.mark.asyncio
    async def test_send_raises_on_api_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises RuntimeError with status code on API failure."""
        # Given: A SendGrid notifier with API returning 400
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        mock_response = _mock_http_response(400)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()
        mock_session.closed = False
        notifier._session = mock_session

        # When/Then: Send raises RuntimeError
        with pytest.raises(RuntimeError, match="400"):
            await notifier.send(_make_alert())

        await notifier.shutdown()

    @pytest.mark.asyncio
    async def test_send_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises RuntimeError when API key is missing."""
        # Given: A notifier without API key
        monkeypatch.delenv("SENDGRID_API_KEY", raising=False)
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        # When/Then: Send raises RuntimeError
        with pytest.raises(RuntimeError, match="API key missing"):
            await notifier.send(_make_alert())

    @pytest.mark.asyncio
    async def test_send_fails_after_shutdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises RuntimeError after shutdown."""
        # Given: A notifier that has been shut down
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)
        await notifier.shutdown()

        # When/Then: Send raises RuntimeError
        with pytest.raises(RuntimeError, match="shut down"):
            await notifier.send(_make_alert())


class TestSendGridNotifierPing:
    """Tests for the ping method."""

    @pytest.mark.asyncio
    async def test_ping_returns_true_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns True when API responds successfully."""
        # Given: A SendGrid notifier with mocked API success
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"{}")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()
        mock_session.closed = False
        notifier._session = mock_session

        # When: Pinging
        result = await notifier.ping()

        # Then: Returns True
        assert result is True

        await notifier.shutdown()

    @pytest.mark.asyncio
    async def test_ping_returns_false_on_api_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns False when API returns error status."""
        # Given: A SendGrid notifier with API returning 401
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()
        mock_session.closed = False
        notifier._session = mock_session

        # When: Pinging
        result = await notifier.ping()

        # Then: Returns False
        assert result is False

        await notifier.shutdown()

    @pytest.mark.asyncio
    async def test_ping_false_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns False when API key is missing."""
        # Given: A notifier without API key
        monkeypatch.delenv("SENDGRID_API_KEY", raising=False)
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        # When: Pinging
        result = await notifier.ping()

        # Then: Returns False
        assert result is False

    @pytest.mark.asyncio
    async def test_ping_false_after_shutdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns False after shutdown."""
        # Given: A notifier that has been shut down
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)
        await notifier.shutdown()

        # When: Pinging
        result = await notifier.ping()

        # Then: Returns False
        assert result is False


class TestSendGridPayloadBuilding:
    """Tests for payload construction - verifying output structure."""

    def test_builds_payload_with_all_recipients(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Payload includes all recipients (to, cc, bcc)."""
        # Given: A notifier with to, cc, bcc recipients
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config(
            to_emails=["to1@example.com", "to2@example.com"],
            cc_emails=["cc@example.com"],
            bcc_emails=["bcc@example.com"],
        )
        notifier = SendGridEmailNotifier(config)

        # When: Building payload
        payload = notifier._build_payload("Subject", "text", "html")

        # Then: All recipients are included correctly
        personalizations = payload["personalizations"]
        assert len(personalizations) == 1
        p = personalizations[0]
        assert p["to"] == [{"email": "to1@example.com"}, {"email": "to2@example.com"}]
        assert p["cc"] == [{"email": "cc@example.com"}]
        assert p["bcc"] == [{"email": "bcc@example.com"}]

    def test_excludes_cc_when_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Payload excludes cc when not configured."""
        # Given: A notifier without cc
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config(cc_emails=[])
        notifier = SendGridEmailNotifier(config)

        # When: Building payload
        payload = notifier._build_payload("Subject", "text", "html")

        # Then: No cc in personalization
        personalizations = payload["personalizations"]
        assert "cc" not in personalizations[0]

    def test_text_only_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Payload with text-only content."""
        # Given: A notifier
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        # When: Building payload with text only
        payload = notifier._build_payload("Subject", "text body", "")

        # Then: Only text content is included
        content = payload["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text/plain"
        assert content[0]["value"] == "text body"

    def test_html_only_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Payload with HTML-only content."""
        # Given: A notifier
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        # When: Building payload with HTML only
        payload = notifier._build_payload("Subject", "", "<b>html</b>")

        # Then: Only HTML content is included
        content = payload["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text/html"
        assert content[0]["value"] == "<b>html</b>"

    def test_sender_includes_name_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sender includes name when configured."""
        # Given: A notifier with from_name
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config(from_name="HomeSec Alerts")
        notifier = SendGridEmailNotifier(config)

        # When: Building payload
        payload = notifier._build_payload("Subject", "text", "")

        # Then: Sender has name
        sender = payload["from"]
        assert sender["email"] == "sender@example.com"
        assert sender["name"] == "HomeSec Alerts"


class TestSendGridTemplateRendering:
    """Tests for template rendering - verifying output text."""

    def test_render_subject_substitutes_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Subject template substitutes variables from alert."""
        # Given: A notifier with subject template
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config(subject_template="[{risk_level}] {camera_name}")
        notifier = SendGridEmailNotifier(config)
        alert = _make_alert(camera_name="back_yard", risk_level="high")

        # When: Rendering subject
        subject = notifier._render_subject(alert)

        # Then: Variables are substituted
        assert subject == "[high] back_yard"

    def test_render_html_escapes_special_chars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HTML rendering escapes special characters to prevent XSS."""
        # Given: A notifier
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        # When: Rendering value with special chars
        result = notifier._render_value_html("<script>alert('xss')</script>")

        # Then: HTML is escaped
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_render_dict_as_html_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dict values are rendered as HTML unordered lists."""
        # Given: A notifier
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        # When: Rendering a dict
        result = notifier._render_value_html({"key1": "value1", "key2": "value2"})

        # Then: Rendered as HTML list with keys and values
        assert "<ul>" in result
        assert "<li>" in result
        assert "<strong>key1:</strong>" in result
        assert "value1" in result
        assert "<strong>key2:</strong>" in result
        assert "value2" in result

    def test_render_analysis_includes_key_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SequenceAnalysis is rendered with key information."""
        # Given: A notifier with analysis
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        analysis = SequenceAnalysis(
            sequence_description="Person walks by",
            max_risk_level="low",
            primary_activity="passerby",
            observations=["entered", "left"],
            entities_timeline=[],
            requires_review=False,
            frame_count=5,
            video_start_time="00:00:00.00",
            video_end_time="00:00:05.00",
        )

        # When: Rendering analysis HTML
        html = notifier._render_analysis_html(analysis)

        # Then: Contains key information
        assert "<ul>" in html
        assert "sequence_description" in html
        assert "Person walks by" in html
        assert "max_risk_level" in html
        assert "low" in html

    def test_render_analysis_none_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns empty string when analysis is None."""
        # Given: A notifier
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        # When: Rendering None analysis
        html = notifier._render_analysis_html(None)

        # Then: Empty string
        assert html == ""

    def test_render_none_value_shows_na(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """None values render as n/a."""
        # Given: A notifier
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        # When: Rendering None value
        result = notifier._render_value_html(None)

        # Then: Shows n/a in italics
        assert result == "<em>n/a</em>"

    def test_render_empty_list_shows_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty lists render as 'none'."""
        # Given: A notifier
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        # When: Rendering empty list
        result = notifier._render_list_html([])

        # Then: Shows none
        assert "none" in result


class TestSendGridShutdown:
    """Tests for shutdown behavior."""

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Shutdown can be called multiple times."""
        # Given: A notifier
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        # When: Calling shutdown multiple times
        await notifier.shutdown()
        await notifier.shutdown()
        await notifier.shutdown()

        # Then: Ping returns False (shutdown state observable)
        result = await notifier.ping()
        assert result is False

    @pytest.mark.asyncio
    async def test_shutdown_closes_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Shutdown closes HTTP session."""
        # Given: A notifier with active session
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)
        session = await notifier._get_session()
        assert not session.closed

        # When: Shutting down
        await notifier.shutdown()

        # Then: Session is closed
        assert session.closed
