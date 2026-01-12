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


class TestSendGridNotifierSend:
    """Tests for the send method."""

    @pytest.mark.asyncio
    async def test_send_success_with_mocked_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Successfully sends email via SendGrid API."""
        # Given: A SendGrid notifier with mocked _get_session
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        mock_response = AsyncMock()
        mock_response.status = 202
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        async def fake_get_session() -> MagicMock:
            return mock_session

        monkeypatch.setattr(notifier, "_get_session", fake_get_session)

        # When: Sending an alert
        await notifier.send(_make_alert())

        # Then: API was called
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "mail/send" in call_args[0][0]
        assert "Bearer test-api-key" in call_args[1]["headers"]["Authorization"]

    @pytest.mark.asyncio
    async def test_send_fails_on_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises RuntimeError on API error."""
        # Given: A SendGrid notifier with API returning 400
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad Request")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        async def fake_get_session() -> MagicMock:
            return mock_session

        monkeypatch.setattr(notifier, "_get_session", fake_get_session)

        # When/Then: Send raises RuntimeError
        with pytest.raises(RuntimeError, match="400"):
            await notifier.send(_make_alert())

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
    async def test_ping_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

        async def fake_get_session() -> MagicMock:
            return mock_session

        monkeypatch.setattr(notifier, "_get_session", fake_get_session)

        # When: Pinging
        result = await notifier.ping()

        # Then: Returns True
        assert result is True

    @pytest.mark.asyncio
    async def test_ping_fails_on_api_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

        async def fake_get_session() -> MagicMock:
            return mock_session

        monkeypatch.setattr(notifier, "_get_session", fake_get_session)

        # When: Pinging
        result = await notifier.ping()

        # Then: Returns False
        assert result is False

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
    """Tests for payload construction."""

    def test_builds_payload_with_all_recipients(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Payload includes all recipients."""
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

        # Then: All recipients are included
        personalizations = payload["personalizations"]
        assert isinstance(personalizations, list)
        personalization = personalizations[0]
        assert isinstance(personalization, dict)
        assert len(personalization["to"]) == 2  # type: ignore[arg-type]
        assert len(personalization["cc"]) == 1  # type: ignore[arg-type]
        assert len(personalization["bcc"]) == 1  # type: ignore[arg-type]

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
        assert isinstance(personalizations, list)
        personalization = personalizations[0]
        assert isinstance(personalization, dict)
        assert "cc" not in personalization

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
        assert isinstance(content, list)
        assert len(content) == 1
        assert content[0]["type"] == "text/plain"  # type: ignore[index]

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
        assert isinstance(content, list)
        assert len(content) == 1
        assert content[0]["type"] == "text/html"  # type: ignore[index]

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
        assert isinstance(sender, dict)
        assert sender["name"] == "HomeSec Alerts"


class TestSendGridTemplateRendering:
    """Tests for template rendering."""

    def test_render_subject_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Subject template substitutes variables."""
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
        """HTML escapes special characters in values."""
        # Given: A notifier
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        # When: Rendering value with special chars
        result = notifier._render_value_html("<script>alert('xss')</script>")

        # Then: HTML is escaped
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_render_analysis_nested_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Renders nested dict as HTML list."""
        # Given: A notifier with analysis containing nested data
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

        # Then: Contains key information as HTML
        assert "<ul>" in html
        assert "sequence_description" in html
        assert "Person walks by" in html

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

    def test_render_value_none_shows_na(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """None values render as n/a."""
        # Given: A notifier
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        # When: Rendering None value
        result = notifier._render_value_html(None)

        # Then: Shows n/a
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

        # Then: No exception

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
