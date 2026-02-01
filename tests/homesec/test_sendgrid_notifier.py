"""Tests for SendGrid email notifier."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from homesec.models.alert import Alert
from homesec.plugins.notifiers.sendgrid_email import SendGridEmailConfig
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
    response.read = AsyncMock(return_value=b"{}")
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    return response


def _patch_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    post_cm: AsyncMock | None = None,
    get_cm: AsyncMock | None = None,
    capture: dict[str, Any] | None = None,
) -> MagicMock:
    session = MagicMock()

    if post_cm is not None:

        def capture_post(url: str, json: dict[str, Any], headers: dict[str, str]) -> AsyncMock:
            if capture is not None:
                capture["url"] = url
                capture["json"] = json
                capture["headers"] = headers
            return post_cm

        session.post = capture_post

    if get_cm is not None:
        session.get = MagicMock(return_value=get_cm)

    async def _close() -> None:
        session.closed = True

    session.close = AsyncMock(side_effect=_close)
    session.closed = False

    monkeypatch.setattr(
        "homesec.plugins.notifiers.sendgrid_email.aiohttp.ClientSession",
        lambda **_kw: session,
    )
    return session


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
        _patch_session(monkeypatch, post_cm=mock_response, capture=captured_request)

        # When: Sending an alert
        await notifier.send(_make_alert(camera_name="back_yard", clip_id="clip_456"))

        # Then: API was called with correct structure
        assert "mail/send" in captured_request["url"]
        assert captured_request["headers"]["Authorization"] == "Bearer test-api-key"

        payload = captured_request["json"]

        # Then: Sender is correct
        assert payload["from"]["email"] == "sender@example.com"

        # Then: Recipients are correct and cc/bcc omitted by default
        personalizations = payload["personalizations"]
        assert len(personalizations) == 1
        assert personalizations[0]["to"] == [{"email": "to@example.com"}]
        assert "cc" not in personalizations[0]
        assert "bcc" not in personalizations[0]

        # Then: Subject was templated
        assert personalizations[0]["subject"] == "Alert: back_yard"

        # Then: Content includes both text and HTML
        content = payload["content"]
        text_content = next(c for c in content if c["type"] == "text/plain")
        html_content = next(c for c in content if c["type"] == "text/html")
        assert text_content["value"] == "Clip clip_456"
        assert html_content["value"] == "<b>clip_456</b>"

        await notifier.shutdown()

    @pytest.mark.asyncio
    async def test_send_includes_cc_and_bcc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Send includes cc/bcc recipients when configured."""
        # Given: A notifier with to, cc, bcc recipients
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config(
            to_emails=["to1@example.com", "to2@example.com"],
            cc_emails=["cc@example.com"],
            bcc_emails=["bcc@example.com"],
        )
        notifier = SendGridEmailNotifier(config)

        captured_request: dict[str, Any] = {}
        mock_response = _mock_http_response(202)
        _patch_session(monkeypatch, post_cm=mock_response, capture=captured_request)

        # When: Sending an alert
        await notifier.send(_make_alert())

        # Then: cc and bcc are included
        personalizations = captured_request["json"]["personalizations"]
        assert personalizations[0]["to"] == [
            {"email": "to1@example.com"},
            {"email": "to2@example.com"},
        ]
        assert personalizations[0]["cc"] == [{"email": "cc@example.com"}]
        assert personalizations[0]["bcc"] == [{"email": "bcc@example.com"}]

        await notifier.shutdown()

    @pytest.mark.asyncio
    async def test_send_text_only_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Send includes only text content when HTML template is empty."""
        # Given: A notifier with text-only template
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config(html_template="")
        notifier = SendGridEmailNotifier(config)

        captured_request: dict[str, Any] = {}
        mock_response = _mock_http_response(202)
        _patch_session(monkeypatch, post_cm=mock_response, capture=captured_request)

        # When: Sending an alert
        await notifier.send(_make_alert())

        # Then: Only text content is included
        content = captured_request["json"]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text/plain"

        await notifier.shutdown()

    @pytest.mark.asyncio
    async def test_send_html_only_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Send includes only HTML content when text template is empty."""
        # Given: A notifier with HTML-only template
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config(text_template="")
        notifier = SendGridEmailNotifier(config)

        captured_request: dict[str, Any] = {}
        mock_response = _mock_http_response(202)
        _patch_session(monkeypatch, post_cm=mock_response, capture=captured_request)

        # When: Sending an alert
        await notifier.send(_make_alert())

        # Then: Only HTML content is included
        content = captured_request["json"]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text/html"

        await notifier.shutdown()

    @pytest.mark.asyncio
    async def test_send_includes_sender_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Send includes sender name when configured."""
        # Given: A notifier with from_name configured
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config(from_name="HomeSec Alerts")
        notifier = SendGridEmailNotifier(config)

        captured_request: dict[str, Any] = {}
        mock_response = _mock_http_response(202)
        _patch_session(monkeypatch, post_cm=mock_response, capture=captured_request)

        # When: Sending an alert
        await notifier.send(_make_alert())

        # Then: Sender includes name
        sender = captured_request["json"]["from"]
        assert sender["email"] == "sender@example.com"
        assert sender["name"] == "HomeSec Alerts"

        await notifier.shutdown()

    @pytest.mark.asyncio
    async def test_send_handles_none_analysis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Send handles alert without analysis gracefully."""
        # Given: A notifier with analysis_html in template and alert without analysis
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config(html_template="Analysis: {analysis_html}")
        notifier = SendGridEmailNotifier(config)

        captured_request: dict[str, Any] = {}
        mock_response = _mock_http_response(202)
        _patch_session(monkeypatch, post_cm=mock_response, capture=captured_request)

        # When: Sending an alert without analysis
        alert = _make_alert()
        assert alert.analysis is None
        await notifier.send(alert)

        # Then: Request succeeds and analysis_html is empty or placeholder
        content = captured_request["json"]["content"]
        html_content = next(c for c in content if c["type"] == "text/html")
        assert "Analysis:" in html_content["value"]

        await notifier.shutdown()

    @pytest.mark.asyncio
    async def test_send_raises_on_api_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises RuntimeError with status code on API failure."""
        # Given: A SendGrid notifier with API returning 400
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        mock_response = _mock_http_response(400)
        _patch_session(monkeypatch, post_cm=mock_response)

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

        mock_response = _mock_http_response(200)
        _patch_session(monkeypatch, get_cm=mock_response)

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

        mock_response = _mock_http_response(401)
        _patch_session(monkeypatch, get_cm=mock_response)

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


class TestSendGridTemplateRendering:
    """Tests for template rendering via public send behavior."""

    @pytest.mark.asyncio
    async def test_send_escapes_analysis_html(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Analysis HTML escapes special characters."""
        # Given: An alert with analysis content that includes HTML
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config(html_template="{analysis_html}")
        notifier = SendGridEmailNotifier(config)

        analysis = SequenceAnalysis(
            sequence_description="<script>alert('x')</script>",
            max_risk_level="low",
            primary_activity="passerby",
            observations=[],
            entities_timeline=[],
            requires_review=False,
            frame_count=1,
            video_start_time="00:00:00.00",
            video_end_time="00:00:01.00",
        )
        alert = _make_alert(analysis=analysis)

        captured_request: dict[str, Any] = {}
        mock_response = _mock_http_response(202)
        _patch_session(monkeypatch, post_cm=mock_response, capture=captured_request)

        # When: Sending the alert
        await notifier.send(alert)

        # Then: Analysis HTML is escaped
        content = captured_request["json"]["content"]
        html_content = next(c for c in content if c["type"] == "text/html")
        assert "<script>" not in html_content["value"]
        assert "&lt;script&gt;" in html_content["value"]
        assert "none" in html_content["value"]

        await notifier.shutdown()


class TestSendGridShutdown:
    """Tests for shutdown behavior."""

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Shutdown can be called multiple times."""
        # Given: A notifier with an active session
        monkeypatch.setenv("SENDGRID_API_KEY", "test-api-key")
        config = _make_config()
        notifier = SendGridEmailNotifier(config)

        mock_response = _mock_http_response(202)
        session = _patch_session(monkeypatch, post_cm=mock_response)

        await notifier.send(_make_alert())

        # When: Calling shutdown multiple times
        await notifier.shutdown()
        await notifier.shutdown()
        await notifier.shutdown()

        # Then: Session is closed (shutdown state observable)
        assert session.closed is True
