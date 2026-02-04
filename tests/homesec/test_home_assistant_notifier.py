"""Tests for Home Assistant notifier plugin."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from homesec.models.alert import Alert
from homesec.models.config import HomeAssistantNotifierConfig
from homesec.models.vlm import EntityTimeline, SequenceAnalysis
from homesec.plugins.notifiers.home_assistant import HomeAssistantNotifier


def _make_config(**overrides: Any) -> HomeAssistantNotifierConfig:
    """Create HomeAssistantNotifierConfig with defaults."""
    defaults: dict[str, Any] = {
        "url_env": "HA_URL",
        "token_env": "HA_TOKEN",
    }
    defaults.update(overrides)
    return HomeAssistantNotifierConfig(**defaults)


def _make_analysis() -> SequenceAnalysis:
    return SequenceAnalysis(
        sequence_description="sequence",
        max_risk_level="low",
        primary_activity="normal_visitor",
        observations=["obs"],
        entities_timeline=[
            EntityTimeline(
                type="person",
                first_seen_timestamp="00:00:01.00",
                last_seen_timestamp="00:00:05.00",
                description="Person near door",
                movement="walking",
                location="front door",
                interaction="none",
            ),
            EntityTimeline(
                type="vehicle",
                first_seen_timestamp="00:00:02.00",
                last_seen_timestamp="00:00:06.00",
                description="Car in driveway",
                movement="parked",
                location="driveway",
                interaction="none",
            ),
            EntityTimeline(
                type="person",
                first_seen_timestamp="00:00:03.00",
                last_seen_timestamp="00:00:07.00",
                description="Person still present",
                movement="standing",
                location="front door",
                interaction="none",
            ),
        ],
        requires_review=False,
        frame_count=1,
        video_start_time="00:00:00.00",
        video_end_time="00:00:01.00",
    )


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
        "analysis": None,
        "detected_classes": ["person", "car", "dog"],
        "ts": datetime.now(),
        "dedupe_key": "clip_123",
        "upload_failed": False,
        "vlm_failed": False,
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
    raise_on_post: Exception | None = None,
    raise_on_get: Exception | None = None,
) -> MagicMock:
    session = MagicMock()

    if raise_on_post is not None:

        def post(*_args: Any, **_kwargs: Any) -> None:
            raise raise_on_post

        session.post = post
    elif post_cm is not None:

        def post(
            url: str, *, json: dict[str, Any], headers: dict[str, str], **_kwargs: Any
        ) -> AsyncMock:
            if capture is not None:
                capture["url"] = url
                capture["json"] = json
                capture["headers"] = headers
            return post_cm

        session.post = post

    if get_cm is not None:
        session.get = MagicMock(return_value=get_cm)
    elif raise_on_get is not None:

        def get(*_args: Any, **_kwargs: Any) -> None:
            raise raise_on_get

        session.get = get

    async def _close() -> None:
        session.closed = True

    session.close = AsyncMock(side_effect=_close)
    session.closed = False

    monkeypatch.setattr(
        "homesec.plugins.notifiers.home_assistant.aiohttp.ClientSession",
        lambda **_kw: session,
    )
    return session


class TestHomeAssistantNotifierModes:
    """Tests for supervisor vs standalone mode selection."""

    @pytest.mark.asyncio
    async def test_supervisor_mode_uses_supervisor_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Supervisor mode uses supervisor URL and token."""
        # Given: SUPERVISOR_TOKEN is set
        monkeypatch.setenv("SUPERVISOR_TOKEN", "super-token")
        config = HomeAssistantNotifierConfig()
        notifier = HomeAssistantNotifier(config)
        captured: dict[str, Any] = {}
        mock_response = _mock_http_response(200)
        _patch_session(monkeypatch, post_cm=mock_response, capture=captured)

        # When: Sending an alert
        await notifier.send(_make_alert(analysis=_make_analysis()))

        # Then: Uses supervisor URL and token
        assert captured["url"] == "http://supervisor/core/api/events/homesec_alert"
        assert captured["headers"]["Authorization"] == "Bearer super-token"
        assert captured["json"]["camera"] == "front_door"
        assert captured["json"]["clip_id"] == "clip_123"
        assert captured["json"]["detected_objects"] == ["person", "vehicle", "animal"]

        await notifier.shutdown()

    @pytest.mark.asyncio
    async def test_standalone_mode_uses_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Standalone mode uses configured env vars for URL and token."""
        # Given: Standalone mode with url/token env vars
        monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
        monkeypatch.setenv("HA_URL", "http://ha.local:8123")
        monkeypatch.setenv("HA_TOKEN", "ha-token")
        config = _make_config()
        notifier = HomeAssistantNotifier(config)
        captured: dict[str, Any] = {}
        mock_response = _mock_http_response(200)
        _patch_session(monkeypatch, post_cm=mock_response, capture=captured)

        # When: Sending an alert
        await notifier.send(_make_alert())

        # Then: Uses standalone URL and token
        assert captured["url"] == "http://ha.local:8123/api/events/homesec_alert"
        assert captured["headers"]["Authorization"] == "Bearer ha-token"

        await notifier.shutdown()

    def test_missing_env_config_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Standalone mode requires url_env and token_env."""
        # Given: Standalone mode with missing url_env
        monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
        config = HomeAssistantNotifierConfig(token_env="HA_TOKEN")

        # When/Then: Creating notifier raises ValueError
        with pytest.raises(ValueError, match="url_env"):
            HomeAssistantNotifier(config)


class TestHomeAssistantNotifierSending:
    """Tests for sending events and error handling."""

    @pytest.mark.asyncio
    async def test_send_raises_on_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HTTP errors raise for retry handling."""
        # Given: Standalone notifier with HA returning 401
        monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
        monkeypatch.setenv("HA_URL", "http://ha.local:8123")
        monkeypatch.setenv("HA_TOKEN", "ha-token")
        notifier = HomeAssistantNotifier(_make_config())
        mock_response = _mock_http_response(401)
        _patch_session(monkeypatch, post_cm=mock_response)

        # When/Then: Sending an alert raises
        with pytest.raises(RuntimeError, match="HTTP 401"):
            await notifier.send(_make_alert())

        await notifier.shutdown()

    @pytest.mark.asyncio
    async def test_send_raises_on_connection_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Connection errors raise for retry handling."""
        # Given: Standalone notifier with connection error
        monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
        monkeypatch.setenv("HA_URL", "http://ha.local:8123")
        monkeypatch.setenv("HA_TOKEN", "ha-token")
        notifier = HomeAssistantNotifier(_make_config())
        _patch_session(monkeypatch, raise_on_post=aiohttp.ClientConnectionError("no route"))

        # When/Then: Sending an alert raises
        with pytest.raises(aiohttp.ClientConnectionError, match="no route"):
            await notifier.send(_make_alert())

        await notifier.shutdown()

    @pytest.mark.asyncio
    async def test_send_raises_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Timeouts raise a clearer exception."""
        # Given: Standalone notifier with timeout on post
        monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
        monkeypatch.setenv("HA_URL", "http://ha.local:8123")
        monkeypatch.setenv("HA_TOKEN", "ha-token")
        notifier = HomeAssistantNotifier(_make_config())
        _patch_session(monkeypatch, raise_on_post=asyncio.TimeoutError())

        # When/Then: Sending an alert raises timeout
        with pytest.raises(asyncio.TimeoutError, match="timed out"):
            await notifier.send(_make_alert())

        await notifier.shutdown()

    @pytest.mark.asyncio
    async def test_send_raises_when_token_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing auth token raises before HTTP request."""
        # Given: Standalone notifier missing token env value
        monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
        monkeypatch.setenv("HA_URL", "http://ha.local:8123")
        monkeypatch.delenv("HA_TOKEN", raising=False)
        notifier = HomeAssistantNotifier(_make_config())

        # When/Then: Sending an alert raises token error
        with pytest.raises(RuntimeError, match="token is missing"):
            await notifier.send(_make_alert())

    @pytest.mark.asyncio
    async def test_send_raises_when_url_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing URL raises before HTTP request."""
        # Given: Standalone notifier missing URL env value
        monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
        monkeypatch.delenv("HA_URL", raising=False)
        monkeypatch.setenv("HA_TOKEN", "ha-token")
        notifier = HomeAssistantNotifier(_make_config())

        # When/Then: Sending an alert raises URL error
        with pytest.raises(RuntimeError, match="URL is missing"):
            await notifier.send(_make_alert())

    @pytest.mark.asyncio
    async def test_send_raises_after_shutdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Send raises after shutdown is called."""
        # Given: Standalone notifier that has been shut down
        monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
        monkeypatch.setenv("HA_URL", "http://ha.local:8123")
        monkeypatch.setenv("HA_TOKEN", "ha-token")
        notifier = HomeAssistantNotifier(_make_config())
        await notifier.shutdown()

        # When/Then: Sending after shutdown raises
        with pytest.raises(RuntimeError, match="shut down"):
            await notifier.send(_make_alert())

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Shutdown can be called multiple times safely."""
        # Given: Standalone notifier with an active session
        monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
        monkeypatch.setenv("HA_URL", "http://ha.local:8123")
        monkeypatch.setenv("HA_TOKEN", "ha-token")
        notifier = HomeAssistantNotifier(_make_config())
        mock_response = _mock_http_response(200)
        session = _patch_session(monkeypatch, post_cm=mock_response)
        await notifier.send(_make_alert())

        # When: Calling shutdown multiple times
        await notifier.shutdown()
        await notifier.shutdown()
        await notifier.shutdown()

        # Then: Session is closed (shutdown state observable)
        assert session.closed is True


class TestHomeAssistantNotifierPing:
    """Tests for notifier ping behavior."""

    @pytest.mark.asyncio
    async def test_ping_returns_true_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ping returns True when API responds OK."""
        # Given: Standalone notifier with healthy API
        monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
        monkeypatch.setenv("HA_URL", "http://ha.local:8123")
        monkeypatch.setenv("HA_TOKEN", "ha-token")
        notifier = HomeAssistantNotifier(_make_config())
        mock_response = _mock_http_response(200)
        _patch_session(monkeypatch, get_cm=mock_response)

        # When: Pinging
        result = await notifier.ping()

        # Then: Ping is healthy
        assert result is True

        await notifier.shutdown()

    @pytest.mark.asyncio
    async def test_ping_returns_false_on_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ping returns False when API responds with error."""
        # Given: Standalone notifier with API error
        monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
        monkeypatch.setenv("HA_URL", "http://ha.local:8123")
        monkeypatch.setenv("HA_TOKEN", "ha-token")
        notifier = HomeAssistantNotifier(_make_config())
        mock_response = _mock_http_response(401)
        _patch_session(monkeypatch, get_cm=mock_response)

        # When: Pinging
        result = await notifier.ping()

        # Then: Ping is unhealthy
        assert result is False

        await notifier.shutdown()

    @pytest.mark.asyncio
    async def test_ping_returns_false_on_connection_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ping returns False when API is unreachable."""
        # Given: Standalone notifier with connection error
        monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
        monkeypatch.setenv("HA_URL", "http://ha.local:8123")
        monkeypatch.setenv("HA_TOKEN", "ha-token")
        notifier = HomeAssistantNotifier(_make_config())
        _patch_session(monkeypatch, raise_on_get=aiohttp.ClientConnectionError("no route"))

        # When: Pinging
        result = await notifier.ping()

        # Then: Ping is unhealthy
        assert result is False

        await notifier.shutdown()

    @pytest.mark.asyncio
    async def test_ping_returns_false_after_shutdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ping returns False after shutdown."""
        # Given: Standalone notifier that has been shut down
        monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
        monkeypatch.setenv("HA_URL", "http://ha.local:8123")
        monkeypatch.setenv("HA_TOKEN", "ha-token")
        notifier = HomeAssistantNotifier(_make_config())
        await notifier.shutdown()

        # When: Pinging after shutdown
        result = await notifier.ping()

        # Then: Ping is unhealthy
        assert result is False

    @pytest.mark.asyncio
    async def test_ping_returns_false_when_url_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ping returns False when URL env is missing."""
        # Given: Standalone notifier missing URL env value
        monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
        monkeypatch.delenv("HA_URL", raising=False)
        monkeypatch.setenv("HA_TOKEN", "ha-token")
        notifier = HomeAssistantNotifier(_make_config())

        # When: Pinging without URL
        result = await notifier.ping()

        # Then: Ping is unhealthy
        assert result is False

    @pytest.mark.asyncio
    async def test_ping_returns_false_when_token_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ping returns False when token env is missing."""
        # Given: Standalone notifier missing token env value
        monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
        monkeypatch.setenv("HA_URL", "http://ha.local:8123")
        monkeypatch.delenv("HA_TOKEN", raising=False)
        notifier = HomeAssistantNotifier(_make_config())

        # When: Pinging without token
        result = await notifier.ping()

        # Then: Ping is unhealthy
        assert result is False


class TestHomeAssistantNotifierDetectedObjects:
    """Tests for detected_objects normalization."""

    @pytest.mark.asyncio
    async def test_detected_objects_normalization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Detected objects normalize and preserve stable ordering."""
        # Given: A notifier and alert with mixed detected classes
        monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
        monkeypatch.setenv("HA_URL", "http://ha.local:8123")
        monkeypatch.setenv("HA_TOKEN", "ha-token")
        notifier = HomeAssistantNotifier(_make_config())
        captured: dict[str, Any] = {}
        mock_response = _mock_http_response(200)
        _patch_session(monkeypatch, post_cm=mock_response, capture=captured)
        alert = _make_alert(detected_classes=["PACKAGE", "bicycle", "unknown", "umbrella", "dog"])

        # When: Sending an alert
        await notifier.send(alert)

        # Then: Normalized detected_objects are stable and ordered
        assert captured["json"]["detected_objects"] == [
            "vehicle",
            "animal",
            "package",
            "object",
            "unknown",
        ]

        await notifier.shutdown()

    @pytest.mark.asyncio
    async def test_detected_objects_empty_when_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Detected objects is empty when no classes are provided."""
        # Given: A notifier and alert without detected classes
        monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
        monkeypatch.setenv("HA_URL", "http://ha.local:8123")
        monkeypatch.setenv("HA_TOKEN", "ha-token")
        notifier = HomeAssistantNotifier(_make_config())
        captured: dict[str, Any] = {}
        mock_response = _mock_http_response(200)
        _patch_session(monkeypatch, post_cm=mock_response, capture=captured)
        alert = _make_alert(detected_classes=None)

        # When: Sending an alert
        await notifier.send(alert)

        # Then: detected_objects is empty
        assert captured["json"]["detected_objects"] == []

        await notifier.shutdown()
