"""Tests for notifier implementations."""

from __future__ import annotations

from datetime import datetime

import pytest

from homesec.models.alert import Alert
from homesec.notifiers.multiplex import MultiplexNotifier, NotifierEntry
from homesec.plugins.notifiers.sendgrid_email import SendGridEmailConfig
from homesec.models.vlm import SequenceAnalysis
from homesec.plugins.notifiers.sendgrid_email import SendGridEmailNotifier


class _StubNotifier:
    def __init__(
        self,
        *,
        should_fail_send: bool = False,
        should_fail_ping: bool = False,
        should_fail_shutdown: bool = False,
    ) -> None:
        self.should_fail_send = should_fail_send
        self.should_fail_ping = should_fail_ping
        self.should_fail_shutdown = should_fail_shutdown
        self.sent: list[Alert] = []

    async def send(self, alert: Alert) -> None:
        if self.should_fail_send:
            raise RuntimeError("send failed")
        self.sent.append(alert)

    async def ping(self) -> bool:
        if self.should_fail_ping:
            return False
        return True

    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout
        if self.should_fail_shutdown:
            raise RuntimeError("shutdown failed")


def _sample_alert() -> Alert:
    return Alert(
        clip_id="clip_123",
        camera_name="front",
        storage_uri="mock://clip_123",
        view_url="http://example.com/clip_123",
        risk_level="low",
        activity_type="delivery",
        notify_reason="risk_level=low",
        summary="Delivery",
        ts=datetime.now(),
        dedupe_key="clip_123",
        upload_failed=False,
    )


@pytest.mark.asyncio
async def test_multiplex_notifier_fans_out_success() -> None:
    """MultiplexNotifier sends to all notifiers on success."""
    # Given two healthy notifiers
    notifier_a = _StubNotifier()
    notifier_b = _StubNotifier()
    mux = MultiplexNotifier(
        [
            NotifierEntry(name="a", notifier=notifier_a),
            NotifierEntry(name="b", notifier=notifier_b),
        ]
    )

    # When sending an alert
    await mux.send(_sample_alert())

    # Then both notifiers receive the alert
    assert len(notifier_a.sent) == 1
    assert len(notifier_b.sent) == 1


@pytest.mark.asyncio
async def test_multiplex_notifier_raises_on_failure() -> None:
    """MultiplexNotifier raises when any notifier fails."""
    # Given one notifier fails
    notifier_ok = _StubNotifier()
    notifier_fail = _StubNotifier(should_fail_send=True)
    mux = MultiplexNotifier(
        [
            NotifierEntry(name="ok", notifier=notifier_ok),
            NotifierEntry(name="fail", notifier=notifier_fail),
        ]
    )

    # When sending an alert
    with pytest.raises(RuntimeError):
        await mux.send(_sample_alert())

    # Then successful notifier still received the alert
    assert len(notifier_ok.sent) == 1


@pytest.mark.asyncio
async def test_multiplex_notifier_ping_aggregates() -> None:
    """MultiplexNotifier returns False if any notifier is unhealthy."""
    # Given one notifier reports unhealthy
    notifier_ok = _StubNotifier()
    notifier_bad = _StubNotifier(should_fail_ping=True)
    mux = MultiplexNotifier(
        [
            NotifierEntry(name="ok", notifier=notifier_ok),
            NotifierEntry(name="bad", notifier=notifier_bad),
        ]
    )

    # When pinging
    healthy = await mux.ping()

    # Then overall health is False
    assert healthy is False


@pytest.mark.asyncio
async def test_multiplex_notifier_shutdown_is_resilient() -> None:
    """MultiplexNotifier shutdown should not raise on individual failures."""
    # Given a notifier that fails on shutdown
    notifier_ok = _StubNotifier()
    notifier_bad = _StubNotifier(should_fail_shutdown=True)
    mux = MultiplexNotifier(
        [
            NotifierEntry(name="ok", notifier=notifier_ok),
            NotifierEntry(name="bad", notifier=notifier_bad),
        ]
    )

    # When shutting down
    await mux.shutdown()

    # Then close completes without exception
    assert True


def test_sendgrid_templates_render() -> None:
    """SendGrid notifier renders templates with alert context."""
    # Given a SendGrid notifier with templates
    config = SendGridEmailConfig(
        from_email="sender@example.com",
        to_emails=["to@example.com"],
        subject_template="Alert {camera_name}",
        text_template="Clip {clip_id} {view_url}",
        html_template="<b>{clip_id}</b> {analysis_html}",
    )
    notifier = SendGridEmailNotifier(config)
    analysis = SequenceAnalysis(
        sequence_description="desc",
        max_risk_level="low",
        primary_activity="passerby",
        observations=["obs"],
        entities_timeline=[],
        requires_review=False,
        frame_count=1,
        video_start_time="00:00:00.00",
        video_end_time="00:00:01.00",
    )
    alert = Alert(
        clip_id="clip_1",
        camera_name="front",
        storage_uri="mock://clip_1",
        view_url=None,
        risk_level="low",
        activity_type="passerby",
        notify_reason="risk_level=low",
        summary="summary",
        analysis=analysis,
        ts=datetime.now(),
        dedupe_key="clip_1",
        upload_failed=False,
    )

    # When rendering templates
    subject = notifier._render_subject(alert)
    text = notifier._render_text(alert)
    html = notifier._render_html(alert)

    # Then rendered outputs include key fields
    assert "Alert front" in subject
    assert "clip_1" in text
    assert "clip_1" in html
