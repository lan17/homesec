"""Tests for MQTT notifier plugin."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from homesec.models.alert import Alert
from homesec.models.config import MQTTAuthConfig, MQTTConfig
from homesec.plugins.notifiers.mqtt import MQTTNotifier


class _FakePublishResult:
    def __init__(self) -> None:
        self.waited = False

    def wait_for_publish(self) -> None:
        self.waited = True


class _FakeClient:
    def __init__(self) -> None:
        self.on_connect = None
        self.on_disconnect = None
        self.connected = False
        self.published: list[tuple[str, str, int, bool]] = []
        self.username: str | None = None
        self.password: str | None = None

    def username_pw_set(self, username: str, password: str) -> None:
        self.username = username
        self.password = password

    def connect(
        self, _host: str, _port: int, _keepalive: int | None = None, **_kwargs: Any
    ) -> None:
        self.connected = True
        if self.on_connect is not None:
            self.on_connect(self, None, {}, 0)

    def loop_start(self) -> None:
        return None

    def loop_stop(self) -> None:
        return None

    def disconnect(self) -> None:
        self.connected = False
        if self.on_disconnect is not None:
            self.on_disconnect(self, None, 0)

    def is_connected(self) -> bool:
        return self.connected

    def publish(self, topic: str, payload: str, qos: int, retain: bool) -> _FakePublishResult:
        self.published.append((topic, payload, qos, retain))
        return _FakePublishResult()


class _FakeClientNoConnect(_FakeClient):
    def connect(
        self, _host: str, _port: int, _keepalive: int | None = None, **_kwargs: Any
    ) -> None:
        self.connected = False


def _sample_alert() -> Alert:
    return Alert(
        clip_id="clip_1",
        camera_name="front",
        storage_uri="mock://clip_1",
        view_url="http://example/clip_1",
        risk_level="low",
        activity_type="passerby",
        notify_reason="risk_level=low",
        summary="summary",
        ts=datetime.now(),
        dedupe_key="clip_1",
        upload_failed=False,
    )


@pytest.mark.asyncio
async def test_mqtt_notifier_publishes_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    """MQTT notifier should publish JSON payload to templated topic."""
    # Given a notifier with stubbed MQTT client and auth env vars
    fake_client = _FakeClient()
    monkeypatch.setattr("homesec.plugins.notifiers.mqtt.mqtt.Client", lambda: fake_client)
    monkeypatch.setenv("MQTT_USER", "user")
    monkeypatch.setenv("MQTT_PASS", "pass")
    config = MQTTConfig(
        host="localhost",
        topic_template="alerts/{camera_name}",
        auth=MQTTAuthConfig(username_env="MQTT_USER", password_env="MQTT_PASS"),
    )
    notifier = MQTTNotifier(config)

    # When sending an alert
    await notifier.send(_sample_alert())

    # Then publish is called with JSON payload and templated topic
    assert fake_client.username == "user"
    assert fake_client.password == "pass"
    assert len(fake_client.published) == 1
    topic, payload, qos, retain = fake_client.published[0]
    assert topic == "alerts/front"
    assert '"clip_id":"clip_1"' in payload
    assert qos == config.qos
    assert retain == config.retain


@pytest.mark.asyncio
async def test_mqtt_notifier_raises_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """MQTT notifier should raise when connection times out."""
    # Given a notifier whose client never connects
    fake_client = _FakeClientNoConnect()
    monkeypatch.setattr("homesec.plugins.notifiers.mqtt.mqtt.Client", lambda: fake_client)
    config = MQTTConfig(host="localhost", connection_timeout=0.01)
    notifier = MQTTNotifier(config)

    # When sending an alert
    with pytest.raises(RuntimeError):
        # Then a connection timeout error is raised
        await notifier.send(_sample_alert())
