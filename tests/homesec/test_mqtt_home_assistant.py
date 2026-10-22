"""Tests for Home Assistant MQTT discovery support in the MQTT notifier."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from homesec.models.alert import Alert
from homesec.plugins.notifiers.mqtt import MQTTConfig, MQTTNotifier
from homesec.plugins.notifiers.mqtt_home_assistant import (
    EVENT_TYPES,
    HomeAssistantConfig,
    build_discovery,
    build_event_payload,
    camera_slug,
)

_TOPIC_SAFE = re.compile(r"^[a-z0-9_]+$")


class _FakePublishResult:
    def __init__(self, published: bool = True) -> None:
        self.published = published
        self.waits: list[float | None] = []

    def wait_for_publish(self, timeout: float | None = None) -> None:
        self.waits.append(timeout)

    def is_published(self) -> bool:
        return self.published


class _FakeClient:
    def __init__(self) -> None:
        self.on_connect: Any = None
        self.on_disconnect: Any = None
        self.connected = False
        self.will: tuple[str, str, int, bool] | None = None
        self.published: list[tuple[str, str, int, bool]] = []
        self.after_publish: Callable[[str], None] | None = None
        self.results: dict[str, _FakePublishResult] = {}
        self.calls: list[str] = []

    def username_pw_set(self, username: str, password: str) -> None:
        _ = (username, password)

    def will_set(self, topic: str, payload: str, qos: int = 0, retain: bool = False) -> None:
        self.will = (topic, payload, qos, retain)

    def connect(self, _host: str, _port: int, _keepalive: int | None = None, **_kw: Any) -> None:
        self.connected = True
        if self.on_connect is not None:
            self.on_connect(self, None, {}, 0)

    def reconnect(self) -> None:
        if self.on_connect is not None:
            self.on_connect(self, None, {}, 0)

    def loop_start(self) -> None:
        return None

    def loop_stop(self) -> None:
        return None

    def disconnect(self) -> None:
        self.connected = False
        self.calls.append("disconnect")

    def is_connected(self) -> bool:
        return self.connected

    def publish(self, topic: str, payload: str, qos: int, retain: bool) -> _FakePublishResult:
        if not topic or any(char in topic for char in ("+", "#", "\x00")):
            raise ValueError("Invalid MQTT publish topic")
        self.published.append((topic, payload, qos, retain))
        if self.after_publish is not None:
            self.after_publish(topic)
        return self.results.get(topic, _FakePublishResult())

    def messages(self, topic: str) -> list[tuple[str, int, bool]]:
        return [(p, q, r) for t, p, q, r in self.published if t == topic]


def _alert(camera_name: str = "Front Door", risk_level: str | None = "high") -> Alert:
    return Alert(
        clip_id="clip_1",
        camera_name=camera_name,
        storage_uri="mock://clip_1",
        view_url="http://example/clip_1",
        risk_level=risk_level,  # type: ignore[arg-type]
        activity_type="person",
        notify_reason="risk_level=high",
        summary="Person at the door",
        ts=datetime(2026, 1, 2, 3, 4, 5),
        dedupe_key="clip_1",
        upload_failed=False,
    )


def _notifier(
    monkeypatch: pytest.MonkeyPatch, home_assistant: HomeAssistantConfig | None
) -> tuple[MQTTNotifier, _FakeClient]:
    client = _FakeClient()
    monkeypatch.setattr("homesec.plugins.notifiers.mqtt.mqtt.Client", lambda: client)
    config = MQTTConfig(host="localhost", home_assistant=home_assistant)
    return MQTTNotifier(config), client


def _ha_config(**overrides: Any) -> HomeAssistantConfig:
    return HomeAssistantConfig(discovery_settle_seconds=0, **overrides)


class TestHomeAssistantConfig:
    def test_defaults_match_documented_topics(self) -> None:
        # Given/When: Creating the config with no overrides
        config = HomeAssistantConfig()

        # Then: Documented default topics and a non-zero discovery settle delay apply
        assert config.discovery_prefix == "homeassistant"
        assert config.event_topic_template == "homecam/events/{camera_slug}"
        assert config.availability_topic == "homecam/status"
        assert config.discovery_settle_seconds > 0

    @pytest.mark.parametrize(
        "overrides",
        [
            {"availability_topic": "homecam/#"},
            {"availability_topic": "homecam/+/status"},
            {"availability_topic": "/leading"},
            {"discovery_prefix": ""},
            {"discovery_prefix": "bad/"},
            {"event_topic_template": "homecam/events"},
            {"event_topic_template": "homecam/{camera_slug}/{unknown}"},
            {"event_topic_template": "homecam/#/{camera_slug}"},
            {"event_topic_template": "homecam/events/{{camera_slug}}"},
            {"event_topic_template": "homecam/{{camera_slug}}/{camera_slug[0]}"},
            {"event_topic_template": "homecam/{camera_slug}/{camera_slug:.1}"},
        ],
    )
    def test_rejects_unpublishable_topics(self, overrides: dict[str, str]) -> None:
        # Given: A topic setting that is not valid for publishing
        # When/Then: Validation fails at the config boundary
        with pytest.raises(ValidationError):
            HomeAssistantConfig(**overrides)

    def test_rejects_negative_settle_delay(self) -> None:
        # Given/When/Then: A negative settle delay is invalid
        with pytest.raises(ValidationError):
            HomeAssistantConfig(discovery_settle_seconds=-1)


class TestCameraSlug:
    @pytest.mark.parametrize("name", ["Front Door", "front/door", "Café ☕", "a#b+c", "", "///"])
    def test_slug_is_topic_and_id_safe(self, name: str) -> None:
        # Given: Camera names with spaces, separators, wildcards, or non-ASCII text
        # When: Slugging them
        slug = camera_slug(name)

        # Then: The slug contains only lowercase letters, digits, and underscores
        assert _TOPIC_SAFE.match(slug)

    def test_names_that_normalize_alike_stay_distinct(self) -> None:
        # Given: Names that would collide after lowercasing and replacing punctuation
        names = ["Front Door", "front-door", "front_door", "FRONT DOOR!"]

        # When: Slugging each
        slugs = {camera_slug(name) for name in names}

        # Then: Every camera keeps a unique identity
        assert len(slugs) == len(names)

    def test_slug_is_stable_across_calls(self) -> None:
        # Given/When/Then: The same name always maps to the same slug
        assert camera_slug("Garage") == camera_slug("Garage")

    def test_long_names_are_bounded(self) -> None:
        # Given: A very long camera name
        # When/Then: The slug stays short enough for topic and ID use
        assert len(camera_slug("x" * 500)) <= 50


class TestDiscoveryPayload:
    def test_discovery_describes_a_camera_event_entity(self) -> None:
        # Given: A configured camera
        config = _ha_config()

        # When: Building its discovery message
        topic, payload = build_discovery("Front Door", config)
        data = json.loads(payload.model_dump_json())

        # Then: The message targets the event platform with identity, device, and availability
        slug = camera_slug("Front Door")
        assert topic == f"homeassistant/event/homesec_{slug}/config"
        assert data["unique_id"] == f"homesec_{slug}_alerts"
        assert data["state_topic"] == f"homecam/events/{slug}"
        assert data["availability_topic"] == "homecam/status"
        assert data["event_types"] == EVENT_TYPES
        assert data["device"]["identifiers"] == [f"homesec_{slug}"]
        assert data["device"]["name"] == "Front Door"
        assert data["origin"]["name"] == "HomeSec"

    def test_discovery_uses_configured_prefix_and_topics(self) -> None:
        # Given: Custom discovery and topic settings
        config = _ha_config(
            discovery_prefix="ha",
            event_topic_template="cams/{camera_slug}/event",
            availability_topic="cams/status",
        )

        # When: Building discovery
        topic, payload = build_discovery("Yard", config)

        # Then: The custom values are used
        slug = camera_slug("Yard")
        assert topic == f"ha/event/homesec_{slug}/config"
        assert payload.state_topic == f"cams/{slug}/event"
        assert payload.availability_topic == "cams/status"


class TestEventPayload:
    @pytest.mark.parametrize("tz", [None, timezone.utc, timezone(timedelta(hours=9))])
    def test_event_timestamp_has_an_offset_and_preserves_the_instant(
        self, tz: timezone | None
    ) -> None:
        # Given: An alert using process-local naive time or an explicit timezone.
        alert = _alert().model_copy(update={"ts": datetime(2026, 1, 2, 3, 4, 5, tzinfo=tz)})

        # When: Converting it to the Home Assistant event payload.
        payload = build_event_payload(alert)

        # Then: HA receives an unambiguous instant without changing the original alert.
        assert payload.ts.tzinfo is not None
        assert payload.ts.timestamp() == alert.ts.timestamp()
        assert alert.ts.tzinfo == tz

    @pytest.mark.parametrize("risk", ["low", "medium", "high", "critical"])
    def test_event_type_is_the_risk_level(self, risk: str) -> None:
        # Given: An analyzed alert
        # When: Building the event payload
        payload = build_event_payload(_alert(risk_level=risk))

        # Then: The event type is the risk level and is a declared discovery type
        assert payload.event_type == risk
        assert payload.event_type in EVENT_TYPES

    def test_unanalyzed_alert_does_not_invent_a_risk(self) -> None:
        # Given: An alert from a skipped or failed analysis
        # When: Building the event payload
        payload = build_event_payload(_alert(risk_level=None))

        # Then: It is typed as unanalyzed and carries no risk level
        assert payload.event_type == "unanalyzed"
        assert payload.risk_level is None
        assert payload.event_type in EVENT_TYPES

    def test_event_carries_clip_context(self) -> None:
        # Given: An alert
        alert = _alert()

        # When: Building the event payload
        data = json.loads(build_event_payload(alert).model_dump_json())

        # Then: Automations receive the fields needed to describe and link the clip
        assert data["clip_id"] == "clip_1"
        assert data["camera_name"] == "Front Door"
        assert data["summary"] == "Person at the door"
        assert data["activity_type"] == "person"
        assert data["view_url"] == "http://example/clip_1"
        assert data["notify_reason"] == "risk_level=high"


class TestNotifierWithHomeAssistant:
    @pytest.mark.asyncio
    async def test_concurrent_first_alerts_discover_each_camera_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given: Concurrent first alerts, including two for the same camera.
        notifier, client = _notifier(
            monkeypatch, HomeAssistantConfig(discovery_settle_seconds=0.01)
        )

        # When: All sends pass through the real notifier concurrently.
        await asyncio.gather(
            notifier.send(_alert("Front Door")),
            notifier.send(_alert("Front Door")),
            notifier.send(_alert("Garage")),
        )

        # Then: Each camera is announced once before its events, and every send delivers.
        topics = [t for t, *_ in client.published]
        for camera, count in [("Front Door", 2), ("Garage", 1)]:
            slug = camera_slug(camera)
            discovery = f"homeassistant/event/homesec_{slug}/config"
            event = f"homecam/events/{slug}"
            assert topics.count(discovery) == 1
            assert topics.count(event) == count
            assert topics.index(discovery) < topics.index(event)

    @pytest.mark.asyncio
    async def test_disabled_by_default_publishes_only_the_alert(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given: A notifier without a home_assistant block
        notifier, client = _notifier(monkeypatch, None)

        # When: Sending an alert
        await notifier.send(_alert())

        # Then: Exactly the existing alert message is published, with no last will or discovery
        assert [t for t, *_ in client.published] == ["homecam/alerts/Front Door"]
        assert client.will is None

    @pytest.mark.asyncio
    async def test_connect_sets_last_will_and_announces_online(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given/When: A notifier with Home Assistant enabled connects
        _, client = _notifier(monkeypatch, _ha_config())

        # Then: Offline is the retained last will and online is announced retained
        assert client.will == ("homecam/status", "offline", 1, True)
        assert client.messages("homecam/status") == [("online", 1, True)]

    @pytest.mark.asyncio
    async def test_first_alert_publishes_discovery_before_the_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given: A connected notifier with Home Assistant enabled
        notifier, client = _notifier(monkeypatch, _ha_config())
        slug = camera_slug("Front Door")
        discovery_topic = f"homeassistant/event/homesec_{slug}/config"
        event_topic = f"homecam/events/{slug}"

        # When: Sending the first alert for a camera
        await notifier.send(_alert())

        # Then: Retained discovery precedes a non-retained event, after the unchanged alert
        topics = [t for t, *_ in client.published]
        assert topics.index(discovery_topic) < topics.index(event_topic)
        assert "homecam/alerts/Front Door" in topics
        ((_, discovery_qos, discovery_retain),) = client.messages(discovery_topic)
        assert (discovery_qos, discovery_retain) == (1, True)
        ((event_json, _, event_retain),) = client.messages(event_topic)
        assert event_retain is False
        assert json.loads(event_json)["event_type"] == "high"

    @pytest.mark.asyncio
    async def test_existing_alert_payload_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given: Home Assistant enabled
        notifier, client = _notifier(monkeypatch, _ha_config())
        alert = _alert()

        # When: Sending an alert
        await notifier.send(alert)

        # Then: The original topic still carries the full alert JSON
        ((payload, _, _),) = client.messages("homecam/alerts/Front Door")
        assert payload == alert.model_dump_json()

    @pytest.mark.asyncio
    async def test_discovery_is_published_once_per_camera(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given: A notifier that has already announced one camera
        notifier, client = _notifier(monkeypatch, _ha_config())
        await notifier.send(_alert("Front Door"))

        # When: Sending more alerts for that camera and one for another camera
        await notifier.send(_alert("Front Door"))
        await notifier.send(_alert("Garage"))

        # Then: Each camera has exactly one discovery message but every alert has an event
        discovery = [t for t, *_ in client.published if t.endswith("/config")]
        events = [t for t, *_ in client.published if t.startswith("homecam/events/")]
        assert len(discovery) == 2
        assert len(set(discovery)) == 2
        assert len(events) == 3

    @pytest.mark.asyncio
    async def test_unanalyzed_alert_publishes_unanalyzed_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given: An alert with no analysis
        notifier, client = _notifier(monkeypatch, _ha_config())

        # When: Sending it
        await notifier.send(_alert(risk_level=None))

        # Then: The event is typed unanalyzed
        event_topic = f"homecam/events/{camera_slug('Front Door')}"
        ((event_json, _, _),) = client.messages(event_topic)
        assert json.loads(event_json)["event_type"] == "unanalyzed"

    @pytest.mark.asyncio
    async def test_reconnect_republishes_discovery_and_availability(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given: A notifier that announced a camera before the broker lost its retained data
        notifier, client = _notifier(monkeypatch, _ha_config())
        await notifier.send(_alert())
        slug = camera_slug("Front Door")
        discovery_topic = f"homeassistant/event/homesec_{slug}/config"

        # When: The client reconnects and another alert is sent
        client.reconnect()
        await notifier.send(_alert())

        # Then: Online is announced again and discovery is republished
        assert len(client.messages("homecam/status")) == 2
        assert len(client.messages(discovery_topic)) == 2

    @pytest.mark.asyncio
    async def test_shutdown_announces_offline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given: A connected notifier with Home Assistant enabled
        notifier, client = _notifier(monkeypatch, _ha_config())

        # When: Shutting down cleanly
        await notifier.shutdown()

        # Then: Offline is published retained, since a clean disconnect skips the last will
        assert client.messages("homecam/status")[-1] == ("offline", 1, True)

    @pytest.mark.asyncio
    async def test_shutdown_without_home_assistant_publishes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given: A notifier without Home Assistant
        notifier, client = _notifier(monkeypatch, None)

        # When: Shutting down
        await notifier.shutdown()

        # Then: No availability message is published
        assert client.published == []

    @pytest.mark.asyncio
    async def test_camera_name_with_wildcards_yields_safe_home_assistant_topics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given: A camera name containing MQTT wildcard and separator characters
        notifier, client = _notifier(monkeypatch, _ha_config())
        # The legacy topic must also be publishable; its contract uses the raw camera name.
        notifier.topic_template = "homecam/alerts"

        # When: Sending an alert for it
        await notifier.send(_alert("Back/Yard #1 +"))

        # Then: Discovery and event topics contain no wildcard characters
        ha_topics = [
            t
            for t, *_ in client.published
            if t.startswith("homeassistant/") or t.startswith("homecam/events/")
        ]
        assert len(ha_topics) == 2
        assert all("#" not in t and "+" not in t for t in ha_topics)

    @pytest.mark.asyncio
    async def test_invalid_legacy_topic_fails_before_home_assistant_publication(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given: A wildcard camera name and the unchanged legacy topic template.
        notifier, client = _notifier(monkeypatch, _ha_config())

        # When: The transport rejects that legacy publish topic.
        with pytest.raises(ValueError, match="Invalid MQTT publish topic"):
            await notifier.send(_alert("Back/Yard #1 +"))

        # Then: No HA discovery or event is falsely reported as delivered.
        assert [t for t, *_ in client.published] == ["homecam/status"]

    @pytest.mark.asyncio
    async def test_reconnect_during_discovery_reannounces_before_the_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given: A broker reconnects after acknowledging discovery on the old connection.
        notifier, client = _notifier(monkeypatch, _ha_config())
        discovery = f"homeassistant/event/homesec_{camera_slug('Front Door')}/config"

        def reconnect_once(topic: str) -> None:
            if topic == discovery and len(client.messages(discovery)) == 1:
                client.reconnect()

        client.after_publish = reconnect_once

        # When: Sending the first alert across that reconnect.
        await notifier.send(_alert())

        # Then: New-connection discovery precedes the event and is cached for later alerts.
        assert len(client.messages(discovery)) == 2
        topics = [t for t, *_ in client.published]
        assert topics[-2:] == [discovery, f"homecam/events/{camera_slug('Front Door')}"]
        await notifier.send(_alert())
        assert len(client.messages(discovery)) == 2

    @pytest.mark.asyncio
    async def test_repeated_reconnects_fail_without_publishing_an_undiscovered_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given: Every discovery acknowledgement is followed by another reconnect.
        notifier, client = _notifier(monkeypatch, _ha_config())

        def reconnect_on_discovery(topic: str) -> None:
            if topic.endswith("/config"):
                client.reconnect()

        client.after_publish = reconnect_on_discovery

        # When: Sending cannot obtain discovery on a stable connection.
        with pytest.raises(RuntimeError, match="connection changed repeatedly"):
            await notifier.send(_alert())

        # Then: The attempt is bounded and does not deliver an event without discovery.
        assert len([t for t, *_ in client.published if t.endswith("/config")]) == 3
        assert not any(t.startswith("homecam/events/") for t, *_ in client.published)

    @pytest.mark.asyncio
    async def test_event_acknowledgement_timeout_fails_after_original_alert_delivery(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given: The broker acknowledges discovery and the original alert, but not the event.
        notifier, client = _notifier(monkeypatch, _ha_config())
        result = _FakePublishResult(published=False)
        client.results[f"homecam/events/{camera_slug('Front Door')}"] = result

        # When: Waiting for the HA event acknowledgement times out without paho raising.
        with pytest.raises(TimeoutError, match="acknowledgement timed out"):
            await notifier.send(_alert())

        # Then: The send reports failure, preserving the existing fail-whole-send policy.
        assert len(client.messages("homecam/alerts/Front Door")) == 1
        assert result.waits == [10.0]

    @pytest.mark.asyncio
    async def test_failed_discovery_is_retried_before_any_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given: Discovery is published but never acknowledged on the first attempt.
        notifier, client = _notifier(monkeypatch, _ha_config())
        discovery = f"homeassistant/event/homesec_{camera_slug('Front Door')}/config"
        client.results[discovery] = _FakePublishResult(published=False)

        # When: The caller retries after the first discovery times out.
        with pytest.raises(TimeoutError):
            await notifier.send(_alert())
        assert not any(t.startswith("homecam/events/") for t, *_ in client.published)
        client.results[discovery] = _FakePublishResult()
        await notifier.send(_alert())

        # Then: Failed discovery was not cached; retry also repeats the legacy alert.
        assert len(client.messages(discovery)) == 2
        assert len(client.messages("homecam/alerts/Front Door")) == 2
        assert len(client.messages(f"homecam/events/{camera_slug('Front Door')}")) == 1

    @pytest.mark.asyncio
    async def test_shutdown_logs_unacknowledged_offline_and_still_disconnects(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Given: The offline publication will remain unacknowledged.
        notifier, client = _notifier(monkeypatch, _ha_config())
        result = _FakePublishResult(published=False)
        client.results["homecam/status"] = result

        # When: Shutting down the notifier.
        await notifier.shutdown()

        # Then: A bounded wait reports the failure and transport cleanup still completes.
        assert result.waits == [2.0]
        assert "Failed to publish MQTT offline availability" in caplog.text
        assert client.calls == ["disconnect"]
