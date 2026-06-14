"""Tests for the APNs mobile notifier."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from homesec.models.alert import Alert
from homesec.models.mobile import MobileDevicePushTarget
from homesec.plugins.notifiers.apns_mobile import (
    APNsMobileConfig,
    APNsMobileNotifier,
    build_apns_payload,
)


class _FakeMobileDeviceRepository:
    def __init__(self, targets: list[MobileDevicePushTarget]) -> None:
        self.targets = targets
        self.list_calls: list[tuple[str, str]] = []
        self.recorded_results: list[tuple[str, str | None, datetime | None]] = []

    async def list_enabled_apns_targets(
        self,
        *,
        environment: str,
        bundle_id: str,
    ) -> list[MobileDevicePushTarget]:
        self.list_calls.append((environment, bundle_id))
        return self.targets

    async def record_push_result(
        self,
        device_id: str,
        *,
        error: str | None,
        now: datetime | None = None,
    ) -> None:
        self.recorded_results.append((device_id, error, now))


class _FakeAPNsClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []
        self.is_closed = False

    async def post(
        self,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
    ) -> httpx.Response:
        self.requests.append({"headers": headers, "json": json, "url": url})
        return self.responses.pop(0)

    async def aclose(self) -> None:
        self.is_closed = True


def _private_key_pem() -> str:
    private_key = ec.generate_private_key(ec.SECP256R1())
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")


def _sample_alert(**overrides: Any) -> Alert:
    defaults: dict[str, Any] = {
        "clip_id": "clip_123",
        "camera_name": "front_door",
        "storage_uri": "mock://clip_123",
        "view_url": "http://example.test/clip_123",
        "risk_level": "high",
        "activity_type": "person",
        "notify_reason": "risk_level=high",
        "summary": "Person near the front door.",
        "ts": datetime(2026, 6, 14, 8, 30, tzinfo=timezone.utc),
        "dedupe_key": "clip_123",
        "upload_failed": False,
    }
    defaults.update(overrides)
    return Alert(**defaults)


def _config(repository: _FakeMobileDeviceRepository) -> APNsMobileConfig:
    return APNsMobileConfig(
        key_id_env="TEST_APNS_KEY_ID",
        team_id_env="TEST_APNS_TEAM_ID",
        private_key_env="TEST_APNS_PRIVATE_KEY",
        bundle_id="com.levneiman.homesec",
        environment="sandbox",
        mobile_device_repository=repository,
    )


def test_build_apns_payload_includes_plain_event_route_without_rich_media() -> None:
    # Given: A HomeSec alert for an analyzed clip
    alert = _sample_alert()

    # When: Building the plain APNs payload
    payload = build_apns_payload(alert)

    # Then: The payload includes the notification route and event context
    assert payload["type"] == "event_alert"
    assert payload["event_id"] == "clip_123"
    assert payload["route"] == "/events/clip_123?from=notification"
    assert payload["camera"] == "front_door"
    assert payload["risk_level"] == "high"
    assert payload["activity_type"] == "person"

    # And: Plain push v1 does not request rich notification thumbnail handling
    aps = payload["aps"]
    assert isinstance(aps, dict)
    assert aps["category"] == "HOMESEC_EVENT"
    assert "mutable-content" not in aps
    assert "thumbnail" not in str(payload).lower()


@pytest.mark.asyncio
async def test_apns_notifier_sends_payload_to_registered_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: A configured APNs notifier with one enabled target and mocked HTTP/2 client
    monkeypatch.setenv("TEST_APNS_KEY_ID", "KEY1234567")
    monkeypatch.setenv("TEST_APNS_TEAM_ID", "TEAM123456")
    monkeypatch.setenv("TEST_APNS_PRIVATE_KEY", _private_key_pem())
    repository = _FakeMobileDeviceRepository(
        [
            MobileDevicePushTarget(
                id="dev_1",
                apns_token="apns-token-1",
                apns_environment="sandbox",
                bundle_id="com.levneiman.homesec",
            )
        ]
    )
    fake_client = _FakeAPNsClient([httpx.Response(200)])
    monkeypatch.setattr(
        "homesec.plugins.notifiers.apns_mobile.httpx.AsyncClient",
        lambda **_kwargs: fake_client,
    )
    notifier = APNsMobileNotifier(_config(repository))

    # When: Sending a HomeSec alert
    await notifier.send(_sample_alert())

    # Then: The notifier queries the repository with the configured APNs scope
    assert repository.list_calls == [("sandbox", "com.levneiman.homesec")]

    # And: APNs receives the expected route payload and required provider headers
    assert len(fake_client.requests) == 1
    request = fake_client.requests[0]
    assert request["url"] == "https://api.sandbox.push.apple.com/3/device/apns-token-1"
    assert request["json"]["route"] == "/events/clip_123?from=notification"
    headers = request["headers"]
    assert headers["apns-topic"] == "com.levneiman.homesec"
    assert headers["apns-push-type"] == "alert"
    assert headers["authorization"].startswith("bearer ")

    # And: Successful delivery clears the device push error
    assert len(repository.recorded_results) == 1
    assert repository.recorded_results[0][0] == "dev_1"
    assert repository.recorded_results[0][1] is None

    await notifier.shutdown()
    assert fake_client.is_closed is True


@pytest.mark.asyncio
async def test_apns_notifier_records_rejected_devices_and_raises_when_all_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: APNs rejects the only enabled target
    monkeypatch.setenv("TEST_APNS_KEY_ID", "KEY1234567")
    monkeypatch.setenv("TEST_APNS_TEAM_ID", "TEAM123456")
    monkeypatch.setenv("TEST_APNS_PRIVATE_KEY", _private_key_pem())
    repository = _FakeMobileDeviceRepository(
        [
            MobileDevicePushTarget(
                id="dev_bad",
                apns_token="bad-token",
                apns_environment="sandbox",
                bundle_id="com.levneiman.homesec",
            )
        ]
    )
    fake_client = _FakeAPNsClient([httpx.Response(400, json={"reason": "BadDeviceToken"})])
    monkeypatch.setattr(
        "homesec.plugins.notifiers.apns_mobile.httpx.AsyncClient",
        lambda **_kwargs: fake_client,
    )
    notifier = APNsMobileNotifier(_config(repository))

    # When: Sending the alert
    with pytest.raises(RuntimeError, match="APNs delivery failed"):
        await notifier.send(_sample_alert())

    # Then: The rejection is recorded against the device without logging token material
    assert len(repository.recorded_results) == 1
    device_id, error, recorded_at = repository.recorded_results[0]
    assert device_id == "dev_bad"
    assert error == "HTTP 400: BadDeviceToken"
    assert recorded_at is not None
