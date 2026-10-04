"""Tests for the APNs mobile notifier."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from homesec.models.alert import Alert
from homesec.models.mobile import MobileDevicePushTarget
from homesec.plugins.notifiers.apns_mobile import (
    APNsDeliveryError,
    APNsMobileConfig,
    APNsMobileNotifier,
    build_apns_payload,
)


class _FakeMobileDeviceRepository:
    def __init__(self, targets: list[MobileDevicePushTarget]) -> None:
        self.targets = targets
        self.disabled_devices: list[tuple[str, datetime | None]] = []
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

    async def disable_device(
        self,
        device_id: str,
        *,
        now: datetime | None = None,
    ) -> None:
        self.disabled_devices.append((device_id, now))


class _FakeAPNsClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []
        self.is_closed = False

    async def post(
        self,
        url: str,
        *,
        content: bytes,
        headers: dict[str, str],
    ) -> httpx.Response:
        request = httpx.Request("POST", url, content=content, headers=headers)
        self.requests.append(
            {
                "headers": dict(request.headers),
                "json": json.loads(request.content),
                "content": request.content,
                "url": str(request.url),
            }
        )
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
    assert request["json"]["aps"]["alert"]["body"] == "Person near the front door."
    headers = request["headers"]
    assert headers["apns-topic"] == "com.levneiman.homesec"
    assert headers["apns-push-type"] == "alert"
    assert headers["content-type"] == "application/json"
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
    # Given: APNs rejects the only enabled target with a retryable provider error
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
    fake_client = _FakeAPNsClient([httpx.Response(500, json={"reason": "InternalServerError"})])
    monkeypatch.setattr(
        "homesec.plugins.notifiers.apns_mobile.httpx.AsyncClient",
        lambda **_kwargs: fake_client,
    )
    notifier = APNsMobileNotifier(_config(repository))

    # When: Sending the alert
    with pytest.raises(APNsDeliveryError, match="APNs delivery failed") as exc_info:
        await notifier.send(_sample_alert())

    # Then: The retryable rejection is recorded without disabling the device
    assert exc_info.value.retryable is True
    assert len(repository.recorded_results) == 1
    device_id, error, recorded_at = repository.recorded_results[0]
    assert device_id == "dev_bad"
    assert error == "HTTP 500: InternalServerError"
    assert recorded_at is not None
    assert repository.disabled_devices == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "reason", "retryable"),
    [
        (200, None, False),
        (410, "Unregistered", False),
        (403, "Forbidden", False),
        (413, "PayloadTooLarge", False),
        (503, "ServiceUnavailable", True),
    ],
)
async def test_apns_outcome_survives_bookkeeping_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    status_code: int,
    reason: str | None,
    retryable: bool,
) -> None:
    # Given: APNs is reachable but both result recording and token disabling fail
    class FailingRepository(_FakeMobileDeviceRepository):
        async def record_push_result(
            self, device_id: str, *, error: str | None, now: datetime | None = None
        ) -> None:
            await super().record_push_result(device_id, error=error, now=now)
            raise ConnectionError("synthetic-sensitive-database-detail")

        async def disable_device(self, device_id: str, *, now: datetime | None = None) -> None:
            await super().disable_device(device_id, now=now)
            raise ConnectionError("synthetic-sensitive-database-detail")

    repository = FailingRepository(
        [
            MobileDevicePushTarget(
                id="dev_1",
                apns_token="synthetic-token",
                apns_environment="sandbox",
                bundle_id="com.levneiman.homesec",
            )
        ]
    )
    monkeypatch.setenv("TEST_APNS_KEY_ID", "KEY1234567")
    monkeypatch.setenv("TEST_APNS_TEAM_ID", "TEAM123456")
    monkeypatch.setenv("TEST_APNS_PRIVATE_KEY", _private_key_pem())
    fake_client = _FakeAPNsClient([httpx.Response(status_code, json={"reason": reason})])
    monkeypatch.setattr(
        "homesec.plugins.notifiers.apns_mobile.httpx.AsyncClient", lambda **_kwargs: fake_client
    )
    notifier = APNsMobileNotifier(_config(repository))

    # When: Sending an alert and APNs accepts or rejects it before the DB failure
    if status_code == 200:
        await notifier.send(_sample_alert())
    else:
        with pytest.raises(APNsDeliveryError) as exc_info:
            await notifier.send(_sample_alert())
        assert exc_info.value.retryable is retryable

    # Then: Bookkeeping cannot turn accepted/permanent outcomes into retries or leak error data
    assert len(fake_client.requests) == 1
    assert len(repository.recorded_results) == 1
    assert len(repository.disabled_devices) == (1 if status_code == 410 else 0)
    assert "synthetic-sensitive-database-detail" not in caplog.text
    await notifier.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("private_key", ["malformed-private-key", ""])
async def test_apns_invalid_credentials_degrade_without_preventing_startup(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    private_key: str,
) -> None:
    # Given: Optional APNs credentials are malformed or absent
    monkeypatch.setenv("TEST_APNS_KEY_ID", "KEY1234567")
    monkeypatch.setenv("TEST_APNS_TEAM_ID", "TEAM123456")
    monkeypatch.setenv("TEST_APNS_PRIVATE_KEY", private_key)

    # When: Constructing the notifier during runtime startup
    notifier = APNsMobileNotifier(_config(_FakeMobileDeviceRepository([])))

    # Then: Startup succeeds, health is false, and failed delivery exposes no credential material
    assert await notifier.ping() is False
    with pytest.raises(RuntimeError, match="credentials missing"):
        await notifier.send(_sample_alert())
    assert "TEST_APNS_PRIVATE_KEY" in caplog.text
    if private_key:
        assert private_key not in caplog.text
    await notifier.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "summary",
    ["A" * 6000, "人" * 1500, "🚪" * 1500, '"\n\\' * 1500],
    ids=["ascii", "cjk", "emoji", "json-escaped"],
)
async def test_apns_notifier_bounds_encoded_payload_without_changing_event_route(
    monkeypatch: pytest.MonkeyPatch,
    summary: str,
) -> None:
    # Given: An alert whose summary exceeds the APNs byte limit after JSON encoding
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
    requests: list[httpx.Request] = []

    def receive_push(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(receive_push))
    monkeypatch.setattr(
        "homesec.plugins.notifiers.apns_mobile.httpx.AsyncClient",
        lambda **_kwargs: http_client,
    )
    notifier = APNsMobileNotifier(_config(repository))
    alert = _sample_alert(clip_id="clip 你好/1", summary=summary)

    # When: Sending through the HTTP boundary
    try:
        await notifier.send(alert)
    finally:
        await notifier.shutdown()

    # Then: The exact request bytes fit APNs and preserve complete routing/context
    assert len(requests) == 1
    request = requests[0]
    assert len(request.content) <= 4096
    assert request.headers["content-type"] == "application/json"
    payload = json.loads(request.content.decode("utf-8"))
    assert payload["event_id"] == alert.clip_id
    assert payload["route"] == "/events/clip%20%E4%BD%A0%E5%A5%BD%2F1?from=notification"
    assert payload["camera"] == alert.camera_name
    assert payload["risk_level"] == "high"
    assert payload["activity_type"] == "person"
    body = payload["aps"]["alert"]["body"]
    assert body
    assert summary.startswith(body)
    assert len(body) < len(summary)
    assert repository.recorded_results[0][1] is None


@pytest.mark.asyncio
async def test_apns_notifier_rejects_oversized_event_metadata_without_sending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: Event identifiers and their route alone exceed the APNs payload budget
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
    fake_client = _FakeAPNsClient([])
    monkeypatch.setattr(
        "homesec.plugins.notifiers.apns_mobile.httpx.AsyncClient",
        lambda **_kwargs: fake_client,
    )
    notifier = APNsMobileNotifier(_config(repository))

    # When: Trying to send an alert that cannot fit even with an empty display body
    with pytest.raises(APNsDeliveryError, match="APNs payload exceeds 4096-byte limit") as exc_info:
        await notifier.send(_sample_alert(clip_id="人" * 500))

    # Then: The invalid payload fails permanently without corrupting the device registration
    assert exc_info.value.retryable is False
    assert fake_client.requests == []
    assert repository.recorded_results == []
    assert repository.disabled_devices == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status_codes", [(413,), (413, 503)], ids=["alone", "mixed-fanout"])
async def test_apns_notifier_does_not_retry_payload_too_large_or_disable_valid_devices(
    monkeypatch: pytest.MonkeyPatch,
    status_codes: tuple[int, ...],
) -> None:
    # Given: APNs rejects the payload, optionally alongside a transient target failure
    monkeypatch.setenv("TEST_APNS_KEY_ID", "KEY1234567")
    monkeypatch.setenv("TEST_APNS_TEAM_ID", "TEAM123456")
    monkeypatch.setenv("TEST_APNS_PRIVATE_KEY", _private_key_pem())
    repository = _FakeMobileDeviceRepository(
        [
            MobileDevicePushTarget(
                id=f"dev_{index}",
                apns_token=f"apns-token-{index}",
                apns_environment="sandbox",
                bundle_id="com.levneiman.homesec",
            )
            for index, _status in enumerate(status_codes)
        ]
    )
    fake_client = _FakeAPNsClient(
        [
            httpx.Response(
                status,
                json={"reason": "PayloadTooLarge" if status == 413 else "ServiceUnavailable"},
            )
            for status in status_codes
        ]
    )
    monkeypatch.setattr(
        "homesec.plugins.notifiers.apns_mobile.httpx.AsyncClient",
        lambda **_kwargs: fake_client,
    )
    notifier = APNsMobileNotifier(_config(repository))

    # When: Sending an otherwise valid alert to the registered targets
    with pytest.raises(APNsDeliveryError, match="APNs delivery failed") as exc_info:
        await notifier.send(_sample_alert())

    # Then: Retrying the same invalid payload is suppressed without disabling valid tokens
    assert exc_info.value.retryable is False
    assert len(fake_client.requests) == len(status_codes)
    assert repository.recorded_results[0][1] == "HTTP 413: PayloadTooLarge"
    assert len(repository.recorded_results) == len(status_codes)
    assert repository.disabled_devices == []


@pytest.mark.asyncio
async def test_apns_notifier_disables_permanent_failures_without_retrying_successes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: APNs accepts one registered target and rejects another target permanently
    monkeypatch.setenv("TEST_APNS_KEY_ID", "KEY1234567")
    monkeypatch.setenv("TEST_APNS_TEAM_ID", "TEAM123456")
    monkeypatch.setenv("TEST_APNS_PRIVATE_KEY", _private_key_pem())
    repository = _FakeMobileDeviceRepository(
        [
            MobileDevicePushTarget(
                id="dev_ok",
                apns_token="good-token",
                apns_environment="sandbox",
                bundle_id="com.levneiman.homesec",
            ),
            MobileDevicePushTarget(
                id="dev_bad",
                apns_token="bad-token",
                apns_environment="sandbox",
                bundle_id="com.levneiman.homesec",
            ),
        ]
    )
    fake_client = _FakeAPNsClient(
        [
            httpx.Response(200),
            httpx.Response(410, json={"reason": "Unregistered"}),
        ]
    )
    monkeypatch.setattr(
        "homesec.plugins.notifiers.apns_mobile.httpx.AsyncClient",
        lambda **_kwargs: fake_client,
    )
    notifier = APNsMobileNotifier(_config(repository))

    # When: Sending the alert
    with pytest.raises(APNsDeliveryError, match="APNs delivery failed") as exc_info:
        await notifier.send(_sample_alert())

    # Then: Successful and failed device outcomes are both recorded without a whole-fanout retry
    assert exc_info.value.retryable is False
    assert [(device_id, error) for device_id, error, _ in repository.recorded_results] == [
        ("dev_ok", None),
        ("dev_bad", "HTTP 410: Unregistered"),
    ]
    disabled_device_id, disabled_at = repository.disabled_devices[0]
    assert disabled_device_id == "dev_bad"
    assert disabled_at == repository.recorded_results[1][2]


@pytest.mark.asyncio
async def test_apns_notifier_raises_on_partial_retryable_delivery_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: APNs accepts one target but has a retryable provider failure for another
    monkeypatch.setenv("TEST_APNS_KEY_ID", "KEY1234567")
    monkeypatch.setenv("TEST_APNS_TEAM_ID", "TEAM123456")
    monkeypatch.setenv("TEST_APNS_PRIVATE_KEY", _private_key_pem())
    repository = _FakeMobileDeviceRepository(
        [
            MobileDevicePushTarget(
                id="dev_ok",
                apns_token="good-token",
                apns_environment="sandbox",
                bundle_id="com.levneiman.homesec",
            ),
            MobileDevicePushTarget(
                id="dev_retry",
                apns_token="retry-token",
                apns_environment="sandbox",
                bundle_id="com.levneiman.homesec",
            ),
        ]
    )
    fake_client = _FakeAPNsClient(
        [
            httpx.Response(200),
            httpx.Response(503, json={"reason": "ServiceUnavailable"}),
        ]
    )
    monkeypatch.setattr(
        "homesec.plugins.notifiers.apns_mobile.httpx.AsyncClient",
        lambda **_kwargs: fake_client,
    )
    notifier = APNsMobileNotifier(_config(repository))

    # When: Sending the alert
    with pytest.raises(
        APNsDeliveryError, match="APNs delivery failed for 1 of 2 device"
    ) as exc_info:
        await notifier.send(_sample_alert())

    # Then: The partial retryable failure is recorded without retrying the already delivered target
    assert exc_info.value.retryable is False
    assert [(device_id, error) for device_id, error, _ in repository.recorded_results] == [
        ("dev_ok", None),
        ("dev_retry", "HTTP 503: ServiceUnavailable"),
    ]
    assert repository.disabled_devices == []
