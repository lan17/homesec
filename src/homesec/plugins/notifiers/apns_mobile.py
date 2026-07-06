"""APNs notifier plugin for registered HomeSec iOS devices."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Protocol, cast
from urllib.parse import quote

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils
from pydantic import BaseModel, Field, field_validator

from homesec.interfaces import Notifier
from homesec.models.alert import Alert
from homesec.models.mobile import APNSEnvironment, MobileDevicePushTarget
from homesec.plugins.registry import PluginType, plugin

logger = logging.getLogger(__name__)

_APNS_CATEGORY = "HOMESEC_EVENT"
_APNS_PUSH_TYPE = "alert"
_PROVIDER_TOKEN_REFRESH_S = 50 * 60
_PERMANENT_TOKEN_REJECTION_REASONS = frozenset(
    {
        "BadDeviceToken",
        "DeviceTokenNotForTopic",
        "Unregistered",
    }
)


class _MobileDevicePushRepository(Protocol):
    async def list_enabled_apns_targets(
        self,
        *,
        environment: APNSEnvironment,
        bundle_id: str,
    ) -> list[MobileDevicePushTarget]:
        """Return enabled APNs targets for one app bundle/environment."""
        ...

    async def record_push_result(
        self,
        device_id: str,
        *,
        error: str | None,
        now: datetime | None = None,
    ) -> object | None:
        """Record the latest APNs send outcome for a device."""
        ...

    async def disable_device(
        self,
        device_id: str,
        *,
        now: datetime | None = None,
    ) -> object | None:
        """Disable a permanently invalid APNs target."""
        ...


class APNsMobileConfig(BaseModel):
    """APNs notifier configuration using Apple token-based provider auth."""

    model_config = {"extra": "forbid"}

    key_id_env: str = "HOMESEC_APNS_KEY_ID"
    team_id_env: str = "HOMESEC_APNS_TEAM_ID"
    private_key_env: str = "HOMESEC_APNS_PRIVATE_KEY"
    bundle_id: str
    environment: APNSEnvironment = "sandbox"
    request_timeout_s: float = Field(default=10.0, gt=0)
    apns_base_url: str | None = None
    mobile_device_repository: Any | None = Field(default=None, exclude=True, repr=False)

    @field_validator("key_id_env", "team_id_env", "private_key_env", "bundle_id")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator("apns_base_url")
    @classmethod
    def _strip_optional_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().rstrip("/")
        return normalized or None


class _APNsProviderTokenSigner:
    """Creates and caches APNs ES256 provider tokens."""

    def __init__(self, *, key_id: str, team_id: str, private_key_pem: str) -> None:
        self._key_id = key_id
        self._team_id = team_id
        self._private_key = _load_signing_key(private_key_pem)
        self._cached_token: str | None = None
        self._cached_issued_at = 0

    def token(self) -> str:
        issued_at = int(time.time())
        if (
            self._cached_token is not None
            and issued_at - self._cached_issued_at < _PROVIDER_TOKEN_REFRESH_S
        ):
            return self._cached_token

        header = {"alg": "ES256", "kid": self._key_id}
        claims = {"iss": self._team_id, "iat": issued_at}
        signing_input = f"{_base64url_json(header)}.{_base64url_json(claims)}".encode("ascii")
        der_signature = self._private_key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
        r_value, s_value = utils.decode_dss_signature(der_signature)
        raw_signature = r_value.to_bytes(32, "big") + s_value.to_bytes(32, "big")
        token = f"{signing_input.decode('ascii')}.{_base64url(raw_signature)}"
        self._cached_token = token
        self._cached_issued_at = issued_at
        return token


@plugin(plugin_type=PluginType.NOTIFIER, name="apns_mobile")
class APNsMobileNotifier(Notifier):
    """Send plain APNs alert notifications to registered HomeSec iOS devices."""

    config_cls = APNsMobileConfig

    @classmethod
    def create(cls, config: APNsMobileConfig) -> Notifier:
        return cls(config)

    def __init__(self, config: APNsMobileConfig) -> None:
        self._bundle_id = config.bundle_id
        self._environment = config.environment
        self._timeout_s = float(config.request_timeout_s)
        self._base_url = config.apns_base_url or _default_apns_base_url(config.environment)
        self._repository = _require_mobile_repository(config.mobile_device_repository)
        self._signer = _build_provider_token_signer(config)
        self._client: httpx.AsyncClient | None = None
        self._shutdown_called = False

    async def send(self, alert: Alert) -> None:
        """Send one alert to all currently enabled iOS APNs targets."""
        if self._shutdown_called:
            raise RuntimeError("Notifier has been shut down")
        if self._signer is None:
            raise RuntimeError("APNs provider credentials missing from environment")

        targets = await self._repository.list_enabled_apns_targets(
            environment=self._environment,
            bundle_id=self._bundle_id,
        )
        if not targets:
            logger.info(
                "APNs mobile notifier found no enabled targets",
                extra={
                    "event_type": "apns_mobile_no_targets",
                    "apns_environment": self._environment,
                    "bundle_id": self._bundle_id,
                },
            )
            return

        payload = build_apns_payload(alert)
        provider_token = self._signer.token()
        sent_at = datetime.now(timezone.utc)
        results = await asyncio.gather(
            *(
                self._send_to_target(
                    target,
                    payload=payload,
                    provider_token=provider_token,
                    sent_at=sent_at,
                )
                for target in targets
            ),
            return_exceptions=True,
        )

        successes = 0
        permanent_failures: list[str] = []
        retryable_failures: list[str] = []
        for target, result in zip(targets, results, strict=True):
            match result:
                case _DeliveryResult() as delivery:
                    if delivery.delivered:
                        successes += 1
                    elif delivery.retryable:
                        retryable_failures.append(target.id)
                    else:
                        permanent_failures.append(target.id)
                case BaseException() as exc:
                    retryable_failures.append(target.id)
                    logger.error(
                        "APNs mobile send failed while recording device result: device_id=%s "
                        "error=%s",
                        target.id,
                        exc,
                        exc_info=exc,
                    )

        failure_count = len(permanent_failures) + len(retryable_failures)
        if failure_count:
            logger.warning(
                "APNs mobile notifier had failed target deliveries: failed=%d succeeded=%d",
                failure_count,
                successes,
                extra={
                    "event_type": "apns_mobile_delivery_partial_failure",
                    "failed_count": failure_count,
                    "permanent_failed_count": len(permanent_failures),
                    "retryable_failed_count": len(retryable_failures),
                    "succeeded_count": successes,
                },
            )
        if retryable_failures or permanent_failures:
            raise APNsDeliveryError(
                f"APNs delivery failed for {failure_count} of {len(targets)} device(s)",
                retryable=successes == 0 and bool(retryable_failures),
            )

    async def ping(self) -> bool:
        """Health check for local APNs notifier configuration."""
        return not self._shutdown_called and self._signer is not None

    async def shutdown(self, timeout: float | None = None) -> None:
        """Close the HTTP client used for APNs delivery."""
        _ = timeout
        if self._shutdown_called:
            return
        self._shutdown_called = True
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def _send_to_target(
        self,
        target: MobileDevicePushTarget,
        *,
        payload: dict[str, object],
        provider_token: str,
        sent_at: datetime,
    ) -> _DeliveryResult:
        headers = {
            "authorization": f"bearer {provider_token}",
            "apns-topic": self._bundle_id,
            "apns-push-type": _APNS_PUSH_TYPE,
            "apns-priority": "10",
        }
        try:
            response = await (await self._get_client()).post(
                _target_url(self._base_url, target.apns_token),
                json=payload,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            error = type(exc).__name__
            await self._repository.record_push_result(target.id, error=error, now=sent_at)
            logger.warning(
                "APNs mobile send transport failed: device_id=%s error=%s",
                target.id,
                error,
            )
            return _DeliveryResult(delivered=False, retryable=True)

        if 200 <= response.status_code < 300:
            await self._repository.record_push_result(target.id, error=None, now=sent_at)
            return _DeliveryResult(delivered=True, retryable=False)

        reason = _apns_response_reason(response)
        error = f"HTTP {response.status_code}: {reason}"
        await self._repository.record_push_result(target.id, error=error, now=sent_at)
        retryable = not _is_permanent_token_rejection(response.status_code, reason)
        if not retryable:
            await self._repository.disable_device(target.id, now=sent_at)
        logger.warning(
            "APNs mobile send rejected: device_id=%s status=%d reason=%s",
            target.id,
            response.status_code,
            reason,
        )
        return _DeliveryResult(delivered=False, retryable=retryable)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                http2=True,
                timeout=httpx.Timeout(self._timeout_s),
            )
        return self._client


def build_apns_payload(alert: Alert) -> dict[str, object]:
    """Build the plain APNs payload for a HomeSec alert."""
    risk_level = str(alert.risk_level) if alert.risk_level is not None else "unknown"
    activity_type = _notification_value(alert.activity_type, fallback="activity")
    title = f"{alert.camera_name}: {activity_type} detected"
    body = _notification_body(alert, risk_level=risk_level)
    route = f"/events/{quote(alert.clip_id, safe='')}?from=notification"

    return {
        "aps": {
            "alert": {
                "title": title,
                "body": body,
            },
            "sound": "default",
            "category": _APNS_CATEGORY,
        },
        "type": "event_alert",
        "event_id": alert.clip_id,
        "camera": alert.camera_name,
        "risk_level": risk_level,
        "activity_type": activity_type,
        "route": route,
    }


def _notification_body(alert: Alert, *, risk_level: str) -> str:
    if alert.summary:
        return alert.summary.strip()
    event_time = alert.ts.strftime("%I:%M %p").lstrip("0")
    if risk_level != "unknown":
        return f"{risk_level.capitalize()}-risk event at {event_time}."
    return f"HomeSec event at {event_time}."


def _notification_value(value: str | None, *, fallback: str) -> str:
    normalized = value.strip() if value is not None else ""
    return normalized or fallback


def _target_url(base_url: str, apns_token: str) -> str:
    return f"{base_url}/3/device/{quote(apns_token, safe='')}"


def _default_apns_base_url(environment: APNSEnvironment) -> str:
    if environment == "sandbox":
        return "https://api.sandbox.push.apple.com"
    return "https://api.push.apple.com"


class _DeliveryResult(BaseModel):
    delivered: bool
    retryable: bool


class APNsDeliveryError(RuntimeError):
    """APNs fanout failed with retry guidance for the pipeline."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def _is_permanent_token_rejection(status_code: int, reason: str) -> bool:
    return status_code == 410 or reason in _PERMANENT_TOKEN_REJECTION_REASONS


def _require_mobile_repository(value: Any | None) -> _MobileDevicePushRepository:
    if value is None:
        raise RuntimeError("APNs mobile notifier requires mobile device repository context")
    return cast(_MobileDevicePushRepository, value)


def _build_provider_token_signer(config: APNsMobileConfig) -> _APNsProviderTokenSigner | None:
    key_id = _resolve_env(config.key_id_env)
    team_id = _resolve_env(config.team_id_env)
    private_key = _resolve_private_key_env(config.private_key_env)
    if not key_id:
        logger.warning("APNs key id not found in env: %s", config.key_id_env)
    if not team_id:
        logger.warning("APNs team id not found in env: %s", config.team_id_env)
    if not private_key:
        logger.warning("APNs private key not found in env: %s", config.private_key_env)
    if not (key_id and team_id and private_key):
        return None
    return _APNsProviderTokenSigner(
        key_id=key_id,
        team_id=team_id,
        private_key_pem=private_key,
    )


def _resolve_env(env_name: str) -> str | None:
    value = os.getenv(env_name)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _resolve_private_key_env(env_name: str) -> str | None:
    value = _resolve_env(env_name)
    if value is None:
        return None
    return value.replace("\\n", "\n")


def _load_signing_key(private_key_pem: str) -> ec.EllipticCurvePrivateKey:
    key = serialization.load_pem_private_key(private_key_pem.encode("utf-8"), password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise RuntimeError("APNs private key must be an EC private key")
    if not isinstance(key.curve, ec.SECP256R1):
        raise RuntimeError("APNs private key must use the P-256 curve")
    return key


def _base64url_json(payload: Mapping[str, object]) -> str:
    data = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _base64url(data)


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _apns_response_reason(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return "unknown"
    if isinstance(payload, dict):
        reason = payload.get("reason")
        if isinstance(reason, str) and reason.strip():
            return reason.strip()[:200]
    return "unknown"
