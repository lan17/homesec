"""Home Assistant notifier plugin (Events API)."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

aiohttp: Any

try:
    import aiohttp as _aiohttp
except Exception:
    aiohttp = None
else:
    aiohttp = _aiohttp

from homesec.interfaces import Notifier
from homesec.models.alert import Alert
from homesec.models.config import HomeAssistantNotifierConfig
from homesec.plugins.registry import PluginType, plugin

logger = logging.getLogger(__name__)

_EVENT_PREFIX = "homesec"
_SUPERVISOR_URL = "http://supervisor/core"
_OBJECT_ORDER = ("person", "vehicle", "animal", "package", "object", "unknown")
_PERSON_CLASSES = {"person", "human"}
_VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "motorbike",
    "bicycle",
    "bike",
    "scooter",
    "van",
    "vehicle",
    "train",
    "boat",
    "ship",
    "airplane",
}
_ANIMAL_CLASSES = {
    "dog",
    "cat",
    "bird",
    "horse",
    "sheep",
    "cow",
    "bear",
    "zebra",
    "giraffe",
    "animal",
}
_PACKAGE_CLASSES = {"package", "parcel", "box", "bag"}


def _ensure_aiohttp_dependencies() -> None:
    """Fail fast with a clear error if aiohttp is missing."""
    if aiohttp is None:
        raise RuntimeError(
            "Missing dependency for Home Assistant notifier. Install with: uv pip install aiohttp"
        )


@plugin(plugin_type=PluginType.NOTIFIER, name="home_assistant")
class HomeAssistantNotifier(Notifier):
    """Push HomeSec alerts to Home Assistant via the Events API."""

    config_cls = HomeAssistantNotifierConfig

    @classmethod
    def create(cls, config: HomeAssistantNotifierConfig) -> Notifier:
        return cls(config)

    def __init__(self, config: HomeAssistantNotifierConfig) -> None:
        _ensure_aiohttp_dependencies()
        self._session: aiohttp.ClientSession | None = None
        self._shutdown_called = False
        self._timeout_s = 10.0

        supervisor_token = os.getenv("SUPERVISOR_TOKEN")
        self._supervisor_mode = bool(supervisor_token)
        self._base_url: str | None = None
        self._token: str | None = None

        if self._supervisor_mode:
            self._base_url = _SUPERVISOR_URL
            self._token = supervisor_token
        else:
            if not config.url_env or not config.token_env:
                raise ValueError(
                    "home_assistant notifier requires url_env and token_env when "
                    "SUPERVISOR_TOKEN is not set"
                )

            self._base_url = os.getenv(config.url_env)
            self._token = os.getenv(config.token_env)

            if not self._base_url:
                logger.warning("Home Assistant URL not found in env: %s", config.url_env)
            if not self._token:
                logger.warning("Home Assistant token not found in env: %s", config.token_env)

        if self._base_url:
            self._base_url = self._base_url.rstrip("/")

    async def send(self, alert: Alert) -> None:
        """Send alert notification to Home Assistant."""
        if self._shutdown_called:
            raise RuntimeError("Notifier has been shut down")

        url, headers = self._get_url_and_headers("alert")
        payload = self._build_event_data(alert)
        session = await self._get_session()

        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status >= 400:
                    details = await response.text()
                    raise RuntimeError(
                        f"Home Assistant event send failed: HTTP {response.status}: {details}"
                    )
                await response.read()
        except asyncio.TimeoutError as exc:
            raise asyncio.TimeoutError("Home Assistant event send timed out") from exc

        logger.info("Home Assistant event sent: clip_id=%s", alert.clip_id)

    async def ping(self) -> bool:
        """Health check - verify Home Assistant is reachable."""
        if self._shutdown_called:
            return False
        if not self._base_url or not self._token:
            return False

        session = await self._get_session()
        url = f"{self._base_url}/api/"
        headers = {"Authorization": f"Bearer {self._token}"}

        try:
            async with session.get(url, headers=headers) as response:
                if response.status >= 400:
                    return False
                await response.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning("Home Assistant ping failed: %s", exc)
            return False

        return True

    async def shutdown(self, timeout: float | None = None) -> None:
        """Cleanup resources - close HTTP session."""
        _ = timeout
        if self._shutdown_called:
            return
        self._shutdown_called = True

        if self._session and not self._session.closed:
            await self._session.close()

    def _get_url_and_headers(self, event_type: str) -> tuple[str, dict[str, str]]:
        if not self._base_url:
            raise RuntimeError("Home Assistant URL is missing")
        if not self._token:
            raise RuntimeError("Home Assistant token is missing")

        event_name = f"{_EVENT_PREFIX}_{event_type}"
        url = f"{self._base_url}/api/events/{event_name}"
        headers = {"Authorization": f"Bearer {self._token}"}
        return url, headers

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            if aiohttp is None:
                raise RuntimeError("aiohttp dependency is required for Home Assistant notifier")
            timeout = aiohttp.ClientTimeout(total=self._timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    def _build_event_data(self, alert: Alert) -> dict[str, object]:
        detected_objects = self._normalize_detected_objects(alert.detected_classes)
        data: dict[str, object] = {
            "camera": alert.camera_name,
            "clip_id": alert.clip_id,
            "activity_type": alert.activity_type or "unknown",
            "risk_level": str(alert.risk_level) if alert.risk_level is not None else "unknown",
            "summary": alert.summary or "",
            "view_url": alert.view_url,
            "storage_uri": alert.storage_uri,
            "timestamp": alert.ts.isoformat(),
            "detected_objects": detected_objects,
        }

        return data

    def _normalize_detected_objects(self, detected_classes: list[str] | None) -> list[str]:
        if not detected_classes:
            return []

        found: set[str] = set()
        for class_name in detected_classes:
            key = class_name.strip().lower()
            if key in _PERSON_CLASSES:
                found.add("person")
            elif key in _VEHICLE_CLASSES:
                found.add("vehicle")
            elif key in _ANIMAL_CLASSES:
                found.add("animal")
            elif key in _PACKAGE_CLASSES:
                found.add("package")
            elif key == "unknown":
                found.add("unknown")
            else:
                found.add("object")

        return [category for category in _OBJECT_ORDER if category in found]
