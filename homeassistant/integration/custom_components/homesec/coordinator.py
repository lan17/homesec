"""Data update coordinator for HomeSec."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import aiohttp
import async_timeout
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    CONF_API_KEY,
    CONF_CAMERAS,
    CONF_HOST,
    CONF_PORT,
    CONF_VERIFY_SSL,
    DEFAULT_MOTION_RESET_SECONDS,
    DEFAULT_SCAN_INTERVAL,
    EVENT_ALERT,
)

_LOGGER = logging.getLogger(__name__)


class HomesecAuthError(HomeAssistantError):
    """Error raised when authentication fails."""


@dataclass
class HomesecApiError(Exception):
    """Error raised for HomeSec API errors."""

    status: int | None
    message: str


class HomesecCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator to manage HomeSec data updates."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry

        self._host = entry.data[CONF_HOST]
        self._port = entry.data[CONF_PORT]
        self._api_key = entry.data.get(CONF_API_KEY)
        self._verify_ssl = entry.data.get(CONF_VERIFY_SSL, True)
        self._camera_names = entry.data.get(CONF_CAMERAS, [])

        options = entry.options
        scan_interval = options.get("scan_interval", DEFAULT_SCAN_INTERVAL)
        if isinstance(scan_interval, (int, float)):
            update_interval = timedelta(seconds=int(scan_interval))
        else:
            update_interval = DEFAULT_SCAN_INTERVAL

        self._motion_reset_seconds = int(
            options.get("motion_reset_seconds", DEFAULT_MOTION_RESET_SECONDS)
        )

        self._connected = False
        self._last_poll_data: dict[str, Any] | None = None
        self._motion_active: dict[str, bool] = dict.fromkeys(self._camera_names, False)
        self._latest_alerts: dict[str, dict[str, Any]] = {}
        self._motion_resets: dict[str, Callable[[], None]] = {}
        self._event_unsub: Callable[[], None] | None = None

        self._session = async_get_clientsession(hass)
        self._timeout = 10.0

        super().__init__(
            hass,
            _LOGGER,
            name="homesec",
            update_interval=update_interval,
        )

    @property
    def base_url(self) -> str:
        host = self._host.rstrip("/")
        if host.startswith("http://") or host.startswith("https://"):
            base = host
        else:
            base = f"http://{host}"

        if self._port and ":" not in base.split("//", 1)[-1]:
            return f"{base}:{self._port}"
        return base

    @property
    def camera_names(self) -> list[str]:
        return list(self._camera_names)

    async def _async_update_data(self) -> dict[str, Any]:
        try:
            health = await self._request_json("GET", "/api/v1/health", auth=False)
            cameras = await self._request_json("GET", "/api/v1/cameras")
            stats = await self._request_json("GET", "/api/v1/stats")
        except HomesecAuthError as exc:
            self._connected = False
            raise UpdateFailed("Authentication failed") from exc
        except HomesecApiError as exc:
            self._connected = False
            if exc.status == 503:
                raise UpdateFailed("HomeSec unavailable") from exc
            raise UpdateFailed(f"HomeSec API error: {exc.message}") from exc
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self._connected = False
            raise UpdateFailed("Error communicating with HomeSec") from exc

        self._connected = True
        self._last_poll_data = {
            "health": health,
            "cameras": cameras,
            "stats": stats,
        }
        return self._build_data()

    async def async_subscribe_events(self) -> None:
        """Subscribe to HomeSec events fired via HA Events API."""
        if self._event_unsub is not None:
            return

        self._event_unsub = self.hass.bus.async_listen(EVENT_ALERT, self._handle_alert_event)

    async def async_unsubscribe_events(self) -> None:
        """Unsubscribe from HomeSec events."""
        if self._event_unsub is not None:
            self._event_unsub()
            self._event_unsub = None

        for cancel in self._motion_resets.values():
            cancel()
        self._motion_resets.clear()

    async def async_add_camera(
        self, name: str, source_backend: str, source_config: dict[str, Any]
    ) -> dict:
        payload = {
            "name": name,
            "enabled": True,
            "source_backend": source_backend,
            "source_config": source_config,
        }
        return await self._request_json("POST", "/api/v1/cameras", payload=payload)

    async def async_update_camera(
        self, camera_name: str, source_config: dict[str, Any] | None = None
    ) -> dict:
        payload: dict[str, Any] = {}
        if source_config is not None:
            payload["source_config"] = source_config
        return await self._request_json("PUT", f"/api/v1/cameras/{camera_name}", payload=payload)

    async def async_delete_camera(self, camera_name: str) -> None:
        await self._request_json("DELETE", f"/api/v1/cameras/{camera_name}")

    async def async_set_camera_enabled(self, camera_name: str, enabled: bool) -> dict:
        payload = {"enabled": enabled}
        return await self._request_json("PUT", f"/api/v1/cameras/{camera_name}", payload=payload)

    async def _handle_alert_event(self, event: Any) -> None:
        data = event.data or {}
        camera = data.get("camera")
        if not camera:
            return

        self._latest_alerts[camera] = dict(data)
        self._motion_active[camera] = True

        cancel = self._motion_resets.pop(camera, None)
        if cancel is not None:
            cancel()

        self._motion_resets[camera] = async_call_later(
            self.hass, self._motion_reset_seconds, self._clear_motion(camera)
        )

        self.async_set_updated_data(self._build_data())

    def _clear_motion(self, camera: str) -> Callable[[Any], None]:
        def _clear(_: Any) -> None:
            self._motion_active[camera] = False
            self._motion_resets.pop(camera, None)
            self.async_set_updated_data(self._build_data())

        return _clear

    def _build_data(self) -> dict[str, Any]:
        base = self._last_poll_data or {"health": {}, "cameras": [], "stats": {}}
        return {
            "health": base.get("health", {}),
            "cameras": base.get("cameras", []),
            "stats": base.get("stats", {}),
            "connected": self._connected,
            "motion_active": dict(self._motion_active),
            "latest_alerts": dict(self._latest_alerts),
        }

    async def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        auth: bool = True,
    ) -> dict[str, Any] | list[Any]:
        url = f"{self.base_url}{path}"
        headers = {}
        if auth and self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with (
                async_timeout.timeout(self._timeout),
                self._session.request(
                    method,
                    url,
                    json=payload,
                    headers=headers,
                    ssl=self._verify_ssl,
                ) as response,
            ):
                if response.status in (401, 403):
                    raise HomesecAuthError("Unauthorized")
                if response.status >= 400:
                    message = await response.text()
                    raise HomesecApiError(response.status, message)
                return await response.json()
        except asyncio.TimeoutError as exc:
            raise HomesecApiError(None, "Request timed out") from exc
