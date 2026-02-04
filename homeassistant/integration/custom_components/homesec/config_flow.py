"""Config flow for HomeSec."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from urllib.parse import urlsplit

import aiohttp
import async_timeout
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    ADDON_SLUG,
    CONF_ADDON,
    CONF_API_KEY,
    CONF_CAMERAS,
    CONF_HOST,
    CONF_PORT,
    CONF_VERIFY_SSL,
    DEFAULT_MOTION_RESET_SECONDS,
    DEFAULT_PORT,
    DEFAULT_VERIFY_SSL,
)


class CannotConnect(Exception):
    """Error to indicate we cannot connect."""


class InvalidAuth(Exception):
    """Error to indicate there is invalid auth."""


class HomesecConfigFlow(config_entries.ConfigFlow, domain="homesec"):
    """Handle a config flow for HomeSec."""

    VERSION = 1

    def __init__(self) -> None:
        self._config_data: dict[str, Any] = {}
        self._cameras: list[str] = []
        self._title: str = "HomeSec"

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Initial step - check for add-on first, then manual."""
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        addon_running, hostname = await detect_addon(self.hass)
        if addon_running and hostname:
            self._config_data = {
                CONF_ADDON: True,
                CONF_HOST: hostname,
                CONF_PORT: DEFAULT_PORT,
                CONF_VERIFY_SSL: DEFAULT_VERIFY_SSL,
            }
            return await self.async_step_addon()

        return await self.async_step_manual()

    async def async_step_addon(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle add-on auto-discovery confirmation."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                info = await validate_connection(
                    self.hass,
                    self._config_data[CONF_HOST],
                    self._config_data[CONF_PORT],
                    None,
                    self._config_data[CONF_VERIFY_SSL],
                )
                self._title = info.get("title", "HomeSec")
                self._cameras = info.get("cameras", [])
                return await self.async_step_cameras()
            except CannotConnect:
                errors["base"] = "cannot_connect"
            except InvalidAuth:
                errors["base"] = "invalid_auth"

        return self.async_show_form(
            step_id="addon",
            data_schema=vol.Schema({}),
            errors=errors,
        )

    async def async_step_manual(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle manual setup for standalone HomeSec."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                info = await validate_connection(
                    self.hass,
                    user_input[CONF_HOST],
                    user_input[CONF_PORT],
                    user_input.get(CONF_API_KEY),
                    user_input.get(CONF_VERIFY_SSL, DEFAULT_VERIFY_SSL),
                )
                self._title = info.get("title", "HomeSec")
                self._cameras = info.get("cameras", [])
                self._config_data = {
                    CONF_ADDON: False,
                    CONF_HOST: user_input[CONF_HOST],
                    CONF_PORT: user_input[CONF_PORT],
                    CONF_API_KEY: user_input.get(CONF_API_KEY),
                    CONF_VERIFY_SSL: user_input.get(CONF_VERIFY_SSL, DEFAULT_VERIFY_SSL),
                }
                return await self.async_step_cameras()
            except CannotConnect:
                errors["base"] = "cannot_connect"
            except InvalidAuth:
                errors["base"] = "invalid_auth"

        schema = vol.Schema(
            {
                vol.Required(CONF_HOST): cv.string,
                vol.Optional(CONF_PORT, default=DEFAULT_PORT): vol.Coerce(int),
                vol.Optional(CONF_API_KEY): cv.string,
                vol.Optional(CONF_VERIFY_SSL, default=DEFAULT_VERIFY_SSL): cv.boolean,
            }
        )
        return self.async_show_form(step_id="manual", data_schema=schema, errors=errors)

    async def async_step_cameras(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle camera selection step."""
        if user_input is not None:
            selected = list(user_input.get(CONF_CAMERAS, []))
            self._config_data[CONF_CAMERAS] = selected

            await self.async_set_unique_id(
                f"homesec_{self._config_data[CONF_HOST]}_{self._config_data[CONF_PORT]}"
            )
            self._abort_if_unique_id_configured()

            return self.async_create_entry(title=self._title, data=self._config_data)

        if not self._cameras:
            self._config_data[CONF_CAMERAS] = []
            await self.async_set_unique_id(
                f"homesec_{self._config_data[CONF_HOST]}_{self._config_data[CONF_PORT]}"
            )
            self._abort_if_unique_id_configured()
            return self.async_create_entry(title=self._title, data=self._config_data)

        options = {name: name for name in self._cameras}
        schema = vol.Schema(
            {vol.Optional(CONF_CAMERAS, default=self._cameras): cv.multi_select(options)}
        )

        return self.async_show_form(step_id="cameras", data_schema=schema)

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        return OptionsFlowHandler(config_entry)


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for HomeSec."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self.config_entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Manage options: scan_interval, motion_reset_seconds."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        options = self.config_entry.options
        schema = vol.Schema(
            {
                vol.Optional(
                    "scan_interval",
                    default=options.get("scan_interval", 30),
                ): vol.Coerce(int),
                vol.Optional(
                    "motion_reset_seconds",
                    default=options.get("motion_reset_seconds", DEFAULT_MOTION_RESET_SECONDS),
                ): vol.Coerce(int),
            }
        )
        return self.async_show_form(step_id="init", data_schema=schema)


async def validate_connection(
    hass: HomeAssistant,
    host: str,
    port: int,
    api_key: str | None = None,
    verify_ssl: bool = True,
) -> dict[str, Any]:
    """Validate connection to HomeSec API."""
    base_url = _format_base_url(host, port)
    session = async_get_clientsession(hass)

    async def _request(path: str, auth: bool = True) -> Any:
        headers = {}
        if auth and api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            async with (
                async_timeout.timeout(10),
                session.get(
                    f"{base_url}{path}",
                    headers=headers,
                    ssl=verify_ssl,
                ) as response,
            ):
                if response.status in (401, 403):
                    raise InvalidAuth
                if response.status >= 400:
                    raise CannotConnect
                return await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise CannotConnect from exc

    await _request("/api/v1/health", auth=False)
    cameras = await _request("/api/v1/cameras", auth=True)

    return {
        "title": "HomeSec",
        "version": "unknown",
        "cameras": [camera.get("name") for camera in cameras if camera.get("name")],
    }


async def detect_addon(hass: HomeAssistant) -> tuple[bool, str | None]:
    """Detect HomeSec add-on via Supervisor API."""
    token = os.getenv("SUPERVISOR_TOKEN")
    if not token:
        return False, None

    session = async_get_clientsession(hass)
    url = f"http://supervisor/addons/{ADDON_SLUG}/info"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        async with async_timeout.timeout(10), session.get(url, headers=headers) as response:
            if response.status != 200:
                return False, None
            data = await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return False, None

    addon = data.get("data", {})
    if not addon.get("installed") or addon.get("state") != "started":
        return False, None

    hostname = addon.get("hostname")
    if not hostname:
        return False, None

    return True, hostname


def _format_base_url(host: str, port: int) -> str:
    host = host.rstrip("/")
    if host.startswith("http://") or host.startswith("https://"):
        base = host
    else:
        base = f"http://{host}"

    parsed = urlsplit(base)
    if parsed.port is None and port:
        return f"{base}:{port}"
    return base
