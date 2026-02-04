"""Diagnostics for HomeSec."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import CONF_API_KEY, DOMAIN


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]

    data = dict(entry.data)
    if CONF_API_KEY in data:
        data[CONF_API_KEY] = "REDACTED"

    return {
        "config_entry": data,
        "options": dict(entry.options),
        "coordinator_data": coordinator.data,
    }
