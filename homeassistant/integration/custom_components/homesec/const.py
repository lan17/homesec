"""Constants for the HomeSec integration."""

from __future__ import annotations

from datetime import timedelta

from homeassistant.const import Platform

DOMAIN = "homesec"

# Config keys
CONF_HOST = "host"
CONF_PORT = "port"
CONF_API_KEY = "api_key"
CONF_VERIFY_SSL = "verify_ssl"
CONF_CAMERAS = "cameras"
CONF_ADDON = "addon"

# Defaults
DEFAULT_PORT = 8080
DEFAULT_VERIFY_SSL = True
ADDON_SLUG = "homesec"

# Motion sensor
DEFAULT_MOTION_RESET_SECONDS = 30

# Platforms
PLATFORMS: list[Platform] = [Platform.BINARY_SENSOR, Platform.SENSOR, Platform.SWITCH]

# Update intervals
SCAN_INTERVAL_SECONDS = 30
DEFAULT_SCAN_INTERVAL = timedelta(seconds=SCAN_INTERVAL_SECONDS)

# Events
EVENT_ALERT = "homesec_alert"
