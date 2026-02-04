"""Tests for HomeSec entities."""

from __future__ import annotations

import pytest
from custom_components.homesec.const import DOMAIN, EVENT_ALERT
from pytest_homeassistant_custom_component.common import MockConfigEntry


@pytest.mark.asyncio
async def test_entities_reflect_api_state(
    hass,
    aioclient_mock,
    homesec_base_url,
    health_payload,
    cameras_payload,
    stats_payload,
) -> None:
    """Test entities reflect API data."""
    # Given: HomeSec API data for hub and cameras
    aioclient_mock.get(f"{homesec_base_url}/api/v1/health", json=health_payload)
    aioclient_mock.get(f"{homesec_base_url}/api/v1/cameras", json=cameras_payload)
    aioclient_mock.get(f"{homesec_base_url}/api/v1/stats", json=stats_payload)

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "host": "homesec.local",
            "port": 8080,
            "api_key": "token",
            "verify_ssl": True,
            "cameras": ["front", "back"],
        },
    )
    entry.add_to_hass(hass)

    # When: The integration is set up
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    # Then: Hub and camera entities are created with expected states
    assert hass.states.get("sensor.homesec_hub_system_health").state == "healthy"
    assert hass.states.get("sensor.homesec_hub_alerts_today").state == "2"
    assert hass.states.get("binary_sensor.homesec_front_online").state == "on"
    assert hass.states.get("binary_sensor.homesec_back_online").state == "off"
    assert hass.states.get("switch.homesec_front_enabled").state == "on"
    assert hass.states.get("switch.homesec_back_enabled").state == "off"


@pytest.mark.asyncio
async def test_motion_event_updates_entities(
    hass,
    aioclient_mock,
    homesec_base_url,
    health_payload,
    cameras_payload,
    stats_payload,
    alert_payload,
) -> None:
    """Test motion event updates motion and last activity."""
    # Given: Integration set up and alert event payload
    aioclient_mock.get(f"{homesec_base_url}/api/v1/health", json=health_payload)
    aioclient_mock.get(f"{homesec_base_url}/api/v1/cameras", json=cameras_payload)
    aioclient_mock.get(f"{homesec_base_url}/api/v1/stats", json=stats_payload)

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "host": "homesec.local",
            "port": 8080,
            "api_key": "token",
            "verify_ssl": True,
            "cameras": ["front"],
        },
        options={"motion_reset_seconds": 30},
    )
    entry.add_to_hass(hass)

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    # When: An alert event is fired
    hass.bus.async_fire(EVENT_ALERT, alert_payload)
    await hass.async_block_till_done()

    # Then: Motion and last activity sensors update
    assert hass.states.get("binary_sensor.homesec_front_motion").state == "on"
    assert hass.states.get("sensor.homesec_front_last_activity").state == "person"
