"""Tests for HomeSec coordinator."""

from __future__ import annotations

from datetime import timedelta

import pytest
from custom_components.homesec.const import DOMAIN, EVENT_ALERT
from custom_components.homesec.coordinator import HomesecCoordinator
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry, async_fire_time_changed


@pytest.mark.asyncio
async def test_coordinator_update_success(
    hass,
    aioclient_mock,
    homesec_base_url,
    health_payload,
    cameras_payload,
    stats_payload,
) -> None:
    """Test successful coordinator refresh."""
    # Given: HomeSec API responds successfully
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

    coordinator = HomesecCoordinator(hass, entry)

    # When: Refreshing coordinator
    await coordinator.async_refresh()

    # Then: Data reflects API payloads
    assert coordinator.data["health"]["status"] == "healthy"
    assert len(coordinator.data["cameras"]) == 2
    assert coordinator.data["stats"]["clips_today"] == 3
    assert coordinator.data["connected"] is True


@pytest.mark.asyncio
async def test_coordinator_update_503(hass, aioclient_mock, homesec_base_url) -> None:
    """Test coordinator handles 503 errors."""
    # Given: Health endpoint returns 503
    aioclient_mock.get(
        f"{homesec_base_url}/api/v1/health",
        status=503,
        json={"status": "unhealthy"},
    )

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "host": "homesec.local",
            "port": 8080,
            "api_key": "token",
            "verify_ssl": True,
            "cameras": ["front"],
        },
    )
    entry.add_to_hass(hass)

    coordinator = HomesecCoordinator(hass, entry)

    # When: Refresh runs
    await coordinator.async_refresh()

    # Then: Coordinator marks update as failed
    assert coordinator.last_update_success is False


@pytest.mark.asyncio
async def test_coordinator_event_motion_resets(
    hass,
    aioclient_mock,
    homesec_base_url,
    health_payload,
    cameras_payload,
    stats_payload,
    alert_payload,
) -> None:
    """Test motion state resets after timeout."""
    # Given: Coordinator has initial data and short reset interval
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
        options={"motion_reset_seconds": 1},
    )
    entry.add_to_hass(hass)

    coordinator = HomesecCoordinator(hass, entry)
    await coordinator.async_refresh()
    await coordinator.async_subscribe_events()

    # When: Alert event fires
    hass.bus.async_fire(EVENT_ALERT, alert_payload)
    await hass.async_block_till_done()

    # Then: Motion is active
    assert coordinator.data["motion_active"]["front"] is True

    # When: Time advances beyond reset window
    future = dt_util.utcnow() + timedelta(seconds=2)
    async_fire_time_changed(hass, future)
    await hass.async_block_till_done()

    # Then: Motion resets to inactive
    assert coordinator.data["motion_active"]["front"] is False
