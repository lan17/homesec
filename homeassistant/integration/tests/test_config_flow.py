"""Tests for HomeSec config flow."""

from __future__ import annotations

import pytest
from custom_components.homesec.const import DOMAIN
from homeassistant import config_entries


@pytest.mark.asyncio
async def test_config_flow_manual_success(
    hass,
    aioclient_mock,
    homesec_base_url,
    health_payload,
    cameras_payload,
) -> None:
    """Test manual setup completes with camera selection."""
    # Given: A reachable HomeSec API with cameras
    aioclient_mock.get(f"{homesec_base_url}/api/v1/health", json=health_payload)
    aioclient_mock.get(f"{homesec_base_url}/api/v1/cameras", json=cameras_payload)

    # When: User starts manual config flow
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    # Then: Manual step is shown
    assert result["type"] == "form"
    assert result["step_id"] == "manual"

    # When: User submits host/port/api key
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "host": "homesec.local",
            "port": 8080,
            "api_key": "token",
            "verify_ssl": True,
        },
    )

    # Then: Camera selection step appears
    assert result["type"] == "form"
    assert result["step_id"] == "cameras"

    # When: User selects cameras
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {"cameras": ["front"]},
    )

    # Then: Config entry is created
    assert result["type"] == "create_entry"
    assert result["data"]["cameras"] == ["front"]


@pytest.mark.asyncio
async def test_config_flow_manual_invalid_auth(
    hass,
    aioclient_mock,
    homesec_base_url,
    health_payload,
) -> None:
    """Test manual setup shows invalid auth on 401."""
    # Given: Health endpoint is reachable but cameras endpoint rejects auth
    aioclient_mock.get(f"{homesec_base_url}/api/v1/health", json=health_payload)
    aioclient_mock.get(
        f"{homesec_base_url}/api/v1/cameras",
        status=401,
        json={"detail": "Unauthorized"},
    )

    # When: User submits manual config
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "host": "homesec.local",
            "port": 8080,
            "api_key": "bad",
            "verify_ssl": True,
        },
    )

    # Then: Invalid auth error shown
    assert result["type"] == "form"
    assert result["errors"]["base"] == "invalid_auth"


@pytest.mark.asyncio
async def test_config_flow_addon_detected(
    hass,
    aioclient_mock,
    monkeypatch,
    health_payload,
    cameras_payload,
) -> None:
    """Test add-on discovery path."""
    # Given: Supervisor reports HomeSec add-on running
    monkeypatch.setenv("SUPERVISOR_TOKEN", "token")
    aioclient_mock.get(
        "http://supervisor/addons/homesec/info",
        json={
            "data": {
                "installed": True,
                "state": "started",
                "hostname": "abc123-homesec",
            }
        },
    )
    aioclient_mock.get("http://abc123-homesec:8080/api/v1/health", json=health_payload)
    aioclient_mock.get("http://abc123-homesec:8080/api/v1/cameras", json=cameras_payload)

    # When: User starts config flow
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    # Then: Add-on step is shown
    assert result["type"] == "form"
    assert result["step_id"] == "addon"

    # When: User confirms add-on
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {})

    # Then: Camera selection step is shown
    assert result["type"] == "form"
    assert result["step_id"] == "cameras"

    # When: User selects cameras
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {"cameras": ["front", "back"]},
    )

    # Then: Config entry is created
    assert result["type"] == "create_entry"
    assert result["data"]["addon"] is True
    assert result["data"]["cameras"] == ["front", "back"]
