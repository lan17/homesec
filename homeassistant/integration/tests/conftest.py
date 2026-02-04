"""Fixtures for HomeSec Home Assistant integration tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

pytest_plugins = "pytest_homeassistant_custom_component"

ROOT = Path(__file__).resolve().parents[2]
INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(INTEGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_ROOT))

from homesec.api.routes.cameras import CameraResponse
from homesec.api.routes.health import HealthResponse
from homesec.api.routes.stats import StatsResponse


@pytest.fixture(autouse=True)
def _enable_custom_integrations(enable_custom_integrations) -> None:
    """Enable custom integrations for HomeSec tests."""
    _ = enable_custom_integrations


@pytest.fixture
def homesec_base_url() -> str:
    return "http://homesec.local:8080"


@pytest.fixture
def health_payload() -> dict[str, Any]:
    data = {
        "status": "healthy",
        "pipeline": "running",
        "postgres": "connected",
        "cameras_online": 1,
    }
    HealthResponse.model_validate(data)
    return data


@pytest.fixture
def stats_payload() -> dict[str, Any]:
    data = {
        "clips_today": 3,
        "alerts_today": 2,
        "cameras_total": 2,
        "cameras_online": 1,
        "uptime_seconds": 120.0,
    }
    StatsResponse.model_validate(data)
    return data


@pytest.fixture
def cameras_payload() -> list[dict[str, Any]]:
    cameras = [
        {
            "name": "front",
            "enabled": True,
            "source_backend": "rtsp",
            "healthy": True,
            "last_heartbeat": 1_695_000_000.0,
            "source_config": {"url": "rtsp://camera"},
        },
        {
            "name": "back",
            "enabled": False,
            "source_backend": "rtsp",
            "healthy": False,
            "last_heartbeat": None,
            "source_config": {"url": "rtsp://camera2"},
        },
    ]

    for camera in cameras:
        CameraResponse.model_validate(camera)
    return cameras


@pytest.fixture
def alert_payload() -> dict[str, Any]:
    return {
        "camera": "front",
        "clip_id": "clip-123",
        "activity_type": "person",
        "risk_level": "high",
        "summary": "Person at front door",
        "view_url": "http://example/clip",
        "storage_uri": "local://clip-123",
        "timestamp": "2026-02-04T00:00:00+00:00",
        "detected_objects": ["person"],
    }
