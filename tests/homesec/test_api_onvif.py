"""Tests for ONVIF API endpoints (discover + probe)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from homesec.api.server import create_app
from homesec.models.config import FastAPIServerConfig
from homesec.onvif.discovery import DiscoveredCamera
from homesec.runtime.models import RuntimeReloadRequest

# ---------------------------------------------------------------------------
# Minimal stub app – the ONVIF routes don't depend on get_homesec_app, but
# the router-level verify_api_key dependency does.
# ---------------------------------------------------------------------------


class _StubRepository:
    async def ping(self) -> bool:
        return True


class _StubStorage:
    async def ping(self) -> bool:
        return True


class _StubApp:
    def __init__(self) -> None:
        self.repository = _StubRepository()
        self.storage = _StubStorage()
        self.sources: list[Any] = []
        self._config = SimpleNamespace(
            server=FastAPIServerConfig(),
            cameras=[],
        )
        self.uptime_seconds = 0.0

    @property
    def config(self):  # type: ignore[override]
        return self._config

    @property
    def pipeline_running(self) -> bool:
        return False

    def get_source(self, camera_name: str) -> None:
        return None

    async def request_runtime_reload(self) -> RuntimeReloadRequest:
        return RuntimeReloadRequest(accepted=True, message="ok", target_generation=1)


def _client() -> TestClient:
    return TestClient(create_app(_StubApp()))


# ---------------------------------------------------------------------------
# POST /api/v1/onvif/discover
# ---------------------------------------------------------------------------


def test_discover_returns_found_cameras(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /onvif/discover should return discovered cameras."""
    fake_cameras = [
        DiscoveredCamera(
            ip="192.168.1.10",
            xaddr="http://192.168.1.10/onvif/device_service",
            scopes=("onvif://www.onvif.org/name/camera1",),
            types=("dn:NetworkVideoTransmitter",),
        ),
        DiscoveredCamera(
            ip="192.168.1.11",
            xaddr="http://192.168.1.11/onvif/device_service",
            scopes=("onvif://www.onvif.org/name/camera2",),
            types=("dn:NetworkVideoTransmitter",),
        ),
    ]

    def mock_discover(timeout_s: float = 8.0, *, attempts: int = 2, ttl: int = 4):
        return fake_cameras

    monkeypatch.setattr("homesec.api.routes.onvif.discover_cameras", mock_discover)

    client = _client()
    response = client.post("/api/v1/onvif/discover", json={"timeout_s": 5.0, "attempts": 1})

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["ip"] == "192.168.1.10"
    assert data[0]["xaddr"] == "http://192.168.1.10/onvif/device_service"
    assert data[0]["scopes"] == ["onvif://www.onvif.org/name/camera1"]
    assert data[0]["types"] == ["dn:NetworkVideoTransmitter"]
    assert data[1]["ip"] == "192.168.1.11"


def test_discover_returns_empty_list_when_none_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /onvif/discover should return empty list when no cameras found."""

    def mock_discover(timeout_s: float = 8.0, *, attempts: int = 2, ttl: int = 4):
        return []

    monkeypatch.setattr("homesec.api.routes.onvif.discover_cameras", mock_discover)

    client = _client()
    response = client.post("/api/v1/onvif/discover", json={})

    assert response.status_code == 200
    assert response.json() == []


# ---------------------------------------------------------------------------
# POST /api/v1/onvif/probe
# ---------------------------------------------------------------------------


class _FakeDeviceService:
    async def GetDeviceInformation(self) -> SimpleNamespace:
        return SimpleNamespace(
            Manufacturer="Acme",
            Model="CamPro",
            FirmwareVersion="1.2.3",
            SerialNumber="SN123",
            HardwareId="HW456",
        )


class _FakeMediaService:
    async def GetProfiles(self) -> list[Any]:
        return [
            SimpleNamespace(
                token="main",
                Name="Main Stream",
                VideoEncoderConfiguration=SimpleNamespace(
                    Encoding="H264",
                    Resolution=SimpleNamespace(Width=1920, Height=1080),
                    FrameRateLimit=15,
                    BitrateLimit=4096,
                ),
            ),
            SimpleNamespace(
                token="sub",
                Name="Sub Stream",
                VideoEncoderConfiguration=SimpleNamespace(
                    Encoding="H264",
                    Resolution=SimpleNamespace(Width=640, Height=360),
                    FrameRateLimit=10,
                    BitrateLimit=512,
                ),
            ),
        ]

    async def GetStreamUri(self, request: dict[str, Any]) -> SimpleNamespace:
        token = request["ProfileToken"]
        return SimpleNamespace(Uri=f"rtsp://192.168.1.10/{token}")


class _FakeOnvifCamera:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.device_service = _FakeDeviceService()
        self.media_service = _FakeMediaService()

    async def update_xaddrs(self) -> None:
        pass

    async def create_devicemgmt_service(self) -> _FakeDeviceService:
        return self.device_service

    async def create_media_service(self) -> _FakeMediaService:
        return self.media_service

    async def close(self) -> None:
        pass


def test_probe_returns_device_info_and_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /onvif/probe should return device info and merged profiles with stream URIs."""
    monkeypatch.setattr("homesec.onvif.client.ONVIFCamera", _FakeOnvifCamera)

    client = _client()
    response = client.post(
        "/api/v1/onvif/probe",
        json={
            "host": "192.168.1.10",
            "port": 80,
            "username": "admin",
            "password": "secret",
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Device info
    assert data["device"]["manufacturer"] == "Acme"
    assert data["device"]["model"] == "CamPro"
    assert data["device"]["firmware_version"] == "1.2.3"
    assert data["device"]["serial_number"] == "SN123"
    assert data["device"]["hardware_id"] == "HW456"

    # Profiles with merged stream URIs
    profiles = data["profiles"]
    assert len(profiles) == 2

    assert profiles[0]["token"] == "main"
    assert profiles[0]["name"] == "Main Stream"
    assert profiles[0]["video_encoding"] == "H264"
    assert profiles[0]["width"] == 1920
    assert profiles[0]["height"] == 1080
    assert profiles[0]["stream_uri"] == "rtsp://192.168.1.10/main"
    assert profiles[0]["stream_error"] is None

    assert profiles[1]["token"] == "sub"
    assert profiles[1]["stream_uri"] == "rtsp://192.168.1.10/sub"


def test_probe_returns_error_on_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /onvif/probe should return ONVIF_PROBE_FAILED when camera is unreachable."""

    class _FailingOnvifCamera:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def update_xaddrs(self) -> None:
            raise ConnectionError("Camera unreachable")

        async def close(self) -> None:
            pass

    monkeypatch.setattr("homesec.onvif.client.ONVIFCamera", _FailingOnvifCamera)

    client = _client()
    response = client.post(
        "/api/v1/onvif/probe",
        json={
            "host": "192.168.1.99",
            "port": 80,
            "username": "admin",
            "password": "wrong",
        },
    )

    assert response.status_code == 502
    data = response.json()
    assert data["error_code"] == "ONVIF_PROBE_FAILED"
    assert "Camera unreachable" in data["detail"]
