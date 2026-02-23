"""Tests for ONVIF API endpoints (discover + probe)."""

from __future__ import annotations

import asyncio
import time
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
    def __init__(self, *, server_config: FastAPIServerConfig | None = None) -> None:
        self.repository = _StubRepository()
        self.storage = _StubStorage()
        self.sources: list[Any] = []
        self._config = SimpleNamespace(
            server=server_config or FastAPIServerConfig(),
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


def _client(*, server_config: FastAPIServerConfig | None = None) -> TestClient:
    return TestClient(create_app(_StubApp(server_config=server_config)))


# ---------------------------------------------------------------------------
# POST /api/v1/onvif/discover
# ---------------------------------------------------------------------------


def test_discover_returns_found_cameras(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /onvif/discover should return discovered cameras."""
    # Given: Discovery returns multiple ONVIF cameras
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

    # When: Calling discover endpoint with explicit scan parameters
    client = _client()
    response = client.post("/api/v1/onvif/discover", json={"timeout_s": 5.0, "attempts": 1})

    # Then: API returns discovered camera metadata
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

    # Given: Discovery returns no cameras
    def mock_discover(timeout_s: float = 8.0, *, attempts: int = 2, ttl: int = 4):
        return []

    monkeypatch.setattr("homesec.api.routes.onvif.discover_cameras", mock_discover)

    # When: Calling discover endpoint with default payload
    client = _client()
    response = client.post("/api/v1/onvif/discover", json={})

    # Then: API returns an empty discovered-camera list
    assert response.status_code == 200
    assert response.json() == []


def test_discover_maps_runtime_failures_to_canonical_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /onvif/discover should map discovery runtime failures to ONVIF_DISCOVER_FAILED."""

    # Given: Discovery runtime fails due underlying WS-Discovery error
    def failing_discover(timeout_s: float = 8.0, *, attempts: int = 2, ttl: int = 4):
        raise RuntimeError("wsd failed")

    monkeypatch.setattr("homesec.api.routes.onvif.discover_cameras", failing_discover)

    # When: Calling discover endpoint
    client = _client()
    response = client.post("/api/v1/onvif/discover", json={})

    # Then: API responds with canonical ONVIF discovery failure
    assert response.status_code == 502
    data = response.json()
    assert data["error_code"] == "ONVIF_DISCOVER_FAILED"
    assert "wsd failed" in data["detail"]


def test_discover_validates_attempt_bounds() -> None:
    """POST /onvif/discover should reject invalid attempt bounds via request validation."""
    # Given: An ONVIF discover request with an invalid attempt count
    client = _client()

    # When: Calling discover endpoint with attempts less than 1
    response = client.post("/api/v1/onvif/discover", json={"attempts": 0})

    # Then: API should return canonical validation error envelope
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "REQUEST_VALIDATION_FAILED"
    assert data["detail"] == "Request validation failed"


def test_onvif_routes_require_api_key_when_auth_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ONVIF endpoints should require API key auth when server auth is enabled."""
    # Given: Auth-enabled server config and deterministic ONVIF dependencies
    monkeypatch.setenv("HOMESEC_TEST_API_KEY", "homesec-test-key")
    server_config = FastAPIServerConfig(
        auth_enabled=True,
        api_key_env="HOMESEC_TEST_API_KEY",
    )
    monkeypatch.setattr("homesec.api.routes.onvif.discover_cameras", lambda *args, **kwargs: [])
    monkeypatch.setattr("homesec.onvif.client.ONVIFCamera", _FakeOnvifCamera)
    client = _client(server_config=server_config)
    payload = {
        "host": "192.168.1.10",
        "port": 80,
        "username": "admin",
        "password": "secret",
    }

    # When: Calling ONVIF endpoints without a valid bearer token
    discover_missing_auth = client.post("/api/v1/onvif/discover", json={})
    discover_wrong_auth = client.post(
        "/api/v1/onvif/discover",
        json={},
        headers={"Authorization": "Bearer wrong-key"},
    )
    probe_missing_auth = client.post("/api/v1/onvif/probe", json=payload)
    probe_wrong_auth = client.post(
        "/api/v1/onvif/probe",
        json=payload,
        headers={"Authorization": "Bearer wrong-key"},
    )

    # Then: API rejects unauthorized ONVIF requests with canonical UNAUTHORIZED envelope
    for response in (
        discover_missing_auth,
        discover_wrong_auth,
        probe_missing_auth,
        probe_wrong_auth,
    ):
        assert response.status_code == 401
        assert response.json()["error_code"] == "UNAUTHORIZED"

    # When: Calling ONVIF endpoints with the configured bearer token
    headers = {"Authorization": "Bearer homesec-test-key"}
    discover_ok = client.post("/api/v1/onvif/discover", json={}, headers=headers)
    probe_ok = client.post("/api/v1/onvif/probe", json=payload, headers=headers)

    # Then: API allows ONVIF requests for the configured API key
    assert discover_ok.status_code == 200
    assert probe_ok.status_code == 200


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
    def __init__(self) -> None:
        self.profile_calls = 0

    async def GetProfiles(self) -> list[Any]:
        self.profile_calls += 1
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
    instances: list[_FakeOnvifCamera] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.device_service = _FakeDeviceService()
        self.media_service = _FakeMediaService()
        self.__class__.instances.append(self)

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
    # Given: A probe request against a camera with deterministic ONVIF responses
    _FakeOnvifCamera.instances = []
    monkeypatch.setattr("homesec.onvif.client.ONVIFCamera", _FakeOnvifCamera)

    # When: Probing camera metadata and stream profiles
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

    # Then: Endpoint returns merged profile info and resolves profiles once
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
    assert _FakeOnvifCamera.instances[0].media_service.profile_calls == 1


def test_probe_returns_error_on_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /onvif/probe should return ONVIF_PROBE_FAILED when camera is unreachable."""

    # Given: ONVIF connection fails while probing device metadata
    class _FailingOnvifCamera:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def update_xaddrs(self) -> None:
            raise ConnectionError("Camera unreachable")

        async def close(self) -> None:
            pass

    monkeypatch.setattr("homesec.onvif.client.ONVIFCamera", _FailingOnvifCamera)

    # When: Calling probe endpoint with unreachable host
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

    # Then: API maps failure to ONVIF probe error envelope
    assert response.status_code == 502
    data = response.json()
    assert data["error_code"] == "ONVIF_PROBE_FAILED"
    assert "Camera unreachable" in data["detail"]


def test_probe_rejects_blank_credentials() -> None:
    """POST /onvif/probe should reject blank credential fields during request validation."""
    # Given: A probe payload with blank credentials
    client = _client()
    payload = {
        "host": "192.168.1.50",
        "port": 80,
        "username": "   ",
        "password": "",
    }

    # When: Calling probe endpoint
    response = client.post("/api/v1/onvif/probe", json=payload)

    # Then: API responds with canonical validation error envelope
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "REQUEST_VALIDATION_FAILED"
    assert data["detail"] == "Request validation failed"


def test_probe_times_out_with_canonical_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /onvif/probe should return ONVIF_PROBE_TIMEOUT when probe exceeds timeout."""

    class _SlowOnvifClient:
        instances: list[_SlowOnvifClient] = []

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ = args
            _ = kwargs
            self.closed = False
            self.__class__.instances.append(self)

        async def get_device_info(self) -> Any:
            await asyncio.sleep(0.05)
            return SimpleNamespace(
                manufacturer="Acme",
                model="SlowCam",
                firmware_version="1.0.0",
                serial_number="SN-TIMEOUT",
                hardware_id="HW-TIMEOUT",
            )

        async def get_media_profiles(self) -> list[Any]:
            return []

        async def get_stream_uris(self, profiles: list[Any] | None = None) -> list[Any]:
            _ = profiles
            return []

        async def close(self) -> None:
            self.closed = True

    # Given: ONVIF probe client exceeds the request timeout
    _SlowOnvifClient.instances = []
    monkeypatch.setattr("homesec.api.routes.onvif.OnvifCameraClient", _SlowOnvifClient)
    client = _client()

    # When: Calling probe endpoint with a short timeout
    response = client.post(
        "/api/v1/onvif/probe",
        json={
            "host": "192.168.1.50",
            "port": 80,
            "timeout_s": 0.01,
            "username": "admin",
            "password": "secret",
        },
    )

    # Then: API maps timeout to canonical ONVIF timeout error and closes client
    assert response.status_code == 504
    data = response.json()
    assert data["error_code"] == "ONVIF_PROBE_TIMEOUT"
    assert "timed out" in data["detail"]
    assert _SlowOnvifClient.instances[0].closed is True


def test_probe_trims_credentials_before_client_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /onvif/probe should normalize credential whitespace before ONVIF client init."""

    class _CapturingOnvifClient:
        instances: list[_CapturingOnvifClient] = []

        def __init__(self, host: str, username: str, password: str, *, port: int) -> None:
            self.host = host
            self.username = username
            self.password = password
            self.port = port
            self.__class__.instances.append(self)

        async def get_device_info(self) -> Any:
            return SimpleNamespace(
                manufacturer="Acme",
                model="TrimCam",
                firmware_version="1.0.0",
                serial_number="SN-TRIM",
                hardware_id="HW-TRIM",
            )

        async def get_media_profiles(self) -> list[Any]:
            return []

        async def get_stream_uris(self, profiles: list[Any] | None = None) -> list[Any]:
            _ = profiles
            return []

        async def close(self) -> None:
            return None

    # Given: A probe payload with padded credentials
    _CapturingOnvifClient.instances = []
    monkeypatch.setattr("homesec.api.routes.onvif.OnvifCameraClient", _CapturingOnvifClient)
    client = _client()

    # When: Calling probe endpoint with whitespace around credentials
    response = client.post(
        "/api/v1/onvif/probe",
        json={
            "host": "192.168.1.30",
            "port": 80,
            "username": "  admin  ",
            "password": "  secret  ",
        },
    )

    # Then: ONVIF client receives normalized credential values
    assert response.status_code == 200
    assert _CapturingOnvifClient.instances[0].username == "admin"
    assert _CapturingOnvifClient.instances[0].password == "secret"


def test_probe_close_timeout_does_not_extend_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /onvif/probe should bound client close time and preserve timeout behavior."""

    class _SlowCloseOnvifClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ = args
            _ = kwargs

        async def get_device_info(self) -> Any:
            await asyncio.sleep(0.05)
            return SimpleNamespace(
                manufacturer="Acme",
                model="SlowCloseCam",
                firmware_version="1.0.0",
                serial_number="SN-SLOW-CLOSE",
                hardware_id="HW-SLOW-CLOSE",
            )

        async def get_media_profiles(self) -> list[Any]:
            return []

        async def get_stream_uris(self, profiles: list[Any] | None = None) -> list[Any]:
            _ = profiles
            return []

        async def close(self) -> None:
            await asyncio.sleep(1.0)

    # Given: A probe that times out and a close routine that hangs longer than close timeout
    monkeypatch.setattr("homesec.api.routes.onvif.OnvifCameraClient", _SlowCloseOnvifClient)
    monkeypatch.setattr("homesec.api.routes.onvif._ONVIF_CLIENT_CLOSE_TIMEOUT_S", 0.01)
    client = _client()

    # When: Calling probe endpoint and measuring request duration
    started = time.perf_counter()
    response = client.post(
        "/api/v1/onvif/probe",
        json={
            "host": "192.168.1.50",
            "port": 80,
            "timeout_s": 0.01,
            "username": "admin",
            "password": "secret",
        },
    )
    elapsed_s = time.perf_counter() - started

    # Then: API returns canonical timeout quickly despite slow close cleanup
    assert response.status_code == 504
    data = response.json()
    assert data["error_code"] == "ONVIF_PROBE_TIMEOUT"
    assert elapsed_s < 0.5


def test_probe_close_failure_does_not_mask_primary_probe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /onvif/probe should preserve probe failure even when close fails."""

    class _FailingProbeAndCloseClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ = args
            _ = kwargs

        async def get_device_info(self) -> Any:
            raise ConnectionError("Camera unreachable")

        async def get_media_profiles(self) -> list[Any]:
            return []

        async def get_stream_uris(self, profiles: list[Any] | None = None) -> list[Any]:
            _ = profiles
            return []

        async def close(self) -> None:
            raise RuntimeError("close failed")

    # Given: Probe fails and cleanup also raises
    monkeypatch.setattr(
        "homesec.api.routes.onvif.OnvifCameraClient",
        _FailingProbeAndCloseClient,
    )
    client = _client()

    # When: Calling probe endpoint
    response = client.post(
        "/api/v1/onvif/probe",
        json={
            "host": "192.168.1.99",
            "port": 80,
            "username": "admin",
            "password": "wrong",
        },
    )

    # Then: API surfaces canonical probe failure, not cleanup failure
    assert response.status_code == 502
    data = response.json()
    assert data["error_code"] == "ONVIF_PROBE_FAILED"
    assert "Camera unreachable" in data["detail"]
