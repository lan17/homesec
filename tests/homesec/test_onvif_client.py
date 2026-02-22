"""Tests for ONVIF camera client wrappers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from homesec.onvif.client import OnvifCameraClient


def test_onvif_camera_client_requires_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """OnvifCameraClient should fail with actionable message when onvif-zeep-async is unavailable."""
    # Given: onvif-zeep-async is unavailable in runtime
    monkeypatch.setattr("homesec.onvif.client._ONVIFCamera", None)

    # When/Then: Instantiating client raises dependency guidance
    with pytest.raises(RuntimeError, match="Missing dependency: onvif-zeep-async"):
        OnvifCameraClient("192.168.1.8", "admin", "password")


async def test_onvif_camera_client_reads_info_profiles_and_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OnvifCameraClient should expose device info, media profiles, and stream URIs."""

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
            self.requests: list[dict[str, Any]] = []

        async def GetProfiles(self) -> list[Any]:
            return [
                SimpleNamespace(
                    token="main-token",
                    Name="Main Stream",
                    VideoEncoderConfiguration=SimpleNamespace(
                        Encoding="H264",
                        Resolution=SimpleNamespace(Width=1920, Height=1080),
                        FrameRateLimit=15,
                        BitrateLimit=4096,
                    ),
                ),
                SimpleNamespace(
                    _token="sub-token",
                    Name="Sub Stream",
                    VideoEncoderConfiguration=SimpleNamespace(
                        Encoding="H264",
                        Resolution=SimpleNamespace(Width=640, Height=360),
                        FrameRateLimit=10,
                        BitrateLimit=512,
                    ),
                ),
            ]

        async def GetStreamUri(self, request: dict[str, Any]) -> Any:
            self.requests.append(request)
            if request["ProfileToken"] == "sub-token":
                raise RuntimeError("stream lookup failed")
            return SimpleNamespace(Uri=f"rtsp://camera/{request['ProfileToken']}")

    class _FakeOnvifCamera:
        instances: list[Any] = []

        def __init__(self, *args: Any) -> None:
            self.args = args
            self.device_service = _FakeDeviceService()
            self.media_service = _FakeMediaService()
            self.__class__.instances.append(self)

        async def update_xaddrs(self) -> None:
            pass

        def create_devicemgmt_service(self) -> _FakeDeviceService:
            return self.device_service

        def create_media_service(self) -> _FakeMediaService:
            return self.media_service

    # Given: A fake ONVIF camera implementation with deterministic responses
    _FakeOnvifCamera.instances = []
    monkeypatch.setattr("homesec.onvif.client._ONVIFCamera", _FakeOnvifCamera)

    # When: Querying device info, profiles, and stream URIs
    client = OnvifCameraClient(
        "192.168.1.10",
        "admin",
        "password",
        port=8080,
        wsdl_dir="/opt/wsdl",
    )
    device_info = await client.get_device_info()
    profiles = await client.get_media_profiles()
    streams = await client.get_stream_uris()

    # Then: ONVIF metadata is translated into stable wrapper models
    assert _FakeOnvifCamera.instances[0].args == (
        "192.168.1.10",
        8080,
        "admin",
        "password",
        "/opt/wsdl",
    )
    assert device_info.manufacturer == "Acme"
    assert device_info.model == "CamPro"
    assert len(profiles) == 2
    assert profiles[0].token == "main-token"
    assert profiles[0].width == 1920
    assert profiles[1].token == "sub-token"
    assert streams[0].uri == "rtsp://camera/main-token"
    assert streams[0].error is None
    assert streams[1].uri is None
    assert streams[1].error == "stream lookup failed"
    assert _FakeOnvifCamera.instances[0].media_service.requests[0]["StreamSetup"]["Transport"] == {
        "Protocol": "RTSP"
    }
