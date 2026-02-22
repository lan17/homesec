"""ONVIF client wrappers for device info, media profiles, and stream URIs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, cast

try:
    import onvif as _onvif_pkg  # type: ignore[import-not-found]
    from onvif import ONVIFCamera as _ONVIFCamera  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - exercised via dependency guard tests
    _onvif_pkg = None
    _ONVIFCamera = None


@dataclass(frozen=True, slots=True)
class OnvifDeviceInfo:
    """Device identity metadata returned by ONVIF GetDeviceInformation."""

    manufacturer: str
    model: str
    firmware_version: str
    serial_number: str
    hardware_id: str


@dataclass(frozen=True, slots=True)
class OnvifMediaProfile:
    """Summary of one ONVIF media profile."""

    token: str
    name: str
    video_encoding: str | None
    width: int | None
    height: int | None
    frame_rate_limit: int | None
    bitrate_limit_kbps: int | None


@dataclass(frozen=True, slots=True)
class OnvifStreamUri:
    """RTSP URI lookup result for one media profile."""

    profile_token: str
    profile_name: str
    uri: str | None
    error: str | None


class OnvifCameraClient:
    """Thin async wrapper around onvif-zeep-async ONVIFCamera."""

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        *,
        port: int = 80,
        wsdl_dir: str | None = None,
    ) -> None:
        camera_class = _require_onvif_camera_class()
        resolved_wsdl = wsdl_dir if wsdl_dir is not None else _default_wsdl_dir()
        self._camera = camera_class(host, port, username, password, resolved_wsdl)
        self._initialized = False
        self._device_service: Any | None = None
        self._media_service: Any | None = None

    async def close(self) -> None:
        """Close the underlying ONVIFCamera and its transport sessions."""
        close = getattr(self._camera, "close", None)
        if close is not None:
            await close()

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self._camera.update_xaddrs()
            self._initialized = True

    async def get_device_info(self) -> OnvifDeviceInfo:
        """Return ONVIF device information."""
        info = await (await self._device()).GetDeviceInformation()
        return OnvifDeviceInfo(
            manufacturer=_as_string(getattr(info, "Manufacturer", None)),
            model=_as_string(getattr(info, "Model", None)),
            firmware_version=_as_string(getattr(info, "FirmwareVersion", None)),
            serial_number=_as_string(getattr(info, "SerialNumber", None)),
            hardware_id=_as_string(getattr(info, "HardwareId", None)),
        )

    async def get_media_profiles(self) -> list[OnvifMediaProfile]:
        """Return ONVIF media profile summaries."""
        media_profiles = list(await (await self._media()).GetProfiles())
        profiles: list[OnvifMediaProfile] = []
        for index, profile in enumerate(media_profiles):
            token = _profile_token(profile, index=index)
            name = _as_string(getattr(profile, "Name", None), fallback=token)
            video_cfg = getattr(profile, "VideoEncoderConfiguration", None)
            resolution = getattr(video_cfg, "Resolution", None)
            profiles.append(
                OnvifMediaProfile(
                    token=token,
                    name=name,
                    video_encoding=_as_optional_string(getattr(video_cfg, "Encoding", None)),
                    width=_as_optional_int(getattr(resolution, "Width", None)),
                    height=_as_optional_int(getattr(resolution, "Height", None)),
                    frame_rate_limit=_as_optional_int(getattr(video_cfg, "FrameRateLimit", None)),
                    bitrate_limit_kbps=_as_optional_int(
                        getattr(video_cfg, "BitrateLimit", None)
                    ),
                )
            )
        return profiles

    async def get_stream_uris(self) -> list[OnvifStreamUri]:
        """Return RTSP URI lookup results for each media profile."""
        media = await self._media()
        stream_results: list[OnvifStreamUri] = []
        for profile in await self.get_media_profiles():
            request = {
                "StreamSetup": {
                    "Stream": "RTP-Unicast",
                    "Transport": {"Protocol": "RTSP"},
                },
                "ProfileToken": profile.token,
            }
            try:
                response = await media.GetStreamUri(request)
                uri = _as_optional_string(getattr(response, "Uri", None))
                if uri is None:
                    stream_results.append(
                        OnvifStreamUri(
                            profile_token=profile.token,
                            profile_name=profile.name,
                            uri=None,
                            error="GetStreamUri returned empty Uri",
                        )
                    )
                    continue

                stream_results.append(
                    OnvifStreamUri(
                        profile_token=profile.token,
                        profile_name=profile.name,
                        uri=uri,
                        error=None,
                    )
                )
            except Exception as exc:
                stream_results.append(
                    OnvifStreamUri(
                        profile_token=profile.token,
                        profile_name=profile.name,
                        uri=None,
                        error=str(exc),
                    )
                )
        return stream_results

    async def _device(self) -> Any:
        await self._ensure_initialized()
        if self._device_service is None:
            self._device_service = self._camera.create_devicemgmt_service()
        return self._device_service

    async def _media(self) -> Any:
        await self._ensure_initialized()
        if self._media_service is None:
            self._media_service = self._camera.create_media_service()
        return self._media_service


def _require_onvif_camera_class() -> type[Any]:
    if _ONVIFCamera is None:
        raise RuntimeError(
            "Missing dependency: onvif-zeep-async. Install with: uv pip install onvif-zeep-async"
        )
    return cast(type[Any], _ONVIFCamera)


def _default_wsdl_dir() -> str:
    """Resolve the WSDL directory bundled with onvif-zeep-async.

    The library's own default (``site-packages/wsdl/``) relies on
    ``data_files`` placement which is unreliable across installers and
    platforms.  We look inside the ``onvif`` package directory first,
    which is always present in the wheel.
    """
    if _onvif_pkg is None:
        return ""
    pkg_dir = os.path.dirname(_onvif_pkg.__file__)
    # Preferred: wsdl/ inside the onvif package itself.
    inside_pkg = os.path.join(pkg_dir, "wsdl")
    if os.path.isdir(inside_pkg):
        return inside_pkg
    # Fallback: library default (site-packages/wsdl/).
    site_packages = os.path.dirname(pkg_dir)
    return os.path.join(site_packages, "wsdl")


def _profile_token(profile: Any, *, index: int) -> str:
    for attr in ("token", "_token", "Token"):
        value = getattr(profile, attr, None)
        if isinstance(value, str) and value:
            return value
    return f"profile-{index}"


def _as_string(value: Any, *, fallback: str = "") -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return fallback
    return str(value)


def _as_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        return None


__all__ = [
    "OnvifCameraClient",
    "OnvifDeviceInfo",
    "OnvifMediaProfile",
    "OnvifStreamUri",
]
