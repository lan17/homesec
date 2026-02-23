"""ONVIF discovery and probe endpoints."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from homesec.api.errors import APIError, APIErrorCode
from homesec.onvif.client import (
    OnvifCameraClient,
    OnvifDeviceInfo,
    OnvifMediaProfile,
    OnvifStreamUri,
)
from homesec.onvif.discovery import discover_cameras

logger = logging.getLogger(__name__)

router = APIRouter(tags=["onvif"])
_ONVIF_CLIENT_CLOSE_TIMEOUT_S = 2.0


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class DiscoverRequest(BaseModel):
    timeout_s: float = Field(default=8.0, gt=0.0, le=30.0)
    attempts: int = Field(default=2, ge=1, le=5)
    ttl: int = Field(default=4, ge=1, le=255)


class DiscoveredCameraResponse(BaseModel):
    ip: str
    xaddr: str
    scopes: list[str]
    types: list[str]


class ProbeRequest(BaseModel):
    host: str
    port: int = Field(default=80, ge=1, le=65535)
    timeout_s: float = Field(default=15.0, gt=0.0, le=120.0)
    username: str = Field(min_length=1, pattern=r".*\S.*")
    password: str = Field(min_length=1, pattern=r".*\S.*")


class MediaProfileResponse(BaseModel):
    token: str
    name: str
    video_encoding: str | None
    width: int | None
    height: int | None
    frame_rate_limit: int | None
    bitrate_limit_kbps: int | None
    stream_uri: str | None
    stream_error: str | None


class DeviceInfoResponse(BaseModel):
    manufacturer: str
    model: str
    firmware_version: str
    serial_number: str
    hardware_id: str


class ProbeResponse(BaseModel):
    device: DeviceInfoResponse
    profiles: list[MediaProfileResponse]


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.post("/api/v1/onvif/discover", response_model=list[DiscoveredCameraResponse])
async def discover_onvif_cameras(
    payload: DiscoverRequest | None = None,
) -> list[DiscoveredCameraResponse]:
    """Trigger WS-Discovery scan and return discovered ONVIF cameras."""
    req = payload or DiscoverRequest()
    try:
        cameras = await asyncio.to_thread(
            discover_cameras,
            req.timeout_s,
            attempts=req.attempts,
            ttl=req.ttl,
        )
    except Exception as exc:
        logger.warning(
            "ONVIF discovery failed (timeout_s=%.2f attempts=%d ttl=%d): %s",
            req.timeout_s,
            req.attempts,
            req.ttl,
            exc,
            exc_info=True,
        )
        raise APIError(
            f"ONVIF discovery failed: {exc}",
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_code=APIErrorCode.ONVIF_DISCOVER_FAILED,
        ) from exc
    return [
        DiscoveredCameraResponse(
            ip=cam.ip,
            xaddr=cam.xaddr,
            scopes=list(cam.scopes),
            types=list(cam.types),
        )
        for cam in cameras
    ]


@router.post("/api/v1/onvif/probe", response_model=ProbeResponse)
async def probe_onvif_camera(payload: ProbeRequest) -> ProbeResponse:
    """Probe an ONVIF camera for device info, profiles, and stream URIs."""
    username = payload.username.strip()
    password = payload.password.strip()
    client = OnvifCameraClient(
        payload.host,
        username,
        password,
        port=payload.port,
    )
    try:
        device_info, profiles, streams = await asyncio.wait_for(
            _run_onvif_probe(client),
            timeout=payload.timeout_s,
        )
    except TimeoutError as exc:
        logger.warning(
            "ONVIF probe timed out after %.2fs for %s:%d",
            payload.timeout_s,
            payload.host,
            payload.port,
            exc_info=True,
        )
        raise APIError(
            f"ONVIF probe timed out after {payload.timeout_s:.2f}s",
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            error_code=APIErrorCode.ONVIF_PROBE_TIMEOUT,
        ) from exc
    except Exception as exc:
        logger.warning(
            "ONVIF probe failed for %s:%d: %s",
            payload.host,
            payload.port,
            exc,
            exc_info=True,
        )
        raise APIError(
            f"ONVIF probe failed: {exc}",
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_code=APIErrorCode.ONVIF_PROBE_FAILED,
        ) from exc
    finally:
        await _close_onvif_client(client, host=payload.host, port=payload.port)

    # Index stream results by profile token for merging.
    stream_by_token = {s.profile_token: s for s in streams}

    merged_profiles = [
        MediaProfileResponse(
            token=p.token,
            name=p.name,
            video_encoding=p.video_encoding,
            width=p.width,
            height=p.height,
            frame_rate_limit=p.frame_rate_limit,
            bitrate_limit_kbps=p.bitrate_limit_kbps,
            stream_uri=stream_by_token[p.token].uri if p.token in stream_by_token else None,
            stream_error=stream_by_token[p.token].error if p.token in stream_by_token else None,
        )
        for p in profiles
    ]

    return ProbeResponse(
        device=DeviceInfoResponse(
            manufacturer=device_info.manufacturer,
            model=device_info.model,
            firmware_version=device_info.firmware_version,
            serial_number=device_info.serial_number,
            hardware_id=device_info.hardware_id,
        ),
        profiles=merged_profiles,
    )


async def _run_onvif_probe(
    client: OnvifCameraClient,
) -> tuple[OnvifDeviceInfo, list[OnvifMediaProfile], list[OnvifStreamUri]]:
    device_info = await client.get_device_info()
    profiles = await client.get_media_profiles()
    streams = await client.get_stream_uris(profiles)
    return device_info, profiles, streams


async def _close_onvif_client(client: OnvifCameraClient, *, host: str, port: int) -> None:
    try:
        await asyncio.wait_for(client.close(), timeout=_ONVIF_CLIENT_CLOSE_TIMEOUT_S)
    except TimeoutError:
        logger.warning(
            "ONVIF probe client close timed out after %.2fs for %s:%d",
            _ONVIF_CLIENT_CLOSE_TIMEOUT_S,
            host,
            port,
            exc_info=True,
        )
    except Exception:
        logger.warning(
            "ONVIF probe client close failed for %s:%d",
            host,
            port,
            exc_info=True,
        )
