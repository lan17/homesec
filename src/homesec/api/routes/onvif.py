"""ONVIF discovery and probe endpoints."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from homesec.api.errors import APIError, APIErrorCode
from homesec.onvif.client import OnvifCameraClient
from homesec.onvif.discovery import discover_cameras

logger = logging.getLogger(__name__)

router = APIRouter(tags=["onvif"])


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
    username: str
    password: str


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
    cameras = await asyncio.to_thread(
        discover_cameras,
        req.timeout_s,
        attempts=req.attempts,
        ttl=req.ttl,
    )
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
    client = OnvifCameraClient(
        payload.host,
        payload.username,
        payload.password,
        port=payload.port,
    )
    try:
        device_info = await client.get_device_info()
        profiles = await client.get_media_profiles()
        streams = await client.get_stream_uris(profiles)
    except Exception as exc:
        logger.warning("ONVIF probe failed for %s:%d: %s", payload.host, payload.port, exc)
        raise APIError(
            f"ONVIF probe failed: {exc}",
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_code=APIErrorCode.ONVIF_PROBE_FAILED,
        ) from exc
    finally:
        await client.close()

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
