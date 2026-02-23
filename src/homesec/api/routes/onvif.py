"""ONVIF discovery and probe endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, status
from pydantic import BaseModel, Field, field_validator

from homesec.api.errors import APIError, APIErrorCode
from homesec.onvif.client import OnvifCameraClient
from homesec.onvif.discovery import discover_cameras
from homesec.onvif.service import (
    DEFAULT_CLIENT_CLOSE_TIMEOUT_S,
    DEFAULT_DISCOVER_ATTEMPTS,
    DEFAULT_DISCOVER_TIMEOUT_S,
    DEFAULT_DISCOVER_TTL,
    DEFAULT_ONVIF_PORT,
    DEFAULT_PROBE_TIMEOUT_S,
    OnvifDiscoverError,
    OnvifDiscoverOptions,
    OnvifProbeError,
    OnvifProbeOptions,
    OnvifProbeTimeoutError,
    OnvifService,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["onvif"])
_ONVIF_CLIENT_CLOSE_TIMEOUT_S = DEFAULT_CLIENT_CLOSE_TIMEOUT_S


def _build_onvif_service() -> OnvifService:
    return OnvifService(
        discover_fn=discover_cameras,
        client_factory=OnvifCameraClient,
        close_timeout_s=_ONVIF_CLIENT_CLOSE_TIMEOUT_S,
        service_logger=logger,
    )


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class DiscoverRequest(BaseModel):
    timeout_s: float = Field(default=DEFAULT_DISCOVER_TIMEOUT_S, gt=0.0, le=30.0)
    attempts: int = Field(default=DEFAULT_DISCOVER_ATTEMPTS, ge=1, le=5)
    ttl: int = Field(default=DEFAULT_DISCOVER_TTL, ge=1, le=255)


class DiscoveredCameraResponse(BaseModel):
    ip: str
    xaddr: str
    scopes: list[str]
    types: list[str]


class ProbeRequest(BaseModel):
    host: str = Field(min_length=1)
    port: int = Field(default=DEFAULT_ONVIF_PORT, ge=1, le=65535)
    timeout_s: float = Field(default=DEFAULT_PROBE_TIMEOUT_S, gt=0.0, le=120.0)
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)

    @field_validator("host", "username", "password", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


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
    onvif_service = _build_onvif_service()
    try:
        cameras = await onvif_service.discover(
            OnvifDiscoverOptions(
                timeout_s=req.timeout_s,
                attempts=req.attempts,
                ttl=req.ttl,
            )
        )
    except OnvifDiscoverError as exc:
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
    onvif_service = _build_onvif_service()
    try:
        probe_result = await onvif_service.probe(
            OnvifProbeOptions(
                host=payload.host,
                username=payload.username,
                password=payload.password,
                port=payload.port,
                timeout_s=payload.timeout_s,
            )
        )
    except OnvifProbeTimeoutError as exc:
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
    except OnvifProbeError as exc:
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

    # Index stream results by profile token for merging.
    stream_by_token = {stream.profile_token: stream for stream in probe_result.streams}

    merged_profiles = [
        MediaProfileResponse(
            token=profile.token,
            name=profile.name,
            video_encoding=profile.video_encoding,
            width=profile.width,
            height=profile.height,
            frame_rate_limit=profile.frame_rate_limit,
            bitrate_limit_kbps=profile.bitrate_limit_kbps,
            stream_uri=(
                stream_by_token[profile.token].uri if profile.token in stream_by_token else None
            ),
            stream_error=(
                stream_by_token[profile.token].error if profile.token in stream_by_token else None
            ),
        )
        for profile in probe_result.profiles
    ]

    return ProbeResponse(
        device=DeviceInfoResponse(
            manufacturer=probe_result.device_info.manufacturer,
            model=probe_result.device_info.model,
            firmware_version=probe_result.device_info.firmware_version,
            serial_number=probe_result.device_info.serial_number,
            hardware_id=probe_result.device_info.hardware_id,
        ),
        profiles=merged_profiles,
    )
