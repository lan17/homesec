"""Mobile device registration endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field, field_validator

from homesec.api.dependencies import get_homesec_app
from homesec.api.errors import APIError, APIErrorCode
from homesec.models.mobile import (
    APNSEnvironment,
    MobileDeviceCapabilities,
    MobileDeviceRecord,
    MobileDeviceRegistration,
    MobileDeviceUpdate,
    MobilePlatform,
)

if TYPE_CHECKING:
    from homesec.app import Application

router = APIRouter(tags=["mobile"])


class MobileDeviceRegisterRequest(BaseModel):
    platform: MobilePlatform = "ios"
    apns_token: str = Field(min_length=1)
    environment: APNSEnvironment
    bundle_id: str = Field(min_length=1)
    device_name: str | None = None
    app_version: str | None = None
    capabilities: MobileDeviceCapabilities = Field(default_factory=MobileDeviceCapabilities)

    @field_validator("apns_token", "bundle_id")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized


class MobileDevicePatchRequest(BaseModel):
    device_name: str | None = None
    app_version: str | None = None
    enabled: bool | None = None
    capabilities: MobileDeviceCapabilities | None = None


class MobileDeviceResponse(BaseModel):
    id: str
    platform: MobilePlatform
    environment: APNSEnvironment
    bundle_id: str
    device_name: str | None = None
    app_version: str | None = None
    capabilities: MobileDeviceCapabilities
    enabled: bool
    token_fingerprint: str
    created_at: datetime
    updated_at: datetime
    last_seen_at: datetime | None = None
    last_push_at: datetime | None = None
    last_push_error: str | None = None


def _device_response(record: MobileDeviceRecord) -> MobileDeviceResponse:
    return MobileDeviceResponse(
        id=record.id,
        platform=record.platform,
        environment=record.apns_environment,
        bundle_id=record.bundle_id,
        device_name=record.device_name,
        app_version=record.app_version,
        capabilities=record.capabilities,
        enabled=record.enabled,
        token_fingerprint=record.token_fingerprint,
        created_at=record.created_at,
        updated_at=record.updated_at,
        last_seen_at=record.last_seen_at,
        last_push_at=record.last_push_at,
        last_push_error=record.last_push_error,
    )


@router.post(
    "/api/v1/mobile/devices",
    response_model=MobileDeviceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_mobile_device(
    payload: MobileDeviceRegisterRequest,
    app: Application = Depends(get_homesec_app),
) -> MobileDeviceResponse:
    """Register or refresh an iOS APNs device."""
    record = await app.mobile_devices.register_device(
        MobileDeviceRegistration(
            platform=payload.platform,
            apns_token=payload.apns_token,
            apns_environment=payload.environment,
            bundle_id=payload.bundle_id,
            device_name=payload.device_name,
            app_version=payload.app_version,
            capabilities=payload.capabilities,
        )
    )
    return _device_response(record)


@router.get("/api/v1/mobile/devices", response_model=list[MobileDeviceResponse])
async def list_mobile_devices(
    include_disabled: bool = False,
    app: Application = Depends(get_homesec_app),
) -> list[MobileDeviceResponse]:
    """List registered iOS devices without raw APNs material."""
    records = await app.mobile_devices.list_devices(include_disabled=include_disabled)
    return [_device_response(record) for record in records]


@router.patch("/api/v1/mobile/devices/{device_id}", response_model=MobileDeviceResponse)
async def update_mobile_device(
    device_id: str,
    payload: MobileDevicePatchRequest,
    app: Application = Depends(get_homesec_app),
) -> MobileDeviceResponse:
    """Update mutable mobile device metadata or enabled state."""
    record = await app.mobile_devices.update_device(
        device_id,
        MobileDeviceUpdate.model_validate(payload.model_dump(exclude_unset=True)),
    )
    if record is None:
        raise APIError(
            "Mobile device not found",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=APIErrorCode.MOBILE_DEVICE_NOT_FOUND,
        )
    return _device_response(record)


@router.delete("/api/v1/mobile/devices/{device_id}", response_model=MobileDeviceResponse)
async def delete_mobile_device(
    device_id: str,
    app: Application = Depends(get_homesec_app),
) -> MobileDeviceResponse:
    """Disable a mobile device without hard-deleting it."""
    record = await app.mobile_devices.disable_device(device_id)
    if record is None:
        raise APIError(
            "Mobile device not found",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=APIErrorCode.MOBILE_DEVICE_NOT_FOUND,
        )
    return _device_response(record)
