"""Mobile device registration models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

MobilePlatform = Literal["ios"]
APNSEnvironment = Literal["sandbox", "production"]


class MobileDeviceCapabilities(BaseModel):
    """Feature flags reported by the current iOS app build."""

    deep_links: bool = True
    rich_notifications: bool = False


class MobileDeviceRegistration(BaseModel):
    """Registration payload for an iOS APNs device."""

    platform: MobilePlatform = "ios"
    apns_token: str = Field(min_length=1)
    apns_environment: APNSEnvironment
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


class MobileDeviceUpdate(BaseModel):
    """Mutable fields for a registered mobile device."""

    device_name: str | None = None
    app_version: str | None = None
    enabled: bool | None = None
    capabilities: MobileDeviceCapabilities | None = None


class MobileDeviceRecord(BaseModel):
    """Public mobile device record without raw APNs registration material."""

    id: str
    platform: MobilePlatform
    apns_environment: APNSEnvironment
    bundle_id: str
    device_name: str | None = None
    app_version: str | None = None
    capabilities: MobileDeviceCapabilities = Field(default_factory=MobileDeviceCapabilities)
    enabled: bool
    token_fingerprint: str
    created_at: datetime
    updated_at: datetime
    last_seen_at: datetime | None = None
    last_push_at: datetime | None = None
    last_push_error: str | None = None
