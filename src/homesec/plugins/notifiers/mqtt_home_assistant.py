"""Home Assistant MQTT discovery and event payloads for the MQTT notifier."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from string import Formatter

from pydantic import BaseModel, Field, field_validator

from homesec.models.alert import Alert
from homesec.models.enums import RiskLevel

UNANALYZED_EVENT_TYPE = "unanalyzed"
EVENT_TYPES: list[str] = [str(level) for level in RiskLevel] + [UNANALYZED_EVENT_TYPE]
AVAILABLE_PAYLOAD = "online"
NOT_AVAILABLE_PAYLOAD = "offline"

_SLUG_UNSAFE = re.compile(r"[^a-z0-9]+")
_SLUG_BASE_MAX = 40
_FORBIDDEN_TOPIC_CHARS = ("+", "#", "\x00")


def _require_publishable_topic(topic: str) -> None:
    if not topic or topic.startswith("/") or topic.endswith("/"):
        raise ValueError("topic must be non-empty and must not start or end with '/'")
    if any(char in topic for char in _FORBIDDEN_TOPIC_CHARS):
        raise ValueError("topic must not contain MQTT wildcards or NUL")


class HomeAssistantConfig(BaseModel):
    """Opt-in Home Assistant MQTT integration settings; presence enables it."""

    discovery_prefix: str = "homeassistant"
    event_topic_template: str = "homecam/events/{camera_slug}"
    availability_topic: str = "homecam/status"
    # HA subscribes to a state topic only after it processes the discovery config.
    discovery_settle_seconds: float = Field(default=1.0, ge=0)

    @field_validator("discovery_prefix", "availability_topic")
    @classmethod
    def _validate_plain_topic(cls, value: str) -> str:
        _require_publishable_topic(value)
        return value

    @field_validator("event_topic_template")
    @classmethod
    def _validate_event_topic_template(cls, value: str) -> str:
        try:
            fields = [
                (name, spec, conversion)
                for _, name, spec, conversion in Formatter().parse(value)
                if name is not None
            ]
            if not fields or any(field != ("camera_slug", "", None) for field in fields):
                raise ValueError("only literal {camera_slug} placeholders are supported")
            expanded = value.format(camera_slug="camera")
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f"event_topic_template has an invalid placeholder: {exc}") from exc
        _require_publishable_topic(expanded)
        return value


class HomeAssistantDevice(BaseModel):
    """Device registry entry that groups a camera's entities."""

    identifiers: list[str]
    name: str
    manufacturer: str = "HomeSec"
    model: str = "Camera alerts"


class HomeAssistantOrigin(BaseModel):
    """Identifies HomeSec as the source of a discovery message."""

    name: str = "HomeSec"
    support_url: str = "https://github.com/lan17/homesec"


class HomeAssistantEventDiscovery(BaseModel):
    """Single-component MQTT discovery payload for a camera's alert event entity."""

    name: str = "Alerts"
    unique_id: str
    state_topic: str
    event_types: list[str]
    availability_topic: str
    icon: str = "mdi:cctv"
    device: HomeAssistantDevice
    origin: HomeAssistantOrigin


class HomeAssistantEventPayload(BaseModel):
    """State message for the event entity; extra keys become entity attributes."""

    event_type: str
    clip_id: str
    camera_name: str
    risk_level: str | None
    activity_type: str | None
    summary: str | None
    notify_reason: str
    view_url: str | None
    ts: datetime
    upload_failed: bool
    vlm_failed: bool


def camera_slug(camera_name: str) -> str:
    """Return an ID- and topic-safe slug that stays distinct for distinct camera names."""
    base = _SLUG_UNSAFE.sub("_", camera_name.lower()).strip("_")[:_SLUG_BASE_MAX].strip("_")
    digest = hashlib.sha256(camera_name.encode("utf-8")).hexdigest()[:6]
    return f"{base or 'camera'}_{digest}"


def event_topic(camera_name: str, config: HomeAssistantConfig) -> str:
    return config.event_topic_template.format(camera_slug=camera_slug(camera_name))


def discovery_topic(camera_name: str, config: HomeAssistantConfig) -> str:
    return f"{config.discovery_prefix}/event/homesec_{camera_slug(camera_name)}/config"


def build_discovery(
    camera_name: str, config: HomeAssistantConfig
) -> tuple[str, HomeAssistantEventDiscovery]:
    """Return the discovery topic and payload for one camera."""
    slug = camera_slug(camera_name)
    payload = HomeAssistantEventDiscovery(
        unique_id=f"homesec_{slug}_alerts",
        state_topic=event_topic(camera_name, config),
        event_types=list(EVENT_TYPES),
        availability_topic=config.availability_topic,
        device=HomeAssistantDevice(identifiers=[f"homesec_{slug}"], name=camera_name),
        origin=HomeAssistantOrigin(),
    )
    return discovery_topic(camera_name, config), payload


def build_event_payload(alert: Alert) -> HomeAssistantEventPayload:
    """Map an alert to an event message; unanalyzed alerts carry no invented risk."""
    risk = str(alert.risk_level) if alert.risk_level is not None else None
    return HomeAssistantEventPayload(
        event_type=risk or UNANALYZED_EVENT_TYPE,
        clip_id=alert.clip_id,
        camera_name=alert.camera_name,
        risk_level=risk,
        activity_type=alert.activity_type,
        summary=alert.summary,
        notify_reason=alert.notify_reason,
        view_url=alert.view_url,
        # Pipeline alerts use naive process-local time; the HA boundary needs an offset.
        ts=alert.ts.astimezone(),
        upload_failed=alert.upload_failed,
        vlm_failed=alert.vlm_failed,
    )
