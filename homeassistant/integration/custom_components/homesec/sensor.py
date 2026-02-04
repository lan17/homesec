"""Sensor platform for HomeSec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_CAMERAS, DOMAIN
from .coordinator import HomesecCoordinator
from .entity import HomesecCameraEntity, HomesecHubEntity


@dataclass(frozen=True)
class HomesecHubSensorDescription(SensorEntityDescription):
    """Description for HomeSec hub sensors."""


@dataclass(frozen=True)
class HomesecCameraSensorDescription(SensorEntityDescription):
    """Description for HomeSec camera sensors."""


HUB_SENSORS: tuple[HomesecHubSensorDescription, ...] = (
    HomesecHubSensorDescription(
        key="cameras_online",
        name="Cameras Online",
        icon="mdi:cctv",
    ),
    HomesecHubSensorDescription(
        key="alerts_today",
        name="Alerts Today",
        icon="mdi:bell",
    ),
    HomesecHubSensorDescription(
        key="clips_today",
        name="Clips Today",
        icon="mdi:video",
    ),
    HomesecHubSensorDescription(
        key="system_health",
        name="System Health",
        icon="mdi:heart-pulse",
    ),
)

CAMERA_SENSORS: tuple[HomesecCameraSensorDescription, ...] = (
    HomesecCameraSensorDescription(
        key="last_activity",
        name="Last Activity",
        icon="mdi:motion-sensor",
    ),
    HomesecCameraSensorDescription(
        key="risk_level",
        name="Risk Level",
        icon="mdi:alert",
    ),
    HomesecCameraSensorDescription(
        key="health",
        name="Health",
        icon="mdi:camera",
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up HomeSec sensors from a config entry."""
    coordinator: HomesecCoordinator = hass.data[DOMAIN][entry.entry_id]

    entities: list[SensorEntity] = [
        HomesecHubSensor(coordinator, description) for description in HUB_SENSORS
    ]

    camera_names = entry.data.get(CONF_CAMERAS, [])
    for camera_name in camera_names:
        entities.extend(
            HomesecCameraSensor(coordinator, camera_name, description)
            for description in CAMERA_SENSORS
        )

    async_add_entities(entities)


class HomesecHubSensor(HomesecHubEntity, SensorEntity):
    """Hub-level HomeSec sensors."""

    entity_description: HomesecHubSensorDescription

    def __init__(
        self, coordinator: HomesecCoordinator, description: HomesecHubSensorDescription
    ) -> None:
        super().__init__(coordinator)
        self.entity_description = description
        self._attr_unique_id = f"homesec_hub_{description.key}"

    @property
    def native_value(self) -> Any:
        key = self.entity_description.key
        stats = self.coordinator.data.get("stats", {})
        health = self.coordinator.data.get("health", {})

        if key == "system_health":
            return health.get("status")
        if key == "cameras_online":
            return stats.get("cameras_online")
        if key == "alerts_today":
            return stats.get("alerts_today")
        if key == "clips_today":
            return stats.get("clips_today")
        return None


class HomesecCameraSensor(HomesecCameraEntity, SensorEntity):
    """Camera-level HomeSec sensors."""

    entity_description: HomesecCameraSensorDescription

    def __init__(
        self,
        coordinator: HomesecCoordinator,
        camera_name: str,
        description: HomesecCameraSensorDescription,
    ) -> None:
        super().__init__(coordinator, camera_name)
        self.entity_description = description
        self._attr_unique_id = f"homesec_{camera_name}_{description.key}"

    @property
    def native_value(self) -> Any:
        key = self.entity_description.key
        camera = self._get_camera_data() or {}
        alert = self._get_latest_alert() or {}

        if key == "health":
            healthy = camera.get("healthy")
            if healthy is None:
                return None
            return "healthy" if healthy else "unhealthy"

        if key == "last_activity":
            return alert.get("activity_type")

        if key == "risk_level":
            return alert.get("risk_level")

        return None
