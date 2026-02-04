"""Binary sensor platform for HomeSec."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.components.binary_sensor import BinarySensorEntity, BinarySensorEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_CAMERAS, DOMAIN
from .coordinator import HomesecCoordinator
from .entity import HomesecCameraEntity


@dataclass(frozen=True)
class HomesecBinarySensorDescription(BinarySensorEntityDescription):
    """Description for HomeSec binary sensors."""


CAMERA_BINARY_SENSORS: tuple[HomesecBinarySensorDescription, ...] = (
    HomesecBinarySensorDescription(
        key="motion",
        name="Motion",
        icon="mdi:motion-sensor",
    ),
    HomesecBinarySensorDescription(
        key="person",
        name="Person",
        icon="mdi:account",
    ),
    HomesecBinarySensorDescription(
        key="online",
        name="Online",
        icon="mdi:wifi",
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up HomeSec binary sensors from a config entry."""
    coordinator: HomesecCoordinator = hass.data[DOMAIN][entry.entry_id]

    camera_names = entry.data.get(CONF_CAMERAS, [])
    entities: list[BinarySensorEntity] = []

    for camera_name in camera_names:
        entities.extend(
            HomesecCameraBinarySensor(coordinator, camera_name, description)
            for description in CAMERA_BINARY_SENSORS
        )

    async_add_entities(entities)


class HomesecCameraBinarySensor(HomesecCameraEntity, BinarySensorEntity):
    """Camera-level HomeSec binary sensors."""

    entity_description: HomesecBinarySensorDescription

    def __init__(
        self,
        coordinator: HomesecCoordinator,
        camera_name: str,
        description: HomesecBinarySensorDescription,
    ) -> None:
        super().__init__(coordinator, camera_name)
        self.entity_description = description
        self._attr_unique_id = f"homesec_{camera_name}_{description.key}"

    @property
    def is_on(self) -> bool | None:
        key = self.entity_description.key
        camera = self._get_camera_data() or {}
        alert = self._get_latest_alert() or {}

        if key == "online":
            enabled = camera.get("enabled")
            healthy = camera.get("healthy")
            if enabled is None or healthy is None:
                return None
            return bool(enabled and healthy)

        if key == "motion":
            return self._is_motion_active()

        if key == "person":
            detected = alert.get("detected_objects", [])
            if isinstance(detected, list) and "person" in detected:
                return True
            return alert.get("activity_type") == "person"

        return None
