"""Switch platform for HomeSec."""

from __future__ import annotations

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_CAMERAS, DOMAIN
from .coordinator import HomesecCoordinator
from .entity import HomesecCameraEntity


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up HomeSec switches from a config entry."""
    coordinator: HomesecCoordinator = hass.data[DOMAIN][entry.entry_id]

    camera_names = entry.data.get(CONF_CAMERAS, [])
    entities = [HomesecCameraEnabledSwitch(coordinator, name) for name in camera_names]
    async_add_entities(entities)


class HomesecCameraEnabledSwitch(HomesecCameraEntity, SwitchEntity):
    """Enable/disable a HomeSec camera."""

    _attr_name = "Enabled"

    def __init__(self, coordinator: HomesecCoordinator, camera_name: str) -> None:
        super().__init__(coordinator, camera_name)
        self._attr_unique_id = f"homesec_{camera_name}_enabled"

    @property
    def is_on(self) -> bool | None:
        camera = self._get_camera_data() or {}
        enabled = camera.get("enabled")
        if enabled is None:
            return None
        return bool(enabled)

    async def async_turn_on(self, **_: object) -> None:
        await self.coordinator.async_set_camera_enabled(self._camera_name, True)
        await self.coordinator.async_refresh()

    async def async_turn_off(self, **_: object) -> None:
        await self.coordinator.async_set_camera_enabled(self._camera_name, False)
        await self.coordinator.async_refresh()
