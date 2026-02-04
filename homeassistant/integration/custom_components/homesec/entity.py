"""Entity base classes for HomeSec."""

from __future__ import annotations

from typing import Any

from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import HomesecCoordinator


class HomesecHubEntity(CoordinatorEntity[HomesecCoordinator]):
    """Base class for HomeSec hub entities."""

    _attr_has_entity_name = True

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, "homesec_hub")},
            name="HomeSec Hub",
            manufacturer="HomeSec",
            model="AI Security Hub",
        )


class HomesecCameraEntity(CoordinatorEntity[HomesecCoordinator]):
    """Base class for HomeSec camera entities."""

    _attr_has_entity_name = True

    def __init__(self, coordinator: HomesecCoordinator, camera_name: str) -> None:
        super().__init__(coordinator)
        self._camera_name = camera_name

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, self._camera_name)},
            name=f"HomeSec {self._camera_name}",
            manufacturer="HomeSec",
            model="Camera",
            via_device=(DOMAIN, "homesec_hub"),
        )

    def _get_camera_data(self) -> dict[str, Any] | None:
        cameras = self.coordinator.data.get("cameras", [])
        for camera in cameras:
            if camera.get("name") == self._camera_name:
                return camera
        return None

    def _is_motion_active(self) -> bool:
        return bool(self.coordinator.data.get("motion_active", {}).get(self._camera_name))

    def _get_latest_alert(self) -> dict[str, Any] | None:
        return self.coordinator.data.get("latest_alerts", {}).get(self._camera_name)
