# Phase 4: Native Home Assistant Integration

**Goal**: Full UI-based configuration and deep entity integration in Home Assistant.

**Estimated Effort**: 7-10 days

**Dependencies**: Phase 1 (REST API), Phase 2 (HA Notifier)

---

## Overview

This phase creates a custom Home Assistant integration that:
- Auto-detects the HomeSec add-on
- Creates entities for cameras (sensors, binary sensors, switches)
- Creates hub-level entities (system health, alerts today, clips today)
- Subscribes to HomeSec events for real-time updates

---

## 4.1 Integration Structure

**Directory**: `homeassistant/integration/custom_components/homesec/`

```
homeassistant/integration/custom_components/homesec/
├── __init__.py           # Setup and entry points
├── manifest.json         # Integration metadata
├── const.py              # Constants
├── config_flow.py        # UI configuration flow
├── coordinator.py        # Data update coordinator
├── entity.py             # Base entity classes
├── sensor.py             # Sensor platform
├── binary_sensor.py      # Binary sensor platform
├── switch.py             # Switch platform
├── diagnostics.py        # Diagnostic data
├── strings.json          # UI strings
└── translations/
    └── en.json           # English translations
```

---

## 4.2 Manifest

**File**: `manifest.json`

```json
{
  "domain": "homesec",
  "name": "HomeSec",
  "codeowners": ["@lan17"],
  "config_flow": true,
  "dependencies": [],
  "single_config_entry": true,
  "documentation": "https://github.com/lan17/homesec",
  "integration_type": "hub",
  "iot_class": "local_push",
  "issue_tracker": "https://github.com/lan17/homesec/issues",
  "requirements": ["aiohttp>=3.8.0"],
  "version": "1.0.0"
}
```

### Constraints

- `single_config_entry: true` - only one HomeSec instance per HA
- `iot_class: local_push` - we use HA Events for real-time updates
- `integration_type: hub` - we create sub-devices for cameras

---

## 4.3 Constants

**File**: `const.py`

```python
DOMAIN = "homesec"

# Config keys
CONF_HOST = "host"
CONF_PORT = "port"
CONF_API_KEY = "api_key"
CONF_VERIFY_SSL = "verify_ssl"

# Defaults
DEFAULT_PORT = 8080
DEFAULT_VERIFY_SSL = True
ADDON_HOSTNAME = "localhost"

# Motion sensor
DEFAULT_MOTION_RESET_SECONDS = 30

# Platforms
PLATFORMS = ["binary_sensor", "sensor", "switch"]

# Update intervals
SCAN_INTERVAL_SECONDS = 30
```

---

## 4.4 Config Flow

**File**: `config_flow.py`

### Interface

```python
class HomesecConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for HomeSec."""

    async def async_step_user(self, user_input=None) -> FlowResult:
        """Initial step - check for add-on first, then manual."""
        ...

    async def async_step_addon(self, user_input=None) -> FlowResult:
        """Handle add-on auto-discovery confirmation."""
        ...

    async def async_step_manual(self, user_input=None) -> FlowResult:
        """Handle manual setup for standalone HomeSec."""
        ...

    async def async_step_cameras(self, user_input=None) -> FlowResult:
        """Handle camera selection step."""
        ...


class OptionsFlowHandler(OptionsFlow):
    """Handle options flow for HomeSec."""

    async def async_step_init(self, user_input=None) -> FlowResult:
        """Manage options: scan_interval, motion_reset_seconds."""
        ...


async def validate_connection(hass, host, port, api_key=None, verify_ssl=True) -> dict:
    """Validate connection to HomeSec API.

    Returns: {"title": str, "version": str, "cameras": list}
    Raises: CannotConnect, InvalidAuth
    """
    ...

async def detect_addon(hass) -> bool:
    """Check if HomeSec add-on is running at localhost:8080."""
    ...
```

### Flow Steps

1. **user**: Check for add-on, redirect to addon or manual
2. **addon**: Confirm add-on connection (one-click setup)
3. **manual**: Enter host, port, API key
4. **cameras**: Select which cameras to add to HA

### Constraints

- Store `addon: bool` in config entry data to track mode
- Unique ID: `homesec_{host}_{port}`
- Options: scan_interval, motion_reset_seconds

---

## 4.5 Data Coordinator

**File**: `coordinator.py`

### Interface

```python
class HomesecCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator to manage HomeSec data updates."""

    def __init__(self, hass, config_entry, host, port, api_key=None, verify_ssl=True):
        ...

    @property
    def base_url(self) -> str: ...

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch data from HomeSec API.

        Returns:
            {
                "health": {...},
                "cameras": [...],
                "stats": {...},
                "connected": bool,
                "motion_active": {camera: bool},      # Event-driven
                "latest_alerts": {camera: {...}},     # Event-driven (for last_activity sensor)
            }
        """
        ...

    async def async_subscribe_events(self) -> None:
        """Subscribe to HomeSec events fired via HA Events API.

        Event: homesec_alert

        Note: Camera health is polled via REST API, not event-driven.
        """
        ...

    async def async_unsubscribe_events(self) -> None:
        """Unsubscribe from HomeSec events."""
        ...

    # API Methods
    async def async_add_camera(self, name, source_backend, source_config) -> dict: ...
    async def async_update_camera(self, camera_name, source_config=None) -> dict: ...
    async def async_delete_camera(self, camera_name) -> None: ...
    async def async_set_camera_enabled(self, camera_name, enabled: bool) -> dict: ...
```

### Motion Timer Logic

When `homesec_alert` event received:
1. Set `motion_active[camera] = True`
2. Cancel any existing reset timer for this camera
3. Schedule new timer for `motion_reset_seconds` (default 30)
4. When timer fires, set `motion_active[camera] = False`

### Constraints

- Preserve `motion_active`, `latest_alerts` across polling updates
- Handle 503 gracefully (Postgres unavailable)

---

## 4.6 Entity Base Classes

**File**: `entity.py`

### Interface

```python
class HomesecHubEntity(CoordinatorEntity[HomesecCoordinator]):
    """Base class for HomeSec Hub entities (system-wide sensors)."""

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info for the HomeSec Hub."""
        # identifiers: (DOMAIN, "homesec_hub")
        # name: "HomeSec Hub"
        # model: "AI Security Hub"
        ...


class HomesecCameraEntity(CoordinatorEntity[HomesecCoordinator]):
    """Base class for HomeSec camera entities."""

    def __init__(self, coordinator, camera_name): ...

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info for this camera."""
        # identifiers: (DOMAIN, camera_name)
        # name: f"HomeSec {camera_name}"
        # via_device: (DOMAIN, "homesec_hub")
        ...

    def _get_camera_data(self) -> dict | None:
        """Get camera data from coordinator."""
        ...

    def _is_motion_active(self) -> bool:
        """Check if motion is currently active for this camera."""
        ...

    def _get_latest_alert(self) -> dict | None:
        """Get the latest alert for this camera."""
        ...
```

### Device Hierarchy

```
HomeSec Hub (identifiers: homesec_hub)
├── HomeSec front_door (identifiers: front_door, via_device: homesec_hub)
├── HomeSec back_yard (identifiers: back_yard, via_device: homesec_hub)
└── ...
```

---

## 4.7 Entity Platforms

### Sensors (sensor.py)

**Hub Sensors** (HomesecHubEntity):
- `cameras_online` - Count of healthy cameras
- `alerts_today` - Alerts count today
- `clips_today` - Clips count today
- `system_health` - "healthy" / "degraded" / "unhealthy"

**Camera Sensors** (HomesecCameraEntity):
- `last_activity` - Activity type from latest alert
- `risk_level` - Risk level from latest alert
- `health` - "healthy" / "unhealthy"

### Binary Sensors (binary_sensor.py)

**Camera Binary Sensors** (HomesecCameraEntity):
- `motion` - Motion detected (auto-resets after timeout)
- `person` - Person detected (based on latest alert activity_type)
- `online` - Camera connectivity

### Switches (switch.py)

**Camera Switches** (HomesecCameraEntity):
- `enabled` - Enable/disable camera (calls API, requires restart)

---

## 4.8 Translations

**File**: `translations/en.json`

Provides translations for:
- Config flow steps (user, addon, manual, cameras)
- Error messages (cannot_connect, invalid_auth)
- Options flow
- Entity names

---

## File Changes Summary

| File | Change |
|------|--------|
| `homeassistant/integration/hacs.json` | HACS configuration |
| `custom_components/homesec/__init__.py` | Setup entry |
| `custom_components/homesec/manifest.json` | Integration metadata |
| `custom_components/homesec/const.py` | Constants |
| `custom_components/homesec/config_flow.py` | Config + options flow |
| `custom_components/homesec/coordinator.py` | Data coordinator |
| `custom_components/homesec/entity.py` | Base entity classes |
| `custom_components/homesec/sensor.py` | Sensor platform |
| `custom_components/homesec/binary_sensor.py` | Binary sensor platform |
| `custom_components/homesec/switch.py` | Switch platform |
| `custom_components/homesec/diagnostics.py` | Diagnostic data |
| `custom_components/homesec/strings.json` | UI strings |
| `custom_components/homesec/translations/en.json` | English translations |

---

## Test Expectations

### Fixtures Needed

- `mock_homesec_api` - Mocked aiohttp responses for HomeSec API
- `mock_coordinator` - HomesecCoordinator with canned data
- `sample_camera_data` - Camera response from API
- `sample_alert_event` - homesec_alert event data

### Test Cases

**Config Flow**
- Given add-on running at localhost:8080, when start config flow, then addon step shown
- Given no add-on, when start config flow, then manual step shown
- Given invalid API key, when submit manual step, then invalid_auth error
- Given valid connection, when complete flow, then config entry created

**Coordinator**
- Given healthy API, when update, then data contains health, cameras, stats
- Given 503 from API, when update, then UpdateFailed raised with message
- Given homesec_alert event, when received, then motion_active set and timer scheduled
- Given motion timer expires, when callback runs, then motion_active cleared

**Entities**
- Given camera "front" online, when check binary_sensor.front_online, then is_on=True
- Given alert for "front", when check sensor.front_last_activity, then shows activity_type
- Given switch turned off, when toggle, then API called with enabled=False

---

## Verification

```bash
# Copy integration to HA config
cp -r homeassistant/integration/custom_components/homesec ~/.homeassistant/custom_components/

# Restart HA and check logs
# Add integration via UI: Settings > Devices & Services > Add Integration > HomeSec

# Verify entities created
# Check Developer Tools > States for homesec.* entities

# Test motion sensor
# Trigger alert and verify binary_sensor.{camera}_motion turns on then off
```

---

## Definition of Done

- [ ] Config flow auto-detects add-on and offers one-click setup
- [ ] Config flow falls back to manual for standalone
- [ ] All entity platforms create entities correctly
- [ ] Hub device created with system-wide sensors
- [ ] Camera devices created with via_device to hub
- [ ] DataUpdateCoordinator fetches at correct intervals
- [ ] HA Events subscription triggers refresh on alerts
- [ ] Motion sensor auto-resets after configurable timeout
- [ ] Camera switch enables/disables via API
- [ ] Options flow allows reconfiguration
- [ ] All strings translatable
- [ ] HACS installation works
- [ ] Error handling shows user-friendly messages
