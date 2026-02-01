# HomeSec + Home Assistant: Deep Integration Brainstorm

## Executive Summary

HomeSec is already well-architected for Home Assistant integration with its plugin system and existing MQTT notifier. The key opportunity is to go from "basic MQTT alerts" to a **first-class Home Assistant experience** where users can:

1. **Install** homesec directly from Home Assistant
2. **Configure** cameras, alerts, and AI settings through the HA UI
3. **Monitor** camera health, detection stats, and pipeline status as HA entities
4. **Automate** based on detection events with rich metadata
5. **View** clips and live streams directly in HA dashboards

---

## Decision Snapshot (2026-02-01)

- Chosen direction: Add-on + native integration (Option A below) with HomeSec as the runtime.
- Required: runtime add/remove cameras and other config changes from HA.
- API stack: FastAPI, async endpoints only, async SQLAlchemy only.
- Restart is acceptable: API writes validated config to disk and returns `restart_required`; HA may trigger restart.
- Repository pattern: API reads and writes go through `ClipRepository` (no direct `StateStore`/`EventStore` access).
- Tests: Given/When/Then comments required for all new tests.
- P0 priority: recording + uploading must keep working even if Postgres is down (API and HA features are best-effort).

## Constraints and Non-Goals

- No blocking work in API endpoints; file I/O and long operations must use `asyncio.to_thread`.
- Avoid in-process hot-reload of pipeline components in v1; prefer restart after config changes.
- Do not move heavy inference into Home Assistant (keep compute inside HomeSec runtime).

---

## Integration Architecture Options

### Option A: Add-on + Native Integration (Chosen)

```
┌─────────────────────────────────────────────────────────────┐
│                    Home Assistant OS                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │   HomeSec Add-on    │    │   HomeSec Integration       │ │
│  │   (Docker container)│◄──►│   (HA Core component)       │ │
│  │                     │    │                             │ │
│  │  - Pipeline service │    │  - Config flow UI           │ │
│  │  - RTSP sources     │    │  - Entity platforms         │ │
│  │  - YOLO/VLM         │    │  - Services                 │ │
│  │  - Storage backends │    │  - Event subscriptions      │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
│           │                            │                     │
│           └────────┬───────────────────┘                     │
│                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Communication Layer                         │ │
│  │  - REST API (new) for config/control                    │ │
│  │  - WebSocket (new) for real-time events                 │ │
│  │  - MQTT for alerts (existing)                           │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Pros:**
- Best of both worlds: heavy processing isolated in add-on
- Full HA UI configuration via the integration
- Works with HA OS, Supervised, and standalone Docker
- Clean separation of concerns

**Cons:**
- Two components to maintain
- Requires API layer between them

---

### Option B: MQTT Discovery (Quick Win)

Enhance the existing MQTT notifier to publish **discovery messages** that auto-create HA entities:

```
homeassistant/sensor/homesec/{camera_name}_status/config
homeassistant/sensor/homesec/{camera_name}_last_detection/config
homeassistant/binary_sensor/homesec/{camera_name}_motion/config
homeassistant/device_trigger/homesec/{camera_name}_alert/config
```

**Pros:**
- Minimal changes to homesec
- Works immediately with any HA installation
- No custom integration needed

**Cons:**
- Limited configuration from HA (still need YAML)
- No deep UI integration

---

### Option C: Pure Native Integration

Move core homesec logic into a Home Assistant integration (runs in HA's Python environment).

**Pros:**
- Single installation point
- Deep HA integration

**Cons:**
- Heavy processing (YOLO, VLM) in HA's event loop could cause issues
- Harder to use outside HA
- Memory/CPU constraints

---

## Recommended Implementation Plan

Execution order for Option A:

1. REST API + config persistence (control plane)
2. Native HA integration (config flow + entities)
3. Home Assistant add-on packaging
4. MQTT discovery (optional parallel track)
5. Advanced UX (optional)

### Phase 1: MQTT Discovery Enhancement (Quick Win)

Enhance the existing MQTT notifier to publish discovery configs:

```python
# New entities auto-created in HA:
- binary_sensor.homesec_{camera}_motion     # Motion detected
- sensor.homesec_{camera}_last_activity     # "person", "delivery", etc.
- sensor.homesec_{camera}_risk_level        # LOW/MEDIUM/HIGH/CRITICAL
- sensor.homesec_{camera}_health            # healthy/degraded/unhealthy
- device_tracker.homesec_{camera}           # Online/offline status
```

**Changes to homesec:**
1. Add `mqtt_discovery: true` config option
2. On startup, publish discovery configs for each camera
3. Listen for HA birth message to republish discovery
4. Publish state updates on detection events

### Phase 2: REST API for Configuration

Add a new REST API to homesec for remote configuration. All endpoints are `async def` and use async SQLAlchemy only.

```yaml
# New endpoints
GET  /api/v1/config                    # Get current config
PUT  /api/v1/config                    # Update config
GET  /api/v1/cameras                   # List cameras
POST /api/v1/cameras                   # Add camera
PUT  /api/v1/cameras/{name}            # Update camera
DELETE /api/v1/cameras/{name}          # Remove camera
GET  /api/v1/cameras/{name}/status     # Camera status
POST /api/v1/cameras/{name}/test       # Test camera connection
GET  /api/v1/clips                     # List recent clips
GET  /api/v1/clips/{id}                # Get clip details
GET  /api/v1/events                    # Event history
WS   /api/v1/ws                        # Real-time events
```

Config updates are validated with Pydantic, written to disk, and return `restart_required: true`. HA can then call a restart endpoint or restart the add-on.

### Phase 3: Home Assistant Add-on

Create a Home Assistant add-on for easy installation:

```yaml
# config.yaml (add-on manifest)
name: HomeSec
description: Self-hosted AI video security
version: 1.2.2
slug: homesec
arch: [amd64, aarch64]
ports:
  8080/tcp: 8080   # API/Health
  8554/tcp: 8554   # RTSP proxy (optional)
map:
  - config:rw       # Store config
  - media:rw        # Store clips
services:
  - mqtt:need       # Requires MQTT broker
options:
  config_file: /config/homesec/config.yaml
```

**Add-on features:**
- Bundled PostgreSQL (or use HA's)
- Auto-configure MQTT from HA's broker
- Ingress support for web UI (if we build one)
- Watchdog for auto-restart

### Phase 4: Native Home Assistant Integration

Build a custom integration (`custom_components/homesec/`):

```
custom_components/homesec/
├── __init__.py           # Setup, config entry
├── manifest.json         # Integration metadata
├── config_flow.py        # UI-based configuration
├── const.py              # Constants
├── coordinator.py        # DataUpdateCoordinator
├── camera.py             # Camera entity platform
├── sensor.py             # Sensor entities
├── binary_sensor.py      # Motion sensors
├── switch.py             # Enable/disable cameras
├── services.yaml         # Service definitions
├── strings.json          # UI strings
└── translations/
    └── en.json
```

**Entity Platforms:**

| Platform | Entities | Description |
|----------|----------|-------------|
| `camera` | Per-camera | Live RTSP stream proxy |
| `binary_sensor` | `motion`, `person_detected` | Detection states |
| `sensor` | `last_activity`, `risk_level`, `clip_count` | Detection metadata |
| `switch` | `camera_enabled`, `alerts_enabled` | Per-camera toggles |
| `select` | `alert_sensitivity` | LOW/MEDIUM/HIGH |
| `image` | `last_snapshot` | Most recent detection frame |

**Services:**

```yaml
homesec.add_camera:
  description: Add a new camera
  fields:
    name: Camera identifier
    rtsp_url: RTSP stream URL

homesec.set_alert_policy:
  description: Configure alert policy for camera
  fields:
    camera: Camera name
    min_risk_level: Minimum risk level to alert
    activity_types: List of activity types to alert on

homesec.process_clip:
  description: Manually process a video clip
  fields:
    file_path: Path to video file
```

**Config Flow:**

```
Step 1: Connection
├── HomeSec URL: http://localhost:8080
└── [Test Connection]

Step 2: Camera Setup
├── Discovered cameras: [front_door, backyard, ...]
├── [x] front_door - Configure →
│       └── Alert sensitivity: [Medium ▼]
│       └── Notify on: [x] Person [x] Vehicle [ ] Animal
└── [+] Add new camera

Step 3: Notifications
├── [x] Create HA notifications for alerts
├── [x] Add events to logbook
└── Device tracker for camera health
```

---

## Deep Integration Features

### 1. Camera Streams in HA

Proxy RTSP streams through homesec with authentication:

```python
class HomesecCamera(Camera):
    """Representation of a HomeSec camera."""

    _attr_supported_features = CameraEntityFeature.STREAM

    async def stream_source(self) -> str:
        """Return the stream source URL."""
        return f"rtsp://{self._host}:8554/{self._camera_name}"
```

### 2. Rich Event Data for Automations

```yaml
# Example automation using homesec events
automation:
  - alias: "Alert on suspicious person at night"
    trigger:
      - platform: state
        entity_id: sensor.homesec_front_door_last_activity
        to: "person"
    condition:
      - condition: state
        entity_id: sensor.homesec_front_door_risk_level
        state: "HIGH"
      - condition: sun
        after: sunset
    action:
      - service: notify.mobile_app
        data:
          title: "Security Alert"
          message: "{{ state_attr('sensor.homesec_front_door_last_activity', 'summary') }}"
          data:
            image: "{{ state_attr('sensor.homesec_front_door_last_activity', 'snapshot_url') }}"
            actions:
              - action: VIEW_CLIP
                title: "View Clip"
                uri: "{{ state_attr('sensor.homesec_front_door_last_activity', 'clip_url') }}"
```

### 3. Dashboard Cards

Create custom Lovelace cards:

```yaml
type: custom:homesec-camera-card
entity: camera.homesec_front_door
show_detections: true
show_timeline: true
detection_overlay: true  # Show bounding boxes on stream
```

### 4. Device Registry Integration

Group all entities under device:

```python
device_info = DeviceInfo(
    identifiers={(DOMAIN, camera_name)},
    name=f"HomeSec {camera_name}",
    manufacturer="HomeSec",
    model="AI Security Camera",
    sw_version="1.2.2",
    via_device=(DOMAIN, "homesec_hub"),
)
```

### 5. Diagnostics

Provide diagnostic data for troubleshooting:

```python
async def async_get_config_entry_diagnostics(hass, entry):
    return {
        "config": {**entry.data, "api_key": "REDACTED"},
        "cameras": [...],
        "health": await coordinator.api.get_health(),
        "recent_events": [...],
    }
```

---

## Configuration Sync Strategy

### Option A: HA as Source of Truth

HA integration manages config, pushes to homesec:

```
User edits in HA UI → Integration → REST API → HomeSec
                                              ↓
                                         Restarts with new config
```

### Option B: HomeSec as Source of Truth

HomeSec config is canonical, HA reads it:

```
User edits YAML → HomeSec → MQTT Discovery → HA entities created
                         → REST API → Integration reads state
```

### Option C: Hybrid (Recommended)

- **Core config** (storage, database, VLM provider): HomeSec YAML
- **Camera config**: Editable from HA via API, persisted to YAML; restart required
- **Alert policies**: Editable from HA, stored in homesec; restart required

---

## HACS Distribution

For custom integration distribution:

```json
// hacs.json
{
  "name": "HomeSec",
  "render_readme": true,
  "domains": ["camera", "sensor", "binary_sensor", "switch"],
  "iot_class": "local_push"
}
```

```json
// manifest.json
{
  "domain": "homesec",
  "name": "HomeSec",
  "version": "1.0.0",
  "documentation": "https://github.com/lan17/homesec",
  "dependencies": ["mqtt"],
  "codeowners": ["@lan17"],
  "iot_class": "local_push",
  "integration_type": "hub"
}
```

---

## Summary: Recommended Roadmap

| Phase | Effort | Value | Description |
|-------|--------|-------|-------------|
| **1. REST API** | Medium | High | Enable remote configuration and control plane |
| **2. Integration** | High | Very High | Full HA UI configuration |
| **3. Add-on** | Medium | High | One-click install for HA OS users |
| **4. MQTT Discovery (optional)** | Low | Medium | Auto-create entities for non-integration users |
| **5. Dashboard Cards** | Medium | Medium | Rich visualization |

---

## References

- [Home Assistant Developer Docs - Creating Integrations](https://developers.home-assistant.io/docs/creating_component_index/)
- [Home Assistant Config Flow](https://developers.home-assistant.io/docs/config_entries_config_flow_handler/)
- [Home Assistant Camera Entity](https://developers.home-assistant.io/docs/core/entity/camera/)
- [Home Assistant DataUpdateCoordinator](https://developers.home-assistant.io/docs/integration_fetching_data/)
- [MQTT Discovery](https://www.home-assistant.io/integrations/mqtt/)
- [HACS Documentation](https://www.hacs.xyz/)
