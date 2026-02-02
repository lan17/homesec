# HomeSec + Home Assistant: Deep Integration Brainstorm

## Executive Summary

HomeSec is already well-architected for Home Assistant integration with its plugin system and existing MQTT notifier. The key opportunity is to go from "basic MQTT alerts" to a **first-class Home Assistant experience** where users can:

1. **Install** homesec directly from Home Assistant
2. **Configure** cameras, alerts, and AI settings through the HA UI
3. **Monitor** camera health, detection stats, and pipeline status as HA entities
4. **Automate** based on detection events with rich metadata
5. **View** clips directly in HA dashboards (RTSP handled by HA)

---

## Decision Snapshot (2026-02-01)

- Chosen direction: Add-on + native integration (Option A below) with HomeSec as the runtime.
- Required: runtime add/remove cameras and other config changes from HA.
- API stack: FastAPI, async endpoints only, async SQLAlchemy only.
- Restart is acceptable: API writes validated config to disk and returns `restart_required`; HA may trigger restart.
- Config storage: **Override YAML file** is source of truth for dynamic config. Base YAML is bootstrap-only.
- Config merge: multiple YAML files loaded left → right; rightmost wins. Dicts deep-merge, lists replace.
- Single instance: HA integration assumes one HomeSec instance (`single_config_entry`).
- Secrets: never stored in HomeSec; config stores env var names; HA/add-on passes env vars at boot.
- Repository pattern: API reads and writes go through `ClipRepository` (no direct `StateStore`/`EventStore` access).
- Tests: Given/When/Then comments required for all new tests.
- P0 priority: recording + uploading must keep working even if Postgres is down (API and HA features are best-effort).

## Constraints and Non-Goals

- No blocking work in API endpoints; file I/O and long operations must use `asyncio.to_thread`.
- Avoid in-process hot-reload of pipeline components in v1; prefer restart after config changes.
- Do not move heavy inference into Home Assistant (keep compute inside HomeSec runtime).
- Prefer behavioral tests that assert outcomes, not internal state.
- Prefer behavioral tests; avoid internal state assertions.

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
│  │  - Storage backends │    │  - MQTT subscriptions       │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
│           │                            │                     │
│           └────────┬───────────────────┘                     │
│                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Communication Layer                         │ │
│  │  - REST API (new) for config/control                    │ │
│  │  - MQTT for events + state topics                       │ │
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

Add a new REST API to HomeSec for remote configuration. All endpoints are `async def` and use async SQLAlchemy only.

Config model for Option B:

- Base YAML is bootstrap-only (DB DSN, server config, storage root, MQTT broker, etc.).
- API writes a machine-owned **override YAML** file for all dynamic config.
- HomeSec loads multiple YAML files left → right; rightmost wins.
- Dicts deep-merge; lists replace (override lists fully replace base lists).
- Override file default: `config/ha-overrides.yaml` (configurable via CLI).
- CLI accepts multiple `--config` flags; order matters.
- All config writes require `config_version` for optimistic concurrency.

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
POST /api/v1/system/restart            # Request graceful restart
```

Config updates are validated with Pydantic, written to the override YAML, and return `restart_required: true`. HA can then call a restart endpoint or restart the add-on.
Real-time updates use MQTT topics (no WebSocket in v1).

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
map:
  - addon_config:rw  # Store HomeSec config/overrides
  - media:rw         # Store clips
services:
  - mqtt:need       # Requires MQTT broker
options:
  config_file: /config/homesec/config.yaml
  override_file: /data/overrides.yaml
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
| `image` | Per-camera | Last snapshot (optional) |
| `binary_sensor` | `motion`, `person_detected` | Detection states |
| `sensor` | `last_activity`, `risk_level`, `clip_count` | Detection metadata |
| `switch` | `camera_enabled`, `alerts_enabled` | Per-camera toggles |
| `select` | `alert_sensitivity` | LOW/MEDIUM/HIGH |
| `device_tracker` | `camera_online` | Connectivity status (optional) |

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

HomeSec will not proxy RTSP streams. HA should use its own camera integration for RTSP
while HomeSec ingests the same stream for analysis.

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

### Adopted Strategy (Override YAML)

- **Base YAML**: bootstrap-only (DB DSN, server config, storage root, MQTT broker, etc.).
- **Override YAML**: machine-owned, fully managed by HA via API.
- **Load order**: multiple YAML files loaded left → right; rightmost wins.
- **Merge semantics**: dicts deep-merge; lists replace (override lists fully replace base lists).
- **Restart**: required for all config changes.

---

## HACS Distribution

For custom integration distribution:

```json
// hacs.json
{
  "name": "HomeSec",
  "render_readme": true,
  "domains": ["sensor", "binary_sensor", "switch", "select", "image", "device_tracker"],
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
  "single_config_entry": true,
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
