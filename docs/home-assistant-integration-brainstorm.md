# HomeSec + Home Assistant: Deep Integration Brainstorm

## Executive Summary

HomeSec is already well-architected for Home Assistant integration with its plugin system. The key opportunity is to provide a **first-class Home Assistant experience** where users can:

1. **Install** homesec directly from Home Assistant (zero-config for add-on users)
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
- Config merge: multiple YAML files loaded left → right; rightmost wins. Dicts deep-merge, lists with `name` field merge by key.
- Single instance: HA integration assumes one HomeSec instance (`single_config_entry`).
- Secrets: never stored in HomeSec; config stores env var names; HA/add-on passes env vars at boot.
- Repository pattern: API reads and writes go through `ClipRepository` (no direct `StateStore`/`EventStore` access).
- Tests: Given/When/Then comments required for all new tests.
- P0 priority: recording + uploading must keep working even if Postgres is down (API and HA features are best-effort).
- **Real-time events**: Use HA Events API (not MQTT). Add-on gets `SUPERVISOR_TOKEN` automatically; standalone users provide HA URL + token.
- **No MQTT broker required**: Primary path uses HA Events API. Existing MQTT notifier remains available for Node-RED and other MQTT consumers.
- **Last write wins**: No optimistic concurrency in v1 (single HA instance assumption).
- **API during Postgres outage**: Return 503 Service Unavailable.

## Constraints and Non-Goals

- No blocking work in API endpoints; file I/O and long operations must use `asyncio.to_thread`.
- Avoid in-process hot-reload of pipeline components in v1; prefer restart after config changes.
- Do not move heavy inference into Home Assistant (keep compute inside HomeSec runtime).
- Prefer behavioral tests that assert outcomes, not internal state.

---

## Repository Structure

All Home Assistant code lives in the main `homesec` monorepo under `homeassistant/`:

```
homesec/
├── repository.json                 # HA Add-on repo manifest (must be at root)
├── src/homesec/                    # Main Python package (PyPI)
│   ├── api/                        # REST API for HA integration
│   ├── config/                     # Config management
│   └── plugins/notifiers/
│       └── home_assistant.py       # HA Events API notifier
│
├── homeassistant/                  # ALL HA-specific code
│   ├── integration/                # Custom component (HACS)
│   │   ├── hacs.json
│   │   └── custom_components/homesec/
│   │
│   └── addon/                      # Add-on (HA Supervisor)
│       └── homesec/
│           ├── config.yaml
│           ├── Dockerfile
│           └── rootfs/             # s6-overlay services
│
├── tests/
└── docs/
```

**Distribution:**
- **PyPI**: `src/homesec/` published as `homesec` package
- **HACS**: Users point to `homeassistant/integration/`
- **Add-on Store**: Users add `https://github.com/lan17/homesec` (repository.json at root points to `homeassistant/addon/homesec/`)

---

## Features

### Core Features (v1)

| Feature | Description | Priority |
|---------|-------------|----------|
| **Zero-config Add-on Install** | User installs add-on, it works immediately with HA | P0 |
| **Real-time Alerts** | Detection events appear in HA instantly | P0 |
| **Camera Management** | Add/remove/configure cameras via HA UI | P0 |
| **Entity Creation** | Sensors, binary sensors for each camera | P0 |
| **Alert Policy Config** | Set risk thresholds and activity filters per camera | P1 |
| **Clip Browsing** | View recent clips and their analysis results | P1 |
| **Health Monitoring** | Camera online/offline status as entities | P1 |
| **HA Automations** | Trigger automations on detection events | P0 |

### Entities Created Per Camera

| Entity | Type | Description |
|--------|------|-------------|
| `binary_sensor.homesec_{camera}_motion` | Binary Sensor | Motion detected (on for 30s after alert) |
| `binary_sensor.homesec_{camera}_online` | Binary Sensor | Camera connectivity |
| `sensor.homesec_{camera}_last_activity` | Sensor | Last activity type (person, vehicle, package, etc.) |
| `sensor.homesec_{camera}_risk_level` | Sensor | Risk level (LOW/MEDIUM/HIGH/CRITICAL) |
| `sensor.homesec_{camera}_health` | Sensor | healthy/unhealthy |
| `switch.homesec_{camera}_enabled` | Switch | Enable/disable camera processing |

### Hub Entities

| Entity | Type | Description |
|--------|------|-------------|
| `sensor.homesec_hub_status` | Sensor | online/offline |
| `sensor.homesec_clips_today` | Sensor | Number of clips recorded today |
| `sensor.homesec_alerts_today` | Sensor | Number of alerts fired today |

### HA Events Fired

| Event | Trigger | Data |
|-------|---------|------|
| `homesec_alert` | Detection alert | camera, clip_id, activity_type, risk_level, summary, view_url |
**Note**: Camera health is **not** pushed via events. The HA Integration polls the REST API (`GET /api/v1/cameras/{name}/status`) every 30-60s to get health status. This keeps the HomeSec core simple and stateless.

### Future Features (v2+)

- Custom Lovelace card with detection timeline
- Clip playback in HA dashboard
- Per-camera alert schedules (e.g., only alert at night)
- Integration with HA zones (e.g., suppress alerts when home)

---

## User Onboarding Flows

### Flow A: Home Assistant Add-on User (Recommended)

This is the zero-config experience for HA OS / Supervised users.

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Install Add-on                                      │
├─────────────────────────────────────────────────────────────┤
│ User goes to Settings → Add-ons → Add-on Store              │
│ Adds HomeSec repository URL                                 │
│ Clicks "Install" on HomeSec add-on                          │
│                                                              │
│ Result: Add-on container starts with SUPERVISOR_TOKEN       │
│         HA Events API works automatically                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Install Integration                                 │
├─────────────────────────────────────────────────────────────┤
│ User goes to Settings → Devices & Services → Add Integration│
│ Searches for "HomeSec"                                      │
│ Integration auto-discovers add-on at localhost:8080         │
│ User clicks "Submit"                                        │
│                                                              │
│ Result: Integration connects, entities created              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Add First Camera                                    │
├─────────────────────────────────────────────────────────────┤
│ Integration prompts: "No cameras configured. Add one?"      │
│ User enters:                                                │
│   - Camera name: "front_door"                               │
│   - Type: RTSP                                              │
│   - RTSP URL: rtsp://192.168.1.100:554/stream              │
│ Clicks "Add Camera"                                         │
│                                                              │
│ Result: Config saved, add-on restarts, camera starts        │
│         Entities appear: homesec_front_door_*               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Configure Notifications (Optional)                  │
├─────────────────────────────────────────────────────────────┤
│ User creates automation in HA:                              │
│   Trigger: Event "homesec_alert"                           │
│   Condition: risk_level = HIGH or CRITICAL                  │
│   Action: Send notification to mobile app                   │
│                                                              │
│ Result: User gets push notifications on high-risk events    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Done!                                               │
├─────────────────────────────────────────────────────────────┤
│ HomeSec is running and integrated with HA                   │
│ - Cameras recording and analyzing                           │
│ - Entities updating in real-time                            │
│ - Automations firing on detections                          │
│ - Clips uploading to configured storage                     │
└─────────────────────────────────────────────────────────────┘
```

**Time to first alert: ~5 minutes**

### Flow B: Standalone Docker User

For users running HomeSec outside HA (separate server, NAS, etc.).

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Deploy HomeSec                                      │
├─────────────────────────────────────────────────────────────┤
│ User runs HomeSec via Docker on their server                │
│ Configures cameras, storage, VLM in config.yaml             │
│                                                              │
│ (This is existing standalone setup)                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Create HA Long-Lived Token                          │
├─────────────────────────────────────────────────────────────┤
│ User goes to HA → Profile → Long-Lived Access Tokens        │
│ Creates token named "HomeSec"                               │
│ Copies token value                                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Configure HomeSec for HA                            │
├─────────────────────────────────────────────────────────────┤
│ User adds to config.yaml:                                   │
│                                                              │
│   notifiers:                                                │
│     - backend: home_assistant                               │
│       config:                                               │
│         url_env: HA_URL    # http://homeassistant:8123     │
│         token_env: HA_TOKEN # the long-lived token         │
│                                                              │
│ Sets environment variables and restarts HomeSec             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Install Integration in HA                           │
├─────────────────────────────────────────────────────────────┤
│ User installs HomeSec integration via HACS                  │
│ Enters HomeSec URL (e.g., http://nas:8080)                 │
│ Integration connects and creates entities                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Done!                                               │
├─────────────────────────────────────────────────────────────┤
│ Same result as Flow A, but with manual token setup          │
└─────────────────────────────────────────────────────────────┘
```

**Time to first alert: ~15 minutes** (more manual steps)

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Add-on install to first entity | < 2 minutes |
| Add-on install to first alert | < 5 minutes |
| Entity update latency (detection → HA) | < 1 second |
| Config change to active (restart) | < 30 seconds |

---

## Add-on Architecture

The HomeSec add-on is a Docker container managed by HA Supervisor. It bundles PostgreSQL for zero-config deployment.

```
┌─────────────────────────────────────────────────────────────┐
│                    Home Assistant OS                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐   ┌──────────────────────────────────┐   │
│   │  HA Core     │   │       HomeSec Add-on             │   │
│   │  (container) │   │       (container)                │   │
│   │              │   │                                  │   │
│   │  - Frontend  │   │  ┌────────────┐ ┌────────────┐  │   │
│   │  - Automtic  │   │  │ PostgreSQL │ │  HomeSec   │  │   │
│   │  - Integratns│   │  │  (s6 svc)  │ │  (s6 svc)  │  │   │
│   │              │   │  │            │ │            │  │   │
│   │              │   │  │ Port 5432  │ │ Port 8080  │  │   │
│   │              │◄──│──│ (internal) │ │ (ingress)  │  │   │
│   │              │   │  └────────────┘ └────────────┘  │   │
│   └──────────────┘   │                                  │   │
│          ▲           │  /data/postgres/  (DB storage)   │   │
│          │           │  /media/homesec/  (clips)        │   │
│          │           └──────────────────────────────────┘   │
│          │  HA Events API                                   │
│          │  (SUPERVISOR_TOKEN)                              │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              HA Supervisor                           │   │
│   │  - Manages all add-on containers                    │   │
│   │  - Injects SUPERVISOR_TOKEN into HomeSec            │   │
│   │  - Handles restarts, updates, logs                  │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Why Bundle PostgreSQL?

| Approach | User Experience | Recommendation |
|----------|-----------------|----------------|
| Bundled PostgreSQL | Zero-config, just works | **Recommended** |
| Separate PostgreSQL add-on | Two installs, manual config | Not recommended |
| SQLite | Simpler but limited concurrency | Not recommended |

- Uses **s6-overlay** to run PostgreSQL + HomeSec as two services in one container
- PostgreSQL starts first, HomeSec waits for it to be ready
- Data persists in `/data/postgres/` (survives container restarts)
- No port exposure - PostgreSQL only accessible inside container
- Advanced users can still use external PostgreSQL via `database_url` option

### Storage Layout

| Path | Contents | Persistence |
|------|----------|-------------|
| `/data/postgres/` | PostgreSQL database files | Add-on private, persistent |
| `/data/overrides.yaml` | HA-managed config overrides | Add-on private, persistent |
| `/media/homesec/clips/` | Video clips (LocalStorage) | Shared with HA, persistent |
| `/config/homesec/` | Base config file | Shared with HA, persistent |

---

## Integration Architecture Options

### Option A: Add-on + Native Integration (Chosen)

```
┌─────────────────────────────────────────────────────────────┐
│                    Home Assistant OS                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │   HomeSec Add-on    │    │   HomeSec Integration       │ │
│  │   (Docker container)│───►│   (HA Core component)       │ │
│  │                     │    │                             │ │
│  │  - Pipeline service │    │  - Config flow UI           │ │
│  │  - RTSP sources     │    │  - Entity platforms         │ │
│  │  - YOLO/VLM         │    │  - Services                 │ │
│  │  - Storage backends │    │  - Event subscriptions      │ │
│  │  - HA Notifier      │    │                             │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
│           │                            ▲                     │
│           │    SUPERVISOR_TOKEN        │                     │
│           └────────────────────────────┘                     │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Communication Layer                         │ │
│  │  - REST API for config/control (Integration → Add-on)   │ │
│  │  - HA Events API for real-time push (Add-on → HA Core)  │ │
│  │    POST /api/events/homesec_alert (auto-authenticated)  │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Pros:**
- Zero-config for add-on users (SUPERVISOR_TOKEN is automatic)
- No MQTT broker required
- Real-time event push without polling
- Full HA UI configuration via the integration
- Works with HA OS, Supervised, and standalone Docker

**Cons:**
- Two components to maintain
- Standalone users must provide HA URL + long-lived token

---

### Option B: MQTT Discovery (Quick Win) - NOT CHOSEN

> **Note:** This option was not chosen. We went with Option A (Add-on + Native Integration) using the HA Events API for real-time communication.

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
2. Home Assistant notifier plugin (HA Events API)
3. Home Assistant add-on packaging
4. Native HA integration (config flow + entities)
5. Advanced UX (optional)

### Phase 1: REST API for Configuration

Add a new REST API to HomeSec for remote configuration. All endpoints are `async def` and use async SQLAlchemy only.

Config model for Option B:

- Base YAML is bootstrap-only (DB DSN, server config, storage root, MQTT broker, etc.).
- API writes a machine-owned **override YAML** file for all dynamic config.
- HomeSec loads multiple YAML files left → right; rightmost wins.
- Dicts deep-merge; lists with `name` field merge by key.
- Override file default: `config/ha-overrides.yaml` (configurable via CLI).
- CLI accepts multiple `--config` flags; order matters.
- Config writes use last-write-wins semantics (no optimistic concurrency in v1).

```yaml
# New endpoints
GET  /api/v1/config                    # Get current config
PUT  /api/v1/config                    # Update config
GET  /api/v1/cameras                   # List cameras
POST /api/v1/cameras                   # Add camera
PUT  /api/v1/cameras/{name}            # Update camera
DELETE /api/v1/cameras/{name}          # Remove camera
GET  /api/v1/cameras/{name}/status     # Camera status
GET  /api/v1/clips                     # List recent clips
GET  /api/v1/clips/{id}                # Get clip details
POST /api/v1/system/restart            # Request graceful restart
```

Config updates are validated with Pydantic, written to the override YAML, and return `restart_required: true`. HA can then call a restart endpoint or restart the add-on.
Real-time updates use HA Events API (no WebSocket or MQTT required in v1).

### Phase 3: Home Assistant Add-on

Create a Home Assistant add-on in the monorepo:

```
homeassistant/addon/
├── README.md
└── homesec/
    ├── config.yaml           # Add-on manifest
    ├── build.yaml
    ├── Dockerfile            # Includes PostgreSQL 16
    ├── DOCS.md
    ├── CHANGELOG.md
    ├── icon.png              # 512x512
    ├── logo.png              # 256x256
    ├── rootfs/
    │   └── etc/
    │       ├── s6-overlay/   # PostgreSQL + HomeSec services
    │       └── nginx/        # Ingress config
    └── translations/
        └── en.yaml
```

Note: `repository.json` lives at the repo root (not in `homeassistant/addon/`) for HA to discover it.

**Add-on features:**
- Zero-config HA integration via `SUPERVISOR_TOKEN` (automatic)
- Real-time event push to HA without MQTT
- Bundled PostgreSQL via s6-overlay (zero-config database)
- Ingress support for web UI (if we build one)
- Watchdog for auto-restart

Users add the add-on repo via: `https://github.com/lan17/homesec`

### Phase 4: Native Home Assistant Integration

Build a custom integration in the monorepo:

```
homeassistant/integration/
├── hacs.json
└── custom_components/
    └── homesec/
        ├── __init__.py           # Setup, config entry
        ├── manifest.json         # Integration metadata
        ├── config_flow.py        # UI-based configuration (auto-detects add-on)
        ├── const.py              # Constants
        ├── coordinator.py        # DataUpdateCoordinator + event subscriptions
        ├── entity.py             # Base entity class
        ├── sensor.py             # Sensor entities
        ├── binary_sensor.py      # Motion sensors (30s auto-reset timer)
        ├── switch.py             # Enable/disable cameras (stops RTSP)
        ├── strings.json          # UI strings
        └── translations/
            └── en.json
```

**Entity Platforms:**

| Platform | Entities | Description | v1 Scope |
|----------|----------|-------------|----------|
| `binary_sensor` | `motion` | Motion detected (30s auto-reset) | ✓ |
| `sensor` | `last_activity`, `risk_level`, `health` | Detection metadata | ✓ |
| `switch` | `camera_enabled` | Enable/disable camera | ✓ |
| `image` | Per-camera | Last snapshot | v2 |
| `select` | `alert_sensitivity` | LOW/MEDIUM/HIGH | v2 |

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
- **Merge semantics**: dicts deep-merge; lists with `name` field merge by key (e.g., cameras merge by camera name).
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
  "dependencies": [],
  "codeowners": ["@lan17"],
  "single_config_entry": true,
  "iot_class": "local_push",
  "integration_type": "hub"
}
```

---

## HomeAssistant Notifier Plugin

New notifier plugin that pushes events directly to Home Assistant:

```python
class HomeAssistantNotifierConfig(BaseModel):
    """Configuration for HA notifier."""
    # For standalone mode (not running as add-on)
    url_env: str | None = None      # e.g., "HA_URL" -> http://homeassistant.local:8123
    token_env: str | None = None    # e.g., "HA_TOKEN" -> long-lived access token

class HomeAssistantNotifier(Notifier):
    """Push events directly to Home Assistant via Events API."""

    async def notify(self, alert: Alert) -> None:
        # When running as add-on, use supervisor API (zero-config)
        if os.environ.get("SUPERVISOR_TOKEN"):
            url = "http://supervisor/core/api/events/homesec_alert"
            headers = {"Authorization": f"Bearer {os.environ['SUPERVISOR_TOKEN']}"}
        else:
            # Standalone mode - use configured URL/token
            url = f"{self._url}/api/events/homesec_alert"
            headers = {"Authorization": f"Bearer {self._token}"}

        await self._session.post(url, json=alert.to_ha_event(), headers=headers)
```

**Event types fired:**
- `homesec_alert` - Detection alert with full metadata

The HA integration subscribes to this event and updates entities in real-time.

**Note**: Camera health is polled via REST API, not pushed via events.

---

## Summary: Recommended Roadmap

| Phase | Effort | Value | Description |
|-------|--------|-------|-------------|
| **1. REST API** | Medium | High | Enable remote configuration and control plane |
| **2. HA Notifier** | Low | High | New notifier plugin for HA Events API |
| **3. Add-on** | Medium | High | Zero-config install for HA OS users |
| **4. Integration** | High | Very High | Full HA UI configuration + event subscriptions |
| **5. Dashboard Cards** | Medium | Medium | Rich visualization |

---

## References

- [Home Assistant Developer Docs - Creating Integrations](https://developers.home-assistant.io/docs/creating_component_index/)
- [Home Assistant Config Flow](https://developers.home-assistant.io/docs/config_entries_config_flow_handler/)
- [Home Assistant Camera Entity](https://developers.home-assistant.io/docs/core/entity/camera/)
- [Home Assistant DataUpdateCoordinator](https://developers.home-assistant.io/docs/integration_fetching_data/)
- [HACS Documentation](https://www.hacs.xyz/)
