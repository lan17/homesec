# HomeSec + Home Assistant: Detailed Implementation Plan

This document provides a comprehensive, step-by-step implementation plan for integrating HomeSec with Home Assistant. Each phase includes specific file changes, code examples, testing strategies, and acceptance criteria.

---

## Decision Snapshot (2026-02-01)

- Chosen approach: Add-on + native integration with HomeSec as the runtime.
- Required: runtime add/remove cameras and other config changes from HA.
- API stack: FastAPI, async endpoints only, async SQLAlchemy only.
- Restart is acceptable: API writes validated config to disk and returns `restart_required`; HA can trigger restart.
- Config storage: **Override YAML file** is source of truth for dynamic config. Base YAML is bootstrap-only.
- Config merge: multiple YAML files loaded left → right; rightmost wins. Dicts deep-merge, lists replace.
- Single instance: HA integration assumes one HomeSec instance (`single_config_entry`).
- Secrets: never stored in HomeSec; config stores env var names; HA/add-on passes env vars at boot.
- Repository pattern: API reads and writes go through `ClipRepository` (no direct `StateStore`/`EventStore` access).
- Tests: Given/When/Then comments required for all new tests.
- P0 priority: recording + uploading must keep working even if Postgres is down (API and HA features are best-effort).
- **Real-time events**: Use HA Events API (not MQTT). Add-on gets `SUPERVISOR_TOKEN` automatically; standalone users provide HA URL + token.
- **No MQTT required**: MQTT Discovery is optional for users who prefer it; primary path uses HA Events API.
- **409 Conflict UX**: Show error to user when config version is stale.
- **API during Postgres outage**: Return 503 Service Unavailable.
- **Camera ping**: RTSP ping implementation should include TODO for real connectivity test (not part of this integration work).
- **Delete clip**: Deletes from both local storage and cloud storage (existing pattern in cleanup_clips.py).

## Prerequisites (Changes to Core HomeSec)

Before implementing the HA integration, these changes are needed in the core HomeSec codebase:

1. **Add `enabled` field to CameraConfig** (`src/homesec/models/config.py`):
   ```python
   class CameraConfig(BaseModel):
       name: str
       enabled: bool = True  # NEW: Allow disabling camera via API
       source: CameraSourceConfig
   ```

2. **Add camera health monitoring**: The Application needs to periodically check camera health and call `notifier.publish_camera_health()` when status changes.

3. **Add stats methods to ClipRepository**:
   - `count_clips_since(since: datetime) -> int`
   - `count_alerts_since(since: datetime) -> int`

## Execution Order (Option A)

1. Phase 2: REST API for Configuration (control plane)
2. Phase 2.5: Home Assistant Notifier Plugin (real-time events)
3. Phase 4: Native Home Assistant Integration
4. Phase 3: Home Assistant Add-on
5. Phase 1: MQTT Discovery Enhancement (optional, for users who prefer MQTT)
6. Phase 5: Advanced Features

---

## Repository Structure

All code lives in the main `homesec` monorepo:

```
homesec/
├── repository.json                     # HA Add-on repo manifest (must be at root)
├── src/homesec/                        # Main Python package (PyPI)
│   ├── api/                            # NEW: REST API
│   │   ├── __init__.py
│   │   ├── server.py
│   │   ├── dependencies.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py
│   │       ├── config.py
│   │       ├── cameras.py
│   │       ├── clips.py
│   │       ├── events.py
│   │       ├── stats.py
│   │       └── system.py
│   ├── config/                         # NEW: Config management
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── manager.py
│   ├── plugins/
│   │   └── notifiers/
│   │       ├── home_assistant.py       # NEW: HA Events API notifier
│   │       └── mqtt_discovery.py       # NEW: MQTT discovery (optional)
│   └── ...existing code...
│
├── homeassistant/                      # ALL HA-specific code
│   ├── README.md
│   │
│   ├── integration/                    # Custom component (HACS)
│   │   ├── hacs.json
│   │   └── custom_components/
│   │       └── homesec/
│   │           ├── __init__.py
│   │           ├── manifest.json
│   │           ├── const.py
│   │           ├── config_flow.py
│   │           ├── coordinator.py
│   │           ├── entity.py
│   │           ├── binary_sensor.py
│   │           ├── sensor.py
│   │           ├── switch.py
│   │           ├── diagnostics.py
│   │           ├── services.yaml
│   │           ├── strings.json
│   │           └── translations/
│   │               └── en.json
│   │
│   └── addon/                          # Add-on (HA Supervisor)
│       ├── README.md
│       └── homesec/
│           ├── config.yaml
│           ├── build.yaml
│           ├── Dockerfile
│           ├── DOCS.md
│           ├── CHANGELOG.md
│           ├── icon.png
│           ├── logo.png
│           ├── rootfs/
│           │   └── etc/
│           │       ├── s6-overlay/
│           │       │   └── s6-rc.d/
│           │       │       ├── postgres-init/
│           │       │       ├── postgres/
│           │       │       ├── homesec/
│           │       │       └── user/
│           │       └── nginx/
│           │           └── includes/
│           │               └── ingress.conf
│           └── translations/
│               └── en.yaml
│
├── tests/
│   ├── unit/
│   │   ├── api/
│   │   │   ├── test_routes_cameras.py
│   │   │   ├── test_routes_clips.py
│   │   │   └── ...
│   │   └── plugins/
│   │       └── notifiers/
│   │           ├── test_home_assistant.py
│   │           └── test_mqtt_discovery.py
│   └── integration/
│       └── test_ha_integration.py
│
├── docs/
└── pyproject.toml                      # Add fastapi, uvicorn deps
```

**Distribution:**
- **PyPI**: `src/homesec/` published as `homesec` package
- **HACS**: Users point to `homeassistant/integration/`
- **Add-on Store**: Users add repo URL `https://github.com/lan17/homesec` (repository.json at repo root points to `homeassistant/addon/homesec/`)

---

## Table of Contents

1. [Decision Snapshot](#decision-snapshot-2026-02-01)
2. [Execution Order](#execution-order-option-a)
3. [Phase 2: REST API for Configuration](#phase-2-rest-api-for-configuration)
4. [Phase 2.5: Home Assistant Notifier Plugin](#phase-25-home-assistant-notifier-plugin)
5. [Phase 4: Native Home Assistant Integration](#phase-4-native-home-assistant-integration)
6. [Phase 3: Home Assistant Add-on](#phase-3-home-assistant-add-on)
7. [Phase 1: MQTT Discovery Enhancement (Optional)](#phase-1-mqtt-discovery-enhancement-optional)
8. [Phase 5: Advanced Features](#phase-5-advanced-features)
9. [Testing Strategy](#testing-strategy)
10. [Migration Guide](#migration-guide)

---

## Phase 1: MQTT Discovery Enhancement (Optional - Future)

> **⚠️ OUT OF SCOPE FOR V1**: This phase is optional and should be implemented only after the core integration (Phases 2-4) is complete and stable. The primary integration path uses the HA Events API which requires no MQTT broker.

**Goal:** Auto-create Home Assistant entities from HomeSec without requiring the native integration. This is for users who prefer MQTT over the primary HA Events API approach.

**Estimated Effort:** 2-3 days

**Warning:** If both MQTT Discovery AND the native integration are enabled, you will have duplicate entities. Users should choose one approach, not both.

### 1.1 Configuration Model Updates

**File:** `src/homesec/models/config.py`

Add MQTT discovery configuration to the existing `MQTTConfig`:

```python
class MQTTDiscoveryConfig(BaseModel):
    """Configuration for Home Assistant MQTT Discovery."""

    enabled: bool = False
    prefix: str = "homeassistant"  # HA discovery prefix
    node_id: str = "homesec"  # Unique node identifier
    device_name: str = "HomeSec"  # Display name in HA
    device_manufacturer: str = "HomeSec"
    device_model: str = "AI Security Pipeline"
    # Republish discovery on HA restart
    subscribe_to_birth: bool = True
    birth_topic: str = "homeassistant/status"


class MQTTConfig(BaseModel):
    """Extended MQTT configuration."""

    host: str
    port: int = 1883
    auth: MQTTAuthConfig | None = None
    topic_template: str = "homecam/alerts/{camera_name}"
    qos: int = 1
    retain: bool = False
    # New: Discovery settings
    discovery: MQTTDiscoveryConfig = MQTTDiscoveryConfig()
```

### 1.2 Discovery Message Builder

**New File:** `src/homesec/plugins/notifiers/mqtt_discovery.py`

```python
"""MQTT Discovery message builder for Home Assistant."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from homesec.models.config import MQTTDiscoveryConfig


@dataclass
class DiscoveryEntity:
    """Represents a single HA entity discovery config."""

    component: str  # sensor, binary_sensor, camera, etc.
    object_id: str  # Unique ID within component
    config: dict[str, Any]  # Discovery payload


class MQTTDiscoveryBuilder:
    """Builds MQTT discovery messages for Home Assistant."""

    def __init__(self, config: MQTTDiscoveryConfig, version: str):
        self.config = config
        self.version = version

    def _device_info(self, camera_name: str) -> dict[str, Any]:
        """Generate device info block for grouping entities."""
        return {
            "identifiers": [f"{self.config.node_id}_{camera_name}"],
            "name": f"{self.config.device_name} {camera_name}",
            "manufacturer": self.config.device_manufacturer,
            "model": self.config.device_model,
            "sw_version": self.version,
            "via_device": f"{self.config.node_id}_hub",
        }

    def _hub_device_info(self) -> dict[str, Any]:
        """Generate device info for the HomeSec hub."""
        return {
            "identifiers": [f"{self.config.node_id}_hub"],
            "name": self.config.device_name,
            "manufacturer": self.config.device_manufacturer,
            "model": f"{self.config.device_model} Hub",
            "sw_version": self.version,
        }

    def build_camera_entities(self, camera_name: str) -> list[DiscoveryEntity]:
        """Build all discovery entities for a single camera."""
        entities = []
        base_topic = f"homesec/{camera_name}"

        # 1. Binary Sensor: Motion Detected
        entities.append(DiscoveryEntity(
            component="binary_sensor",
            object_id=f"{camera_name}_motion",
            config={
                "name": "Motion",
                "unique_id": f"homesec_{camera_name}_motion",
                "device_class": "motion",
                "state_topic": f"{base_topic}/motion",
                "payload_on": "ON",
                "payload_off": "OFF",
                "device": self._device_info(camera_name),
            }
        ))

        # 2. Binary Sensor: Person Detected
        entities.append(DiscoveryEntity(
            component="binary_sensor",
            object_id=f"{camera_name}_person",
            config={
                "name": "Person Detected",
                "unique_id": f"homesec_{camera_name}_person",
                "device_class": "occupancy",
                "state_topic": f"{base_topic}/person",
                "payload_on": "ON",
                "payload_off": "OFF",
                "device": self._device_info(camera_name),
            }
        ))

        # 3. Sensor: Last Activity Type
        entities.append(DiscoveryEntity(
            component="sensor",
            object_id=f"{camera_name}_activity",
            config={
                "name": "Last Activity",
                "unique_id": f"homesec_{camera_name}_activity",
                "state_topic": f"{base_topic}/activity",
                "value_template": "{{ value_json.activity_type }}",
                "json_attributes_topic": f"{base_topic}/activity",
                "json_attributes_template": "{{ value_json | tojson }}",
                "icon": "mdi:motion-sensor",
                "device": self._device_info(camera_name),
            }
        ))

        # 4. Sensor: Risk Level
        entities.append(DiscoveryEntity(
            component="sensor",
            object_id=f"{camera_name}_risk",
            config={
                "name": "Risk Level",
                "unique_id": f"homesec_{camera_name}_risk",
                "state_topic": f"{base_topic}/risk",
                "icon": "mdi:shield-alert",
                "device": self._device_info(camera_name),
            }
        ))

        # 5. Sensor: Camera Health
        entities.append(DiscoveryEntity(
            component="sensor",
            object_id=f"{camera_name}_health",
            config={
                "name": "Health",
                "unique_id": f"homesec_{camera_name}_health",
                "state_topic": f"{base_topic}/health",
                "icon": "mdi:heart-pulse",
                "device": self._device_info(camera_name),
            }
        ))

        # 6. Sensor: Last Clip URL
        entities.append(DiscoveryEntity(
            component="sensor",
            object_id=f"{camera_name}_last_clip",
            config={
                "name": "Last Clip",
                "unique_id": f"homesec_{camera_name}_last_clip",
                "state_topic": f"{base_topic}/clip",
                "value_template": "{{ value_json.view_url }}",
                "json_attributes_topic": f"{base_topic}/clip",
                "icon": "mdi:filmstrip",
                "device": self._device_info(camera_name),
            }
        ))

        # 7. Image: Last Snapshot
        entities.append(DiscoveryEntity(
            component="image",
            object_id=f"{camera_name}_snapshot",
            config={
                "name": "Last Snapshot",
                "unique_id": f"homesec_{camera_name}_snapshot",
                "image_topic": f"{base_topic}/snapshot",
                "device": self._device_info(camera_name),
            }
        ))

        # 8. Device Trigger: Alert
        entities.append(DiscoveryEntity(
            component="device_automation",
            object_id=f"{camera_name}_alert_trigger",
            config={
                "automation_type": "trigger",
                "type": "alert",
                "subtype": "security_alert",
                "topic": f"{base_topic}/alert",
                "device": self._device_info(camera_name),
            }
        ))

        return entities

    def build_hub_entities(self) -> list[DiscoveryEntity]:
        """Build discovery entities for the HomeSec hub."""
        entities = []
        base_topic = "homesec/hub"

        # Hub Health
        entities.append(DiscoveryEntity(
            component="sensor",
            object_id="hub_status",
            config={
                "name": "Status",
                "unique_id": "homesec_hub_status",
                "state_topic": f"{base_topic}/status",
                "icon": "mdi:server",
                "device": self._hub_device_info(),
            }
        ))

        # Total Clips Today
        entities.append(DiscoveryEntity(
            component="sensor",
            object_id="hub_clips_today",
            config={
                "name": "Clips Today",
                "unique_id": "homesec_hub_clips_today",
                "state_topic": f"{base_topic}/stats",
                "value_template": "{{ value_json.clips_today }}",
                "icon": "mdi:filmstrip-box-multiple",
                "device": self._hub_device_info(),
            }
        ))

        # Total Alerts Today
        entities.append(DiscoveryEntity(
            component="sensor",
            object_id="hub_alerts_today",
            config={
                "name": "Alerts Today",
                "unique_id": "homesec_hub_alerts_today",
                "state_topic": f"{base_topic}/stats",
                "value_template": "{{ value_json.alerts_today }}",
                "icon": "mdi:bell-alert",
                "device": self._hub_device_info(),
            }
        ))

        return entities

    def get_discovery_topic(self, entity: DiscoveryEntity) -> str:
        """Get the MQTT topic for publishing discovery config."""
        return f"{self.config.prefix}/{entity.component}/{self.config.node_id}/{entity.object_id}/config"

    def get_discovery_payload(self, entity: DiscoveryEntity) -> str:
        """Get the JSON payload for discovery config."""
        return json.dumps(entity.config)
```

### 1.3 Enhanced MQTT Notifier

**File:** `src/homesec/plugins/notifiers/mqtt.py`

Extend the existing `paho-mqtt` notifier (do not switch libraries). Key changes:

- Keep the `Notifier.send()` interface and use `asyncio.to_thread` for publish operations.
- Add discovery publish helpers using `MQTTDiscoveryBuilder` (new file).
- Store camera names in-memory to publish discovery per camera.
- On HA birth message, republish discovery and current state topics.
- Keep backward-compatible alert topic publishing.

Implementation notes:

- Use `MQTTConfig.auth.username_env` and `MQTTConfig.auth.password_env` for credentials.
- Avoid blocking `loop_start`/`loop_stop` in the event loop (wrap in `asyncio.to_thread`).

### 1.4 Application Integration

**File:** `src/homesec/app.py`

Update the application to register cameras with the MQTT notifier. Prefer a small
`DiscoveryNotifier` Protocol (or `isinstance` check) over `hasattr` on private methods.

```python
# In Application._create_components(), after sources are created:

# Register cameras with discovery-capable notifier(s)
for entry in self._notifier_entries:
    notifier = entry.notifier
    if isinstance(notifier, DiscoveryNotifier):
        for source in self._sources:
            await notifier.register_camera(source.camera_name)

        # Publish initial discovery
        await notifier.publish_discovery()
```

### 1.5 Configuration Example

**File:** `config/example.yaml` (add section)

```yaml
notifiers:
  - backend: mqtt
    config:
      host: localhost
      port: 1883
      auth:
        username_env: MQTT_USERNAME
        password_env: MQTT_PASSWORD
      topic_template: "homecam/alerts/{camera_name}"
      qos: 1
      retain: false
      discovery:
        enabled: true
        prefix: homeassistant
        node_id: homesec
        device_name: HomeSec
        subscribe_to_birth: true
        birth_topic: homeassistant/status
```

### 1.6 Testing

**New File:** `tests/unit/plugins/notifiers/test_mqtt_discovery.py`

```python
"""Tests for MQTT Discovery functionality."""

import pytest
import json

from homesec.models.config import MQTTDiscoveryConfig
from homesec.plugins.notifiers.mqtt_discovery import MQTTDiscoveryBuilder


class TestMQTTDiscoveryBuilder:
    """Tests for MQTTDiscoveryBuilder."""

    @pytest.fixture
    def builder(self):
        config = MQTTDiscoveryConfig(enabled=True)
        return MQTTDiscoveryBuilder(config, "1.2.3")

    def test_build_camera_entities_creates_all_types(self, builder):
        """Should create all expected entity types for a camera."""
        entities = builder.build_camera_entities("front_door")

        components = {e.component for e in entities}
        assert "binary_sensor" in components
        assert "sensor" in components
        assert "image" in components
        assert "device_automation" in components

    def test_discovery_topic_format(self, builder):
        """Should generate correct discovery topic."""
        entities = builder.build_camera_entities("front_door")
        motion_entity = next(e for e in entities if "motion" in e.object_id)

        topic = builder.get_discovery_topic(motion_entity)
        assert topic == "homeassistant/binary_sensor/homesec/front_door_motion/config"

    def test_device_info_grouping(self, builder):
        """Should group entities under same device."""
        entities = builder.build_camera_entities("front_door")

        device_ids = {
            e.config["device"]["identifiers"][0]
            for e in entities
            if "device" in e.config
        }
        assert len(device_ids) == 1
        assert "homesec_front_door" in device_ids.pop()

    def test_hub_entities(self, builder):
        """Should create hub-level entities."""
        entities = builder.build_hub_entities()

        assert len(entities) >= 2
        assert any("status" in e.object_id for e in entities)
        assert any("clips_today" in e.object_id for e in entities)
```

### 1.7 Acceptance Criteria

- [ ] `mqtt.discovery.enabled: true` causes discovery messages to be published
- [ ] All cameras appear as devices in Home Assistant
- [ ] Binary sensors (motion, person) work correctly
- [ ] Sensors (activity, risk, health) update on alerts
- [ ] Device triggers fire for automations
- [ ] Discovery republishes when HA restarts (birth message)
- [ ] Backwards compatible with existing MQTT alert topic

---

## Phase 2: REST API for Configuration

**Goal:** Enable remote configuration and monitoring of HomeSec via HTTP API.

**Estimated Effort:** 5-7 days

### 2.0 Control Plane Requirements

- FastAPI only; all endpoints are `async def`.
- Use async SQLAlchemy only for DB access (no sync engines or blocking DB calls).
- No blocking operations inside endpoints; use `asyncio.to_thread` for file I/O and restarts.
- API must not write directly to `StateStore`/`EventStore`. Add read methods on `ClipRepository`.
- Config is loaded from **multiple YAML files** (left → right). Rightmost wins.
- Merge semantics: dicts deep-merge; lists replace.
- Config updates are validated with Pydantic, **written to the override YAML only**, and return `restart_required: true`.
- API provides a restart endpoint to request a graceful shutdown.
- Server config: introduce `FastAPIServerConfig` (host/port, enabled, api_key_env, CORS, health path) to replace `HealthConfig`.
- Secrets are never stored in config; only env var names are persisted.

### 2.1 API Framework Setup

**New File:** `src/homesec/api/__init__.py`

```python
"""HomeSec REST API package."""

from .server import create_app, APIServer

__all__ = ["create_app", "APIServer"]
```

**New File:** `src/homesec/api/server.py`

```python
"""REST API server for HomeSec."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import cameras, clips, config, events, health

if TYPE_CHECKING:
    from homesec.app import Application

logger = logging.getLogger(__name__)


def create_app(app_instance: Application) -> FastAPI:
    """Create the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Store reference to Application
        app.state.homesec = app_instance
        yield

    app = FastAPI(
        title="HomeSec API",
        description="REST API for HomeSec video security pipeline",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS for Home Assistant frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(config.router, prefix="/api/v1", tags=["config"])
    app.include_router(cameras.router, prefix="/api/v1", tags=["cameras"])
    app.include_router(clips.router, prefix="/api/v1", tags=["clips"])
    app.include_router(events.router, prefix="/api/v1", tags=["events"])
    # MQTT is used for event push (no WebSocket in v1)

    return app


class APIServer:
    """Manages the API server lifecycle."""

    def __init__(self, app: FastAPI, host: str = "0.0.0.0", port: int = 8080):
        self.app = app
        self.host = host
        self.port = port
        self._server = None

    async def start(self) -> None:
        """Start the API server."""
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)
        await self._server.serve()

    async def stop(self) -> None:
        """Stop the API server."""
        if self._server:
            self._server.should_exit = True
```

**Port coordination:** Replace the aiohttp `HealthServer` with FastAPI and make
the port configurable via `FastAPIServerConfig` (default 8080). Provide both
`/health` and `/api/v1/health` for compatibility.

### 2.2 Config Persistence + Restart

**New File:** `src/homesec/config/manager.py`

Responsibilities:

- Load and validate config from **multiple YAML files** (left → right). Rightmost wins.
- Base YAML is bootstrap-only; **override YAML** contains all HA-managed config.
- Merge semantics: dicts deep-merge; lists replace.
- Persist updated override YAML atomically (write temp file, fsync, rename).
- Store `config_version` in the override file and enforce optimistic concurrency.
- Return `restart_required: true` for any config-changing endpoints.
- Expose methods:
  - `get_config() -> Config`
  - `update_config(new_config: dict) -> ConfigUpdateResult`
  - `update_camera(...) -> ConfigUpdateResult`
  - `remove_camera(...) -> ConfigUpdateResult`
- Use `asyncio.to_thread` for file I/O to keep endpoints non-blocking.
- Provide `dump_override(path: Path)` to export backup YAML.
  - `ConfigUpdateResult` should include the new `config_version`.
- Application should expose `config_store` and `config_version` for API routes.
- Override file is machine-owned; comment preservation is not required.
- Application should load configs via ConfigManager with multiple `--config` paths.

CLI requirements:

- Support multiple `--config` flags (order matters).
- Default override path: `config/ha-overrides.yaml` (override can be passed as the last `--config`).

**Repository extensions:**

- Add read APIs to `ClipRepository`:
  - `get_clip(clip_id)`
  - `list_clips(...)`
  - `list_events(...)`
  - `delete_clip(clip_id)` (mark deleted + emit event)
- Implement with async SQLAlchemy in `PostgresStateStore` / `PostgresEventStore`.

**New Endpoint:** `POST /api/v1/system/restart`

- Triggers graceful shutdown (`Application.request_shutdown()`).
- HA can call this after config update, or restart the add-on.

### 2.3 API Routes

**New File:** `src/homesec/api/routes/cameras.py`

```python
"""Camera management API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..dependencies import get_homesec_app

router = APIRouter(prefix="/cameras")

# Note: All config-mutating endpoints return restart_required=True and do not
# attempt hot-reload. HA may call /api/v1/system/restart or restart the add-on.
# All config-mutating endpoints require config_version for optimistic concurrency.


class CameraCreate(BaseModel):
    """Request model for creating a camera."""
    name: str
    source_backend: str  # rtsp, ftp, local_folder
    source_config: dict  # Backend-specific configuration
    config_version: int


class CameraUpdate(BaseModel):
    """Request model for updating a camera."""
    source_config: dict | None = None
    config_version: int


class CameraResponse(BaseModel):
    """Response model for camera."""
    name: str
    source_backend: str
    healthy: bool
    last_heartbeat: float | None
    source_config: dict


class CameraListResponse(BaseModel):
    """Response model for camera list."""
    cameras: list[CameraResponse]
    total: int


class ConfigChangeResponse(BaseModel):
    """Response model for config changes."""
    restart_required: bool = True
    config_version: int
    camera: CameraResponse | None = None


@router.get("", response_model=CameraListResponse)
async def list_cameras(app=Depends(get_homesec_app)):
    """List all configured cameras."""
    cameras = []
    config = app.config_store.get_config()
    for camera in config.cameras:
        source = app.get_source(camera.name)
        cameras.append(CameraResponse(
            name=camera.name,
            source_backend=camera.source.backend,
            healthy=source.is_healthy() if source else False,
            last_heartbeat=source.last_heartbeat() if source else None,
            source_config=camera.source.config if isinstance(camera.source.config, dict) else camera.source.config.model_dump(),
        ))
    return CameraListResponse(cameras=cameras, total=len(cameras))


@router.get("/{camera_name}", response_model=CameraResponse)
async def get_camera(camera_name: str, app=Depends(get_homesec_app)):
    """Get a specific camera's configuration."""
    config = app.config_store.get_config()
    camera = next((c for c in config.cameras if c.name == camera_name), None)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{camera_name}' not found",
        )
    source = app.get_source(camera_name)
    return CameraResponse(
        name=camera.name,
        source_backend=camera.source.backend,
        healthy=source.is_healthy() if source else False,
        last_heartbeat=source.last_heartbeat() if source else None,
        source_config=camera.source.config if isinstance(camera.source.config, dict) else camera.source.config.model_dump(),
    )


@router.post("", response_model=ConfigChangeResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(camera: CameraCreate, app=Depends(get_homesec_app)):
    """Add a new camera."""
    if app.get_source(camera.name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Camera '{camera.name}' already exists",
        )

    try:
        result = await app.config_store.add_camera(
            name=camera.name,
            source_backend=camera.source_backend,
            source_config=camera.source_config,
            config_version=camera.config_version,
        )
        return ConfigChangeResponse(
            restart_required=True,
            config_version=result.config_version,
            camera=CameraResponse(
                name=camera.name,
                source_backend=camera.source_backend,
                healthy=False,
                last_heartbeat=None,
                source_config=camera.source_config,
            ),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.put("/{camera_name}", response_model=ConfigChangeResponse)
async def update_camera(
    camera_name: str,
    update: CameraUpdate,
    app=Depends(get_homesec_app),
):
    """Update a camera's source configuration."""
    source = app.get_source(camera_name)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{camera_name}' not found",
        )

    result = await app.config_store.update_camera(
        camera_name=camera_name,
        source_config=update.source_config,
        config_version=update.config_version,
    )

    return ConfigChangeResponse(
        restart_required=True,
        config_version=result.config_version,
        camera=result.camera,
    )


@router.delete("/{camera_name}", response_model=ConfigChangeResponse)
async def delete_camera(
    camera_name: str,
    config_version: int,
    app=Depends(get_homesec_app),
):
    """Remove a camera."""
    source = app.get_source(camera_name)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{camera_name}' not found",
        )

    result = await app.config_store.remove_camera(
        camera_name=camera_name,
        config_version=config_version,
    )
    return ConfigChangeResponse(
        restart_required=True,
        config_version=result.config_version,
        camera=None,
    )


@router.get("/{camera_name}/status")
async def get_camera_status(camera_name: str, app=Depends(get_homesec_app)):
    """Get detailed camera status."""
    source = app.get_source(camera_name)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{camera_name}' not found",
        )

    return {
        "name": camera_name,
        "healthy": source.is_healthy(),
        "last_heartbeat": source.last_heartbeat(),
        "last_heartbeat_age_s": source.last_heartbeat_age(),
        "stats": await source.get_stats(),
    }


@router.post("/{camera_name}/test")
async def test_camera(camera_name: str, app=Depends(get_homesec_app)):
    """Test camera connection."""
    source = app.get_source(camera_name)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{camera_name}' not found",
        )

    try:
        result = await source.ping()
        return {"success": result, "message": "Connection successful" if result else "Connection failed"}
    except Exception as e:
        return {"success": False, "message": str(e)}
```

**New File:** `src/homesec/api/routes/clips.py`

```python
"""Clip management API routes."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from ..dependencies import get_homesec_app

router = APIRouter(prefix="/clips")


class ClipResponse(BaseModel):
    """Response model for a clip."""
    clip_id: str
    camera_name: str
    status: str
    created_at: datetime
    storage_uri: str | None
    view_url: str | None
    filter_result: dict | None
    analysis_result: dict | None
    alert_decision: dict | None


class ClipListResponse(BaseModel):
    """Response model for clip list."""
    clips: list[ClipResponse]
    total: int
    page: int
    page_size: int


@router.get("", response_model=ClipListResponse)
async def list_clips(
    app=Depends(get_homesec_app),
    camera: str | None = None,
    status: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
):
    """List clips with optional filtering."""
    clips, total = await app.repository.list_clips(
        camera=camera,
        status=status,
        since=since,
        until=until,
        offset=(page - 1) * page_size,
        limit=page_size,
    )

    return ClipListResponse(
        clips=[ClipResponse(**c.model_dump()) for c in clips],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{clip_id}", response_model=ClipResponse)
async def get_clip(clip_id: str, app=Depends(get_homesec_app)):
    """Get a specific clip."""
    clip = await app.repository.get_clip(clip_id)
    if not clip:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clip '{clip_id}' not found",
        )
    return ClipResponse(**clip.model_dump())


@router.delete("/{clip_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_clip(clip_id: str, app=Depends(get_homesec_app)):
    """Delete a clip."""
    clip = await app.repository.get_clip(clip_id)
    if not clip:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clip '{clip_id}' not found",
        )
    await app.repository.delete_clip(clip_id)


@router.post("/{clip_id}/reprocess", status_code=status.HTTP_202_ACCEPTED)
async def reprocess_clip(clip_id: str, app=Depends(get_homesec_app)):
    """Reprocess a clip through the pipeline."""
    clip = await app.repository.get_clip(clip_id)
    if not clip:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clip '{clip_id}' not found",
        )

    await app.pipeline.reprocess(clip_id)
    return {"message": "Clip queued for reprocessing", "clip_id": clip_id}
```

**New File:** `src/homesec/api/routes/events.py`

```python
"""Event history API routes."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from ..dependencies import get_homesec_app

router = APIRouter(prefix="/events")


class EventResponse(BaseModel):
    """Response model for an event."""
    id: int
    clip_id: str
    event_type: str
    timestamp: datetime
    event_data: dict


class EventListResponse(BaseModel):
    """Response model for event list."""
    events: list[EventResponse]
    total: int


@router.get("", response_model=EventListResponse)
async def list_events(
    app=Depends(get_homesec_app),
    clip_id: str | None = None,
    event_type: str | None = None,
    camera: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = Query(100, ge=1, le=1000),
):
    """List events with optional filtering."""
    events, total = await app.repository.list_events(
        clip_id=clip_id,
        event_type=event_type,
        camera=camera,
        since=since,
        until=until,
        limit=limit,
    )

    return EventListResponse(
        events=[EventResponse(**e.model_dump()) for e in events],
        total=total,
    )
```

**New File:** `src/homesec/api/routes/stats.py`

```python
"""Statistics API routes."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..dependencies import get_homesec_app

router = APIRouter(prefix="/stats")


class StatsResponse(BaseModel):
    """Response model for statistics."""
    clips_today: int
    clips_total: int
    alerts_today: int
    alerts_total: int
    cameras_online: int
    cameras_total: int
    last_alert_at: datetime | None
    last_clip_at: datetime | None


@router.get("", response_model=StatsResponse)
async def get_stats(app=Depends(get_homesec_app)):
    """Get system-wide statistics."""
    today = date.today()
    today_start = datetime.combine(today, datetime.min.time())

    # Get clip counts
    _, clips_today = await app.repository.list_clips(since=today_start, limit=0)
    _, clips_total = await app.repository.list_clips(limit=0)

    # Get alert counts (notifications sent)
    _, alerts_today = await app.repository.list_events(
        event_type="notification_sent",
        since=today_start,
        limit=0,
    )
    _, alerts_total = await app.repository.list_events(
        event_type="notification_sent",
        limit=0,
    )

    # Get camera health
    cameras = app.get_all_sources()
    cameras_online = sum(1 for c in cameras if c.is_healthy())

    # Get last timestamps
    recent_clips, _ = await app.repository.list_clips(limit=1)
    recent_alerts, _ = await app.repository.list_events(
        event_type="notification_sent",
        limit=1,
    )

    return StatsResponse(
        clips_today=clips_today,
        clips_total=clips_total,
        alerts_today=alerts_today,
        alerts_total=alerts_total,
        cameras_online=cameras_online,
        cameras_total=len(cameras),
        last_alert_at=recent_alerts[0].timestamp if recent_alerts else None,
        last_clip_at=recent_clips[0].created_at if recent_clips else None,
    )
```

**New File:** `src/homesec/api/routes/health.py`

```python
"""Health check API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel

from ..dependencies import get_homesec_app

router = APIRouter()


class SourceHealth(BaseModel):
    """Health status for a single source."""
    name: str
    healthy: bool
    last_heartbeat: float | None
    last_heartbeat_age_s: float | None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    uptime_s: float
    sources: list[SourceHealth]
    postgres_connected: bool


@router.get("/health", response_model=HealthResponse)
@router.get("/api/v1/health", response_model=HealthResponse)
async def health_check(
    response: Response,
    app=Depends(get_homesec_app),
):
    """Health check endpoint."""
    sources = []
    all_healthy = True
    any_healthy = False

    for source in app.get_all_sources():
        healthy = source.is_healthy()
        if healthy:
            any_healthy = True
        else:
            all_healthy = False

        sources.append(SourceHealth(
            name=source.camera_name,
            healthy=healthy,
            last_heartbeat=source.last_heartbeat(),
            last_heartbeat_age_s=source.last_heartbeat_age(),
        ))

    # Check Postgres connectivity
    postgres_connected = await app.repository.ping()

    # Determine overall status
    if all_healthy and postgres_connected:
        health_status = "healthy"
    elif any_healthy:
        health_status = "degraded"
    else:
        health_status = "unhealthy"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status=health_status,
        version=app.version,
        uptime_s=app.uptime_seconds(),
        sources=sources,
        postgres_connected=postgres_connected,
    )
```

Real-time updates use HA Events API; no WebSocket endpoint in v1.

### 2.4 API Configuration

**File:** `src/homesec/models/config.py` (add)

```python
class FastAPIServerConfig(BaseModel):
    """Configuration for the FastAPI server."""

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: list[str] = ["*"]
    # Authentication (optional)
    auth_enabled: bool = False
    api_key_env: str | None = None
    # Health endpoints
    health_path: str = "/health"
    api_health_path: str = "/api/v1/health"

# In Config:
# - Replace `health: HealthConfig` with `server: FastAPIServerConfig`
```

### 2.5 OpenAPI Documentation

The FastAPI app automatically generates OpenAPI docs at `/api/v1/docs` (Swagger UI) and `/api/v1/redoc` (ReDoc).

### 2.6 Acceptance Criteria

- [ ] All CRUD operations for cameras work
- [ ] Config changes are validated, persisted, and return `restart_required: true`
- [ ] Stale `config_version` updates return 409 Conflict
- [ ] Restart endpoint triggers graceful shutdown
- [ ] Clip listing with filtering works
- [ ] Event history API works
- [ ] OpenAPI documentation is accurate
- [ ] CORS works for Home Assistant frontend
- [ ] API authentication (optional) works
- [ ] Returns 503 when Postgres is unavailable

---

## Phase 2.5: Home Assistant Notifier Plugin

**Goal:** Enable real-time event push from HomeSec to Home Assistant without requiring MQTT.

**Estimated Effort:** 2-3 days

### 2.5.1 Configuration Model

**File:** `src/homesec/models/config.py`

```python
class HomeAssistantNotifierConfig(BaseModel):
    """Configuration for Home Assistant notifier."""

    # For standalone mode (not running as add-on)
    # When running as HA add-on, SUPERVISOR_TOKEN is used automatically
    url_env: str | None = None      # e.g., "HA_URL" -> http://homeassistant.local:8123
    token_env: str | None = None    # e.g., "HA_TOKEN" -> long-lived access token

    # Event configuration
    event_prefix: str = "homesec"   # Events will be homesec_alert, homesec_health, etc.
```

### 2.5.2 Home Assistant Notifier Implementation

**New File:** `src/homesec/plugins/notifiers/home_assistant.py`

```python
"""Home Assistant notifier - pushes events directly to HA Events API."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import aiohttp

from homesec.interfaces import Notifier
from homesec.models.config import HomeAssistantNotifierConfig
from homesec.plugins.registry import plugin, PluginType

if TYPE_CHECKING:
    from homesec.models.alert import Alert

logger = logging.getLogger(__name__)


@plugin(plugin_type=PluginType.NOTIFIER, name="home_assistant")
class HomeAssistantNotifier(Notifier):
    """Push events directly to Home Assistant via Events API."""

    config_cls = HomeAssistantNotifierConfig

    def __init__(self, config: HomeAssistantNotifierConfig):
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        self._supervisor_mode = False

    async def start(self) -> None:
        """Initialize the HTTP session and detect supervisor mode."""
        self._session = aiohttp.ClientSession()

        # Detect if running as HA add-on (SUPERVISOR_TOKEN is injected automatically)
        if os.environ.get("SUPERVISOR_TOKEN"):
            self._supervisor_mode = True
            logger.info("HomeAssistantNotifier: Running in supervisor mode (zero-config)")
        else:
            # Standalone mode - validate config
            if not self.config.url_env or not self.config.token_env:
                raise ValueError(
                    "HomeAssistantNotifier requires url_env and token_env in standalone mode"
                )
            logger.info("HomeAssistantNotifier: Running in standalone mode")

    async def stop(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    def _get_url_and_headers(self, event_type: str) -> tuple[str, dict[str, str]]:
        """Get the URL and headers for the HA Events API."""
        if self._supervisor_mode:
            url = f"http://supervisor/core/api/events/{self.config.event_prefix}_{event_type}"
            headers = {
                "Authorization": f"Bearer {os.environ['SUPERVISOR_TOKEN']}",
                "Content-Type": "application/json",
            }
        else:
            from homesec.config import resolve_env_var
            base_url = resolve_env_var(self.config.url_env)
            token = resolve_env_var(self.config.token_env)
            url = f"{base_url}/api/events/{self.config.event_prefix}_{event_type}"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
        return url, headers

    async def notify(self, alert: Alert) -> None:
        """Send alert to Home Assistant as an event."""
        if not self._session:
            raise RuntimeError("HomeAssistantNotifier not started")

        url, headers = self._get_url_and_headers("alert")

        event_data = {
            "camera": alert.camera_name,
            "clip_id": alert.clip_id,
            "activity_type": alert.activity_type,
            "risk_level": alert.risk_level.value if alert.risk_level else None,
            "summary": alert.summary,
            "view_url": alert.view_url,
            "storage_uri": alert.storage_uri,
            "timestamp": alert.ts.isoformat(),
        }

        # Add analysis details if present
        if alert.analysis:
            event_data["detected_objects"] = alert.analysis.detected_objects
            event_data["analysis"] = alert.analysis.model_dump(mode="json")

        try:
            async with self._session.post(url, json=event_data, headers=headers) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error(
                        "Failed to send event to HA: %s %s - %s",
                        resp.status,
                        resp.reason,
                        body,
                    )
                else:
                    logger.debug("Sent homesec_alert event to HA for clip %s", alert.clip_id)
        except aiohttp.ClientError as exc:
            logger.error("Failed to connect to Home Assistant: %s", exc)
            # Don't raise - notifications are best-effort

    async def publish_camera_health(self, camera_name: str, healthy: bool) -> None:
        """Publish camera health status to HA."""
        if not self._session:
            return

        url, headers = self._get_url_and_headers("camera_health")
        event_data = {
            "camera": camera_name,
            "healthy": healthy,
            "status": "healthy" if healthy else "unhealthy",
        }

        try:
            async with self._session.post(url, json=event_data, headers=headers) as resp:
                if resp.status >= 400:
                    logger.warning("Failed to send camera health to HA: %s", resp.status)
        except aiohttp.ClientError:
            pass  # Best effort

    async def publish_clip_recorded(self, clip_id: str, camera_name: str) -> None:
        """Publish clip recorded event to HA."""
        if not self._session:
            return

        url, headers = self._get_url_and_headers("clip_recorded")
        event_data = {
            "clip_id": clip_id,
            "camera": camera_name,
        }

        try:
            async with self._session.post(url, json=event_data, headers=headers) as resp:
                if resp.status >= 400:
                    logger.warning("Failed to send clip_recorded to HA: %s", resp.status)
        except aiohttp.ClientError:
            pass  # Best effort
```

### 2.5.3 Configuration Example

**File:** `config/example.yaml` (add section)

```yaml
notifiers:
  # Home Assistant notifier (recommended for HA users)
  - backend: home_assistant
    config:
      # When running as HA add-on, no configuration needed (uses SUPERVISOR_TOKEN)
      # For standalone mode, provide HA URL and token:
      # url_env: HA_URL           # http://homeassistant.local:8123
      # token_env: HA_TOKEN       # Long-lived access token from HA
```

### 2.5.4 Testing

**New File:** `tests/unit/plugins/notifiers/test_home_assistant.py`

```python
"""Tests for Home Assistant notifier."""

import os
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import ClientResponseError

from homesec.models.config import HomeAssistantNotifierConfig
from homesec.plugins.notifiers.home_assistant import HomeAssistantNotifier


class TestHomeAssistantNotifier:
    """Tests for HomeAssistantNotifier."""

    @pytest.fixture
    def config(self):
        return HomeAssistantNotifierConfig(
            url_env="HA_URL",
            token_env="HA_TOKEN",
        )

    @pytest.fixture
    def notifier(self, config):
        return HomeAssistantNotifier(config)

    async def test_supervisor_mode_detection(self, notifier):
        # Given: SUPERVISOR_TOKEN is set
        with patch.dict(os.environ, {"SUPERVISOR_TOKEN": "test_token"}):
            # When: notifier starts
            await notifier.start()

            # Then: supervisor mode is enabled
            assert notifier._supervisor_mode is True

        await notifier.stop()

    async def test_standalone_mode_requires_config(self, config):
        # Given: config without url_env
        config.url_env = None

        # When: notifier starts without SUPERVISOR_TOKEN
        notifier = HomeAssistantNotifier(config)
        with patch.dict(os.environ, {}, clear=True):
            # Then: ValueError is raised
            with pytest.raises(ValueError, match="url_env and token_env"):
                await notifier.start()

    async def test_notify_sends_event(self, notifier, alert_fixture):
        # Given: notifier is started in standalone mode
        with patch.dict(os.environ, {"HA_URL": "http://ha:8123", "HA_TOKEN": "token"}):
            await notifier.start()

            # When: notify is called
            with patch.object(notifier._session, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value.__aenter__.return_value.status = 200
                await notifier.notify(alert_fixture)

                # Then: event is posted to HA
                mock_post.assert_called_once()
                call_args = mock_post.call_args
                assert "homesec_alert" in call_args[0][0]

        await notifier.stop()
```

### 2.5.5 Acceptance Criteria

- [ ] Notifier auto-detects supervisor mode via `SUPERVISOR_TOKEN`
- [ ] Zero-config when running as HA add-on
- [ ] Standalone mode requires `url_env` and `token_env`
- [ ] Events are fired: `homesec_alert`, `homesec_camera_health`, `homesec_clip_recorded`
- [ ] Notification failures don't crash the pipeline (best-effort)
- [ ] Events contain all required metadata for HA integration

---

## Phase 3: Home Assistant Add-on

**Goal:** Provide one-click installation for Home Assistant OS/Supervised users.

**Estimated Effort:** 3-4 days

### 3.1 Add-on Repository Structure

**Location:** `homeassistant/addon/` in the main homesec monorepo.

Users add the add-on via: `https://github.com/lan17/homesec`

Note: `repository.json` must be at the repo root (not in `homeassistant/addon/`) for Home Assistant to discover it. The file points to `homeassistant/addon/homesec/` as the add-on location.

```
homeassistant/addon/
├── README.md
└── homesec/
    ├── config.yaml           # Add-on manifest
    ├── Dockerfile            # Container build
    ├── build.yaml            # Build configuration
    ├── DOCS.md               # Documentation
    ├── CHANGELOG.md          # Version history
    ├── icon.png              # Add-on icon (512x512)
    ├── logo.png              # Add-on logo (256x256)
    ├── rootfs/               # s6-overlay services
    │   └── etc/
    │       ├── s6-overlay/
    │       └── nginx/
    └── translations/
        └── en.yaml           # UI strings
```

### 3.2 Add-on Manifest

**File:** `homeassistant/addon/homesec/config.yaml`

Note: Update to current Home Assistant add-on schema:

- Use `addon_config` mapping instead of `config:rw` where possible.
- Read runtime options from `/data/options.json` via Bashio.
- Keep secrets out of the generated config (env vars only).

```yaml
name: HomeSec
version: "1.2.2"
slug: homesec
description: Self-hosted AI video security pipeline
url: https://github.com/lan17/homesec
arch:
  - amd64
  - aarch64
init: false
homeassistant_api: true
hassio_api: true
host_network: false
ingress: true
ingress_port: 8080
ingress_stream: true
panel_icon: mdi:cctv
panel_title: HomeSec

# Port mappings
ports:
  8080/tcp: null    # API (exposed via ingress)

# Volume mappings
map:
  - addon_config:rw  # /config - HomeSec managed config
  - media:rw         # /media - Media storage
  - share:rw         # /share - Shared data

# Services
services:
  - mqtt:want        # Optional - only needed if user enables MQTT Discovery

# Options schema
schema:
  config_path: str?
  override_path: str?
  log_level: list(debug|info|warning|error)?
  # Database
  database_url: str?
  # Storage
  storage_type: list(local|dropbox)?
  storage_path: str?
  dropbox_token: password?
  # VLM
  vlm_enabled: bool?
  openai_api_key: password?
  # VLM model
  openai_model: str?
  # MQTT Discovery (optional - primary path uses HA Events API)
  mqtt_discovery: bool?

# Default options
options:
  config_path: /config/homesec/config.yaml
  override_path: /data/overrides.yaml
  log_level: info
  database_url: ""
  storage_type: local
  storage_path: /media/homesec/clips
  vlm_enabled: false
  mqtt_discovery: true

# Startup dependencies
startup: services
stage: stable

# Advanced options
advanced: true
privileged: []
apparmor: true

# Watchdog for auto-restart
watchdog: http://[HOST]:[PORT:8080]/api/v1/health
```

### 3.3 Add-on Dockerfile

**File:** `homeassistant/addon/homesec/Dockerfile`

```dockerfile
# syntax=docker/dockerfile:1
ARG BUILD_FROM=ghcr.io/hassio-addons/base:15.0.8
FROM ${BUILD_FROM}

# Install runtime dependencies including PostgreSQL
RUN apk add --no-cache \
    python3 \
    py3-pip \
    ffmpeg \
    postgresql16 \
    postgresql16-contrib \
    opencv \
    curl \
    && rm -rf /var/cache/apk/*

# Create PostgreSQL directories
RUN mkdir -p /run/postgresql /data/postgres \
    && chown -R postgres:postgres /run/postgresql /data/postgres

# Install HomeSec
ARG HOMESEC_VERSION=1.2.2
RUN pip3 install --no-cache-dir homesec==${HOMESEC_VERSION}

# Copy root filesystem (s6-overlay services)
COPY rootfs /

# Set working directory
WORKDIR /app

# Labels
LABEL \
    io.hass.name="HomeSec" \
    io.hass.description="Self-hosted AI video security pipeline" \
    io.hass.version="${HOMESEC_VERSION}" \
    io.hass.type="addon" \
    io.hass.arch="amd64|aarch64"

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1
```

### 3.4 s6-overlay Service Structure

The add-on uses s6-overlay to run PostgreSQL and HomeSec as two services with proper dependency ordering.

**Directory structure:**

```
rootfs/etc/s6-overlay/s6-rc.d/
├── postgres/
│   ├── type                    # Contains: longrun
│   ├── run                     # PostgreSQL startup script
│   └── dependencies.d/
│       └── base                # Depends on base setup
├── postgres-init/
│   ├── type                    # Contains: oneshot
│   ├── up                      # Initialize DB if needed
│   └── dependencies.d/
│       └── base
├── homesec/
│   ├── type                    # Contains: longrun
│   ├── run                     # HomeSec startup script
│   └── dependencies.d/
│       └── postgres            # Waits for PostgreSQL
└── user/
    └── contents.d/
        ├── postgres-init
        ├── postgres
        └── homesec
```

### 3.5 PostgreSQL Initialization Service

**File:** `homeassistant/addon/homesec/rootfs/etc/s6-overlay/s6-rc.d/postgres-init/up`

```bash
#!/command/with-contenv bashio

# Skip if already initialized
if [[ -f /data/postgres/data/PG_VERSION ]]; then
    bashio::log.info "PostgreSQL already initialized"
    exit 0
fi

bashio::log.info "Initializing PostgreSQL database..."

# Create data directory
mkdir -p /data/postgres/data
chown -R postgres:postgres /data/postgres

# Initialize database cluster
su postgres -c "initdb -D /data/postgres/data --encoding=UTF8 --locale=C"

# Configure PostgreSQL for local connections only
cat >> /data/postgres/data/postgresql.conf << EOF
listen_addresses = 'localhost'
max_connections = 20
shared_buffers = 128MB
EOF

# Start PostgreSQL temporarily to create database
su postgres -c "pg_ctl -D /data/postgres/data start -w -o '-c listen_addresses=localhost'"

# Create homesec database and user
su postgres -c "createdb homesec"
su postgres -c "psql -c \"ALTER USER postgres PASSWORD 'homesec';\""

# Stop PostgreSQL (will be started by longrun service)
su postgres -c "pg_ctl -D /data/postgres/data stop -w"

bashio::log.info "PostgreSQL initialization complete"
```

### 3.6 PostgreSQL Service

**File:** `homeassistant/addon/homesec/rootfs/etc/s6-overlay/s6-rc.d/postgres/run`

```bash
#!/command/with-contenv bashio

bashio::log.info "Starting PostgreSQL..."

# Run PostgreSQL in foreground
exec su postgres -c "postgres -D /data/postgres/data"
```

### 3.7 HomeSec Service

**File:** `homeassistant/addon/homesec/rootfs/etc/s6-overlay/s6-rc.d/homesec/run`

```bash
#!/command/with-contenv bashio
# ==============================================================================
# HomeSec Service - runs after PostgreSQL is ready
# ==============================================================================

# Read add-on options
CONFIG_PATH=$(bashio::config 'config_path')
OVERRIDE_PATH=$(bashio::config 'override_path')
LOG_LEVEL=$(bashio::config 'log_level')
EXTERNAL_DB_URL=$(bashio::config 'database_url')
STORAGE_TYPE=$(bashio::config 'storage_type')
STORAGE_PATH=$(bashio::config 'storage_path')
MQTT_DISCOVERY=$(bashio::config 'mqtt_discovery')

# Wait for PostgreSQL to be ready (if using bundled)
if [[ -z "${EXTERNAL_DB_URL}" ]]; then
    bashio::log.info "Waiting for bundled PostgreSQL..."
    until pg_isready -h localhost -U postgres -q; do
        sleep 1
    done
    export DATABASE_URL="postgresql://postgres:homesec@localhost/homesec"
    bashio::log.info "Bundled PostgreSQL is ready"
else
    export DATABASE_URL="${EXTERNAL_DB_URL}"
    bashio::log.info "Using external database: ${EXTERNAL_DB_URL%%@*}@..."
fi

# Get MQTT credentials from HA if MQTT Discovery enabled
if [[ "${MQTT_DISCOVERY}" == "true" ]] && bashio::services.available "mqtt"; then
    MQTT_HOST=$(bashio::services mqtt "host")
    MQTT_PORT=$(bashio::services mqtt "port")
    MQTT_USER=$(bashio::services mqtt "username")
    MQTT_PASS=$(bashio::services mqtt "password")
    export MQTT_HOST MQTT_PORT MQTT_USER MQTT_PASS
    bashio::log.info "MQTT Discovery enabled via ${MQTT_HOST}:${MQTT_PORT}"
fi

# Create directories
mkdir -p "$(dirname "${CONFIG_PATH}")"
mkdir -p "$(dirname "${OVERRIDE_PATH}")"
mkdir -p "${STORAGE_PATH}"

# Generate base config if it doesn't exist
if [[ ! -f "${CONFIG_PATH}" ]]; then
    bashio::log.info "Generating initial configuration at ${CONFIG_PATH}"
    cat > "${CONFIG_PATH}" << EOF
version: 1

cameras: []

storage:
  backend: ${STORAGE_TYPE}
  config:
    path: ${STORAGE_PATH}

state_store:
  dsn_env: DATABASE_URL

notifiers:
  # Primary: Push events to HA via Events API (uses SUPERVISOR_TOKEN automatically)
  - backend: home_assistant
    config: {}

server:
  enabled: true
  host: 0.0.0.0
  port: 8080
EOF

    # Optionally add MQTT notifier if user enabled MQTT Discovery
    if [[ "${MQTT_DISCOVERY}" == "true" ]] && bashio::services.available "mqtt"; then
        bashio::log.info "Adding MQTT Discovery notifier to config"
        cat >> "${CONFIG_PATH}" << EOF

  # Optional: MQTT Discovery for users who prefer MQTT entities
  - backend: mqtt
    config:
      host_env: MQTT_HOST
      port_env: MQTT_PORT
      auth:
        username_env: MQTT_USER
        password_env: MQTT_PASS
      discovery:
        enabled: true
EOF
    fi
fi

# Create override file if missing
if [[ ! -f "${OVERRIDE_PATH}" ]]; then
    bashio::log.info "Creating override file at ${OVERRIDE_PATH}"
    cat > "${OVERRIDE_PATH}" << EOF
version: 1
config_version: 1
EOF
fi

bashio::log.info "Starting HomeSec..."

# Run HomeSec with both config files (base + overrides)
exec python3 -m homesec.cli run \
    --config "${CONFIG_PATH}" \
    --config "${OVERRIDE_PATH}" \
    --log-level "${LOG_LEVEL}"
```

### 3.8 Ingress Configuration

**File:** `homeassistant/addon/homesec/rootfs/etc/nginx/includes/ingress.conf`

```nginx
# Proxy to HomeSec API
location / {
    proxy_pass http://127.0.0.1:8080;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # Timeouts for long-running connections
    proxy_connect_timeout 60s;
    proxy_send_timeout 60s;
    proxy_read_timeout 60s;
}
```

### 3.9 Acceptance Criteria

- [ ] Add-on installs successfully from monorepo URL
- [ ] Bundled PostgreSQL starts and initializes correctly
- [ ] HomeSec waits for PostgreSQL before starting
- [ ] SUPERVISOR_TOKEN enables zero-config HA Events API
- [ ] Ingress provides access to API/UI
- [ ] Configuration options work correctly
- [ ] Watchdog restarts on failure
- [ ] Logs are accessible in HA
- [ ] Works on both amd64 and aarch64
- [ ] Optional: MQTT Discovery works if user enables it

---

## Phase 4: Native Home Assistant Integration

**Goal:** Full UI-based configuration and deep entity integration in Home Assistant.

**Estimated Effort:** 7-10 days

### 4.1 Integration Structure

**Directory:** `homeassistant/integration/custom_components/homesec/`

```
homeassistant/integration/custom_components/homesec/
├── __init__.py           # Setup and entry points
├── manifest.json         # Integration metadata
├── const.py              # Constants
├── config_flow.py        # UI configuration flow
├── coordinator.py        # Data update coordinator
├── entity.py             # Base entity class
├── sensor.py             # Sensor platform
├── binary_sensor.py      # Binary sensor platform
├── switch.py             # Switch platform
├── select.py             # Select platform
├── image.py              # Image platform
├── diagnostics.py        # Diagnostic data
├── services.yaml         # Service definitions
├── strings.json          # UI strings
└── translations/
    └── en.json           # English translations
```

### 4.2 Manifest

**File:** `homeassistant/integration/custom_components/homesec/manifest.json`

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

### 4.3 Constants

**File:** `homeassistant/integration/custom_components/homesec/const.py`

```python
"""Constants for HomeSec integration."""

from typing import Final

DOMAIN: Final = "homesec"

# Configuration keys
CONF_HOST: Final = "host"
CONF_PORT: Final = "port"
CONF_API_KEY: Final = "api_key"
CONF_VERIFY_SSL: Final = "verify_ssl"

# Default values
DEFAULT_PORT: Final = 8080
DEFAULT_VERIFY_SSL: Final = True
ADDON_HOSTNAME: Final = "localhost"  # Add-on runs on same host as HA

# Motion sensor
DEFAULT_MOTION_RESET_SECONDS: Final = 30

# Platforms (v1 - core functionality only)
# TODO v2: Add "camera", "image", "select" platforms
PLATFORMS: Final = [
    "binary_sensor",
    "sensor",
    "switch",
]

# Entity categories
DIAGNOSTIC_SENSORS: Final = ["health", "last_heartbeat"]

# Update intervals
SCAN_INTERVAL_SECONDS: Final = 30

# Attributes
ATTR_CLIP_ID: Final = "clip_id"
ATTR_CLIP_URL: Final = "clip_url"
ATTR_SNAPSHOT_URL: Final = "snapshot_url"
ATTR_ACTIVITY_TYPE: Final = "activity_type"
ATTR_RISK_LEVEL: Final = "risk_level"
ATTR_SUMMARY: Final = "summary"
ATTR_DETECTED_OBJECTS: Final = "detected_objects"
```

### 4.4 Config Flow

**File:** `homeassistant/integration/custom_components/homesec/config_flow.py`

The config flow automatically detects if the HomeSec add-on is running and offers one-click setup.

```python
"""Config flow for HomeSec integration."""

from __future__ import annotations

import logging
from typing import Any

import aiohttp
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_API_KEY
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    DOMAIN,
    DEFAULT_PORT,
    CONF_VERIFY_SSL,
    DEFAULT_VERIFY_SSL,
    ADDON_HOSTNAME,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_HOST): str,
        vol.Optional(CONF_PORT, default=DEFAULT_PORT): int,
        vol.Optional(CONF_API_KEY): str,
        vol.Optional(CONF_VERIFY_SSL, default=DEFAULT_VERIFY_SSL): bool,
    }
)


async def validate_connection(
    hass: HomeAssistant,
    host: str,
    port: int,
    api_key: str | None = None,
    verify_ssl: bool = True,
) -> dict[str, Any]:
    """Validate the user input allows us to connect."""
    session = async_get_clientsession(hass, verify_ssl=verify_ssl)

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"http://{host}:{port}/api/v1/health"

    async with session.get(url, headers=headers) as response:
        if response.status == 401:
            raise InvalidAuth
        if response.status != 200:
            raise CannotConnect

        data = await response.json()
        return {
            "title": f"HomeSec ({host})",
            "version": data.get("version", "unknown"),
            "cameras": data.get("sources", []),
        }


async def detect_addon(hass: HomeAssistant) -> bool:
    """Check if HomeSec add-on is running."""
    try:
        # Try to connect to add-on at localhost:8080
        await validate_connection(hass, ADDON_HOSTNAME, DEFAULT_PORT, verify_ssl=False)
        return True
    except (CannotConnect, InvalidAuth, Exception):
        return False


class HomesecConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for HomeSec."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._host: str | None = None
        self._port: int = DEFAULT_PORT
        self._api_key: str | None = None
        self._verify_ssl: bool = DEFAULT_VERIFY_SSL
        self._cameras: list[dict] = []
        self._addon_detected: bool = False

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step - check for add-on first."""
        # Check if add-on is running
        if await detect_addon(self.hass):
            self._addon_detected = True
            return await self.async_step_addon()

        # No add-on found, show manual setup
        return await self.async_step_manual(user_input)

    async def async_step_addon(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle add-on auto-discovery confirmation."""
        errors = {}

        if user_input is not None:
            # User confirmed, connect to add-on
            self._host = ADDON_HOSTNAME
            self._port = DEFAULT_PORT
            self._verify_ssl = False

            try:
                info = await validate_connection(
                    self.hass, self._host, self._port, verify_ssl=False
                )
                self._cameras = info.get("cameras", [])
            except CannotConnect:
                errors["base"] = "cannot_connect"
                # Fall back to manual setup
                return await self.async_step_manual()
            except Exception:
                _LOGGER.exception("Unexpected exception connecting to add-on")
                errors["base"] = "unknown"
                return await self.async_step_manual()
            else:
                await self.async_set_unique_id(f"homesec_{self._host}_{self._port}")
                self._abort_if_unique_id_configured()
                return await self.async_step_cameras()

        # Show confirmation form
        return self.async_show_form(
            step_id="addon",
            description_placeholders={
                "addon_url": f"http://{ADDON_HOSTNAME}:{DEFAULT_PORT}",
            },
            errors=errors,
        )

    async def async_step_manual(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle manual setup for standalone HomeSec."""
        errors = {}

        if user_input is not None:
            self._host = user_input[CONF_HOST]
            self._port = user_input.get(CONF_PORT, DEFAULT_PORT)
            self._api_key = user_input.get(CONF_API_KEY)
            self._verify_ssl = user_input.get(CONF_VERIFY_SSL, DEFAULT_VERIFY_SSL)

            try:
                info = await validate_connection(
                    self.hass,
                    self._host,
                    self._port,
                    self._api_key,
                    self._verify_ssl,
                )
                self._cameras = info.get("cameras", [])
            except CannotConnect:
                errors["base"] = "cannot_connect"
            except InvalidAuth:
                errors["base"] = "invalid_auth"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                await self.async_set_unique_id(f"homesec_{self._host}_{self._port}")
                self._abort_if_unique_id_configured()
                return await self.async_step_cameras()

        return self.async_show_form(
            step_id="manual",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
        )

    async def async_step_cameras(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle camera configuration step."""
        if user_input is not None:
            return self.async_create_entry(
                title=f"HomeSec ({'Add-on' if self._addon_detected else self._host})",
                data={
                    CONF_HOST: self._host,
                    CONF_PORT: self._port,
                    CONF_API_KEY: self._api_key,
                    CONF_VERIFY_SSL: self._verify_ssl,
                    "addon": self._addon_detected,
                },
                options={
                    "cameras": user_input.get("cameras", []),
                    "motion_reset_seconds": 30,  # Configurable motion sensor reset
                },
            )

        camera_names = [c["name"] for c in self._cameras]
        schema = vol.Schema(
            {
                vol.Optional("cameras", default=camera_names): vol.All(
                    vol.Coerce(list),
                    [vol.In(camera_names)],
                ),
            }
        )

        return self.async_show_form(
            step_id="cameras",
            data_schema=schema,
            description_placeholders={"camera_count": str(len(self._cameras))},
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> OptionsFlowHandler:
        """Get the options flow for this handler."""
        return OptionsFlowHandler(config_entry)


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for HomeSec."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        coordinator = self.hass.data[DOMAIN][self.config_entry.entry_id]
        camera_names = [c["name"] for c in coordinator.data.get("cameras", [])]
        current_cameras = self.config_entry.options.get("cameras", camera_names)

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional("cameras", default=current_cameras): vol.All(
                        vol.Coerce(list),
                        [vol.In(camera_names)],
                    ),
                    vol.Optional(
                        "scan_interval",
                        default=self.config_entry.options.get("scan_interval", 30),
                    ): vol.All(vol.Coerce(int), vol.Range(min=10, max=300)),
                    vol.Optional(
                        "motion_reset_seconds",
                        default=self.config_entry.options.get("motion_reset_seconds", 30),
                    ): vol.All(vol.Coerce(int), vol.Range(min=5, max=300)),
                }
            ),
        )


class CannotConnect(Exception):
    """Error to indicate we cannot connect."""


class InvalidAuth(Exception):
    """Error to indicate there is invalid auth."""
```

Integration behavior for config changes:

- When users add/update/remove cameras in HA, call the HomeSec API.
- If `restart_required: true`, show a confirmation and invoke `/api/v1/system/restart`
  (or instruct the user to restart the add-on).

### 4.5 Data Coordinator

**File:** `homeassistant/integration/custom_components/homesec/coordinator.py`

```python
"""Data coordinator for HomeSec integration."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

import aiohttp
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.update_coordinator import (
    DataUpdateCoordinator,
    UpdateFailed,
)

from .const import DOMAIN, SCAN_INTERVAL_SECONDS

_LOGGER = logging.getLogger(__name__)


class HomesecCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator to manage HomeSec data updates."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        host: str,
        port: int,
        api_key: str | None = None,
        verify_ssl: bool = True,
    ) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=SCAN_INTERVAL_SECONDS),
        )
        self.host = host
        self.port = port
        self.api_key = api_key
        self.verify_ssl = verify_ssl
        self.config_entry = config_entry
        self._session = async_get_clientsession(hass, verify_ssl=verify_ssl)
        self._event_unsubs: list[callable] = []
        self._motion_timers: dict[str, callable] = {}  # Camera -> cancel callback
        self._config_version: int = 0  # Track for optimistic concurrency

    @property
    def base_url(self) -> str:
        """Return the base URL for the API."""
        return f"http://{self.host}:{self.port}/api/v1"

    @property
    def _headers(self) -> dict[str, str]:
        """Return headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch data from HomeSec API."""
        try:
            async with asyncio.timeout(10):
                # Fetch health status
                async with self._session.get(
                    f"{self.base_url}/health",
                    headers=self._headers,
                ) as response:
                    response.raise_for_status()
                    health = await response.json()

                # Fetch cameras
                async with self._session.get(
                    f"{self.base_url}/cameras",
                    headers=self._headers,
                ) as response:
                    response.raise_for_status()
                    cameras_data = await response.json()

                # Fetch config for version tracking
                async with self._session.get(
                    f"{self.base_url}/config",
                    headers=self._headers,
                ) as response:
                    response.raise_for_status()
                    config_data = await response.json()
                    self._config_version = config_data.get("config_version", 0)

                return {
                    "health": health,
                    "cameras": cameras_data.get("cameras", []),
                    "config_version": self._config_version,
                    "connected": True,
                }

        except asyncio.TimeoutError as err:
            raise UpdateFailed("Timeout connecting to HomeSec") from err
        except aiohttp.ClientError as err:
            raise UpdateFailed(f"Error communicating with HomeSec: {err}") from err

    async def async_subscribe_events(self) -> None:
        """Subscribe to HomeSec events fired via HA Events API."""
        if self._event_unsubs:
            return

        from homeassistant.core import Event
        from homeassistant.helpers.event import async_call_later

        async def _on_alert(event: Event) -> None:
            """Handle homesec_alert event with motion timer."""
            _LOGGER.debug("Received homesec_alert event: %s", event.data)
            camera = event.data.get("camera")
            if not camera:
                return

            # Store latest alert data
            self.data.setdefault("latest_alerts", {})[camera] = event.data

            # Set motion active for this camera
            self.data.setdefault("motion_active", {})[camera] = True

            # Cancel any existing reset timer for this camera
            cancel_key = f"motion_reset_{camera}"
            if cancel_key in self._motion_timers:
                self._motion_timers[cancel_key]()  # Cancel existing timer

            # Schedule motion reset after configured duration
            reset_seconds = self.config_entry.options.get("motion_reset_seconds", 30)

            async def reset_motion(_now: datetime) -> None:
                """Reset motion sensor after timeout."""
                self.data.setdefault("motion_active", {})[camera] = False
                self._motion_timers.pop(cancel_key, None)
                self.async_set_updated_data(self.data)

            self._motion_timers[cancel_key] = async_call_later(
                self.hass, reset_seconds, reset_motion
            )

            # Trigger immediate data refresh
            self.async_set_updated_data(self.data)

        async def _on_camera_health(event: Event) -> None:
            """Handle homesec_camera_health event."""
            _LOGGER.debug("Received homesec_camera_health event: %s", event.data)
            await self.async_request_refresh()

        async def _on_clip_recorded(event: Event) -> None:
            """Handle homesec_clip_recorded event."""
            _LOGGER.debug("Received homesec_clip_recorded event: %s", event.data)

        # Subscribe to HomeSec events
        self._event_unsubs.append(
            self.hass.bus.async_listen("homesec_alert", _on_alert)
        )
        self._event_unsubs.append(
            self.hass.bus.async_listen("homesec_camera_health", _on_camera_health)
        )
        self._event_unsubs.append(
            self.hass.bus.async_listen("homesec_clip_recorded", _on_clip_recorded)
        )
        _LOGGER.info("Subscribed to HomeSec events")

    async def async_unsubscribe_events(self) -> None:
        """Unsubscribe from HomeSec events."""
        for unsub in self._event_unsubs:
            unsub()
        self._event_unsubs.clear()

    # API Methods

    async def async_get_cameras(self) -> list[dict]:
        """Get list of cameras."""
        async with self._session.get(
            f"{self.base_url}/cameras",
            headers=self._headers,
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return data.get("cameras", [])

    async def async_add_camera(
        self,
        name: str,
        source_backend: str,
        source_config: dict,
    ) -> dict:
        """Add a new camera. Uses optimistic concurrency with config_version."""
        payload = {
            "name": name,
            "source_backend": source_backend,
            "source_config": source_config,
            "config_version": self._config_version,
        }
        async with self._session.post(
            f"{self.base_url}/cameras",
            headers=self._headers,
            json=payload,
        ) as response:
            if response.status == 409:
                raise ConfigVersionConflict("Config was modified, please refresh")
            response.raise_for_status()
            result = await response.json()
            self._config_version = result.get("config_version", self._config_version)
            return result

    async def async_update_camera(
        self,
        camera_name: str,
        source_config: dict | None = None,
    ) -> dict:
        """Update camera source configuration. Uses optimistic concurrency."""
        payload = {"config_version": self._config_version}
        if source_config is not None:
            payload["source_config"] = source_config

        async with self._session.put(
            f"{self.base_url}/cameras/{camera_name}",
            headers=self._headers,
            json=payload,
        ) as response:
            if response.status == 409:
                raise ConfigVersionConflict("Config was modified, please refresh")
            response.raise_for_status()
            result = await response.json()
            self._config_version = result.get("config_version", self._config_version)
            return result

    async def async_delete_camera(self, camera_name: str) -> None:
        """Delete a camera. Uses optimistic concurrency."""
        async with self._session.delete(
            f"{self.base_url}/cameras/{camera_name}",
            headers=self._headers,
            params={"config_version": self._config_version},
        ) as response:
            if response.status == 409:
                raise ConfigVersionConflict("Config was modified, please refresh")
            response.raise_for_status()
            result = await response.json()
            self._config_version = result.get("config_version", self._config_version)

    async def async_set_camera_enabled(self, camera_name: str, enabled: bool) -> dict:
        """Enable or disable a camera (stops RTSP connection when disabled)."""
        return await self.async_update_camera(
            camera_name,
            source_config={"enabled": enabled},
        )


class ConfigVersionConflict(Exception):
    """Raised when config_version is stale (409 Conflict)."""

    async def async_test_camera(self, camera_name: str) -> dict:
        """Test camera connection."""
        async with self._session.post(
            f"{self.base_url}/cameras/{camera_name}/test",
            headers=self._headers,
        ) as response:
            response.raise_for_status()
            return await response.json()
```

### 4.6 Integration Setup

**File:** `homeassistant/integration/custom_components/homesec/__init__.py`

```python
"""HomeSec integration for Home Assistant."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant

from .const import DOMAIN, CONF_VERIFY_SSL, DEFAULT_VERIFY_SSL
from .coordinator import HomesecCoordinator

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [
    Platform.BINARY_SENSOR,
    Platform.SENSOR,
    Platform.SWITCH,
]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up HomeSec from a config entry."""
    coordinator = HomesecCoordinator(
        hass,
        entry,
        host=entry.data[CONF_HOST],
        port=entry.data[CONF_PORT],
        api_key=entry.data.get(CONF_API_KEY),
        verify_ssl=entry.data.get(CONF_VERIFY_SSL, DEFAULT_VERIFY_SSL),
    )

    # Fetch initial data
    await coordinator.async_config_entry_first_refresh()

    # Subscribe to HomeSec events (via HA Events API)
    await coordinator.async_subscribe_events()

    # Store coordinator
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = coordinator

    # Set up platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register update listener for options changes
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # Unsubscribe from events
    coordinator: HomesecCoordinator = hass.data[DOMAIN][entry.entry_id]
    await coordinator.async_unsubscribe_events()

    # Cancel any pending motion timers
    for cancel in coordinator._motion_timers.values():
        cancel()
    coordinator._motion_timers.clear()

    # Unload platforms
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)

    return unload_ok


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)
```

### 4.7 Base Entity

**File:** `homeassistant/integration/custom_components/homesec/entity.py`

```python
"""Base entity for HomeSec integration."""

from __future__ import annotations

from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import HomesecCoordinator


class HomesecEntity(CoordinatorEntity[HomesecCoordinator]):
    """Base class for HomeSec entities."""

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: HomesecCoordinator,
        camera_name: str,
    ) -> None:
        """Initialize the entity."""
        super().__init__(coordinator)
        self._camera_name = camera_name

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info for this entity."""
        return DeviceInfo(
            identifiers={(DOMAIN, self._camera_name)},
            name=f"HomeSec {self._camera_name}",
            manufacturer="HomeSec",
            model="AI Security Camera",
            sw_version=self.coordinator.data.get("health", {}).get("version", "unknown"),
            via_device=(DOMAIN, "homesec_hub"),
        )

    def _get_camera_data(self) -> dict | None:
        """Get camera data from coordinator."""
        cameras = self.coordinator.data.get("cameras", [])
        for camera in cameras:
            if camera.get("name") == self._camera_name:
                return camera
        return None

    def _is_motion_active(self) -> bool:
        """Check if motion is currently active for this camera."""
        return self.coordinator.data.get("motion_active", {}).get(self._camera_name, False)

    def _get_latest_alert(self) -> dict | None:
        """Get the latest alert for this camera."""
        return self.coordinator.data.get("latest_alerts", {}).get(self._camera_name)
```

### 4.8 Entity Platforms

**File:** `homeassistant/integration/custom_components/homesec/sensor.py`

```python
"""Sensor platform for HomeSec integration."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import (
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, ATTR_ACTIVITY_TYPE, ATTR_RISK_LEVEL, ATTR_SUMMARY
from .coordinator import HomesecCoordinator
from .entity import HomesecEntity

CAMERA_SENSORS: tuple[SensorEntityDescription, ...] = (
    SensorEntityDescription(
        key="last_activity",
        name="Last Activity",
        icon="mdi:motion-sensor",
    ),
    SensorEntityDescription(
        key="risk_level",
        name="Risk Level",
        icon="mdi:shield-alert",
    ),
    SensorEntityDescription(
        key="health",
        name="Health",
        icon="mdi:heart-pulse",
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    SensorEntityDescription(
        key="clips_today",
        name="Clips Today",
        icon="mdi:filmstrip-box",
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up HomeSec sensors."""
    coordinator: HomesecCoordinator = hass.data[DOMAIN][entry.entry_id]

    entities: list[SensorEntity] = []

    for camera in coordinator.data.get("cameras", []):
        camera_name = camera["name"]
        for description in CAMERA_SENSORS:
            entities.append(
                HomesecCameraSensor(coordinator, camera_name, description)
            )

    async_add_entities(entities)


class HomesecCameraSensor(HomesecEntity, SensorEntity):
    """Representation of a HomeSec camera sensor."""

    def __init__(
        self,
        coordinator: HomesecCoordinator,
        camera_name: str,
        description: SensorEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, camera_name)
        self.entity_description = description
        self._attr_unique_id = f"{camera_name}_{description.key}"

    @property
    def native_value(self) -> str | int | None:
        """Return the state of the sensor."""
        camera = self._get_camera_data()
        if not camera:
            return None

        key = self.entity_description.key

        if key == "last_activity":
            return camera.get("last_activity", {}).get("activity_type")
        elif key == "risk_level":
            return camera.get("last_activity", {}).get("risk_level")
        elif key == "health":
            return "healthy" if camera.get("healthy") else "unhealthy"
        elif key == "clips_today":
            return camera.get("stats", {}).get("clips_today", 0)

        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional state attributes."""
        camera = self._get_camera_data()
        if not camera:
            return {}

        key = self.entity_description.key
        attrs = {}

        if key == "last_activity":
            activity = camera.get("last_activity", {})
            attrs[ATTR_ACTIVITY_TYPE] = activity.get("activity_type")
            attrs[ATTR_RISK_LEVEL] = activity.get("risk_level")
            attrs[ATTR_SUMMARY] = activity.get("summary")
            attrs["clip_id"] = activity.get("clip_id")
            attrs["view_url"] = activity.get("view_url")
            attrs["timestamp"] = activity.get("timestamp")

        return attrs
```

**File:** `homeassistant/integration/custom_components/homesec/binary_sensor.py`

```python
"""Binary sensor platform for HomeSec integration."""

from __future__ import annotations

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
    BinarySensorEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .coordinator import HomesecCoordinator
from .entity import HomesecEntity

CAMERA_BINARY_SENSORS: tuple[BinarySensorEntityDescription, ...] = (
    BinarySensorEntityDescription(
        key="motion",
        name="Motion",
        device_class=BinarySensorDeviceClass.MOTION,
    ),
    BinarySensorEntityDescription(
        key="person",
        name="Person Detected",
        device_class=BinarySensorDeviceClass.OCCUPANCY,
    ),
    BinarySensorEntityDescription(
        key="online",
        name="Online",
        device_class=BinarySensorDeviceClass.CONNECTIVITY,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up HomeSec binary sensors."""
    coordinator: HomesecCoordinator = hass.data[DOMAIN][entry.entry_id]

    entities: list[BinarySensorEntity] = []

    for camera in coordinator.data.get("cameras", []):
        camera_name = camera["name"]
        for description in CAMERA_BINARY_SENSORS:
            entities.append(
                HomesecCameraBinarySensor(coordinator, camera_name, description)
            )

    async_add_entities(entities)


class HomesecCameraBinarySensor(HomesecEntity, BinarySensorEntity):
    """Representation of a HomeSec camera binary sensor."""

    def __init__(
        self,
        coordinator: HomesecCoordinator,
        camera_name: str,
        description: BinarySensorEntityDescription,
    ) -> None:
        """Initialize the binary sensor."""
        super().__init__(coordinator, camera_name)
        self.entity_description = description
        self._attr_unique_id = f"{camera_name}_{description.key}"

    @property
    def is_on(self) -> bool | None:
        """Return true if the binary sensor is on."""
        camera = self._get_camera_data()
        if not camera:
            return None

        key = self.entity_description.key

        if key == "motion":
            # Use timer-based motion state from coordinator
            return self._is_motion_active()
        elif key == "person":
            # Person detected based on latest alert
            alert = self._get_latest_alert()
            if alert and self._is_motion_active():
                return alert.get("activity_type") == "person"
            return False
        elif key == "online":
            return camera.get("healthy", False)

        return None
```

**File:** `homeassistant/integration/custom_components/homesec/switch.py`

```python
"""Switch platform for HomeSec integration.

Prerequisite: Add `enabled: bool = True` field to CameraConfig in
src/homesec/models/config.py. The switch toggles this field via the API.
"""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.switch import SwitchEntity, SwitchEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)
from .coordinator import HomesecCoordinator
from .entity import HomesecEntity

CAMERA_SWITCHES: tuple[SwitchEntityDescription, ...] = (
    SwitchEntityDescription(
        key="enabled",
        name="Enabled",
        icon="mdi:video",
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up HomeSec switches."""
    coordinator: HomesecCoordinator = hass.data[DOMAIN][entry.entry_id]

    entities: list[SwitchEntity] = []

    for camera in coordinator.data.get("cameras", []):
        camera_name = camera["name"]
        for description in CAMERA_SWITCHES:
            entities.append(
                HomesecCameraSwitch(coordinator, camera_name, description)
            )

    async_add_entities(entities)


class HomesecCameraSwitch(HomesecEntity, SwitchEntity):
    """Representation of a HomeSec camera switch."""

    def __init__(
        self,
        coordinator: HomesecCoordinator,
        camera_name: str,
        description: SwitchEntityDescription,
    ) -> None:
        """Initialize the switch."""
        super().__init__(coordinator, camera_name)
        self.entity_description = description
        self._attr_unique_id = f"{camera_name}_{description.key}"

    @property
    def is_on(self) -> bool | None:
        """Return true if the switch is on."""
        camera = self._get_camera_data()
        if not camera:
            return None

        key = self.entity_description.key

        if key == "enabled":
            # NOTE: Requires `enabled: bool` field on CameraConfig (see prerequisite below)
            return camera.get("enabled", True)

        return None

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the switch on (enable camera, starts RTSP connection)."""
        try:
            await self.coordinator.async_set_camera_enabled(self._camera_name, True)
            await self.coordinator.async_request_refresh()
        except Exception as err:
            _LOGGER.error("Failed to enable camera %s: %s", self._camera_name, err)
            raise HomeAssistantError(f"Failed to enable camera: {err}") from err

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the switch off (disable camera, stops RTSP connection)."""
        try:
            await self.coordinator.async_set_camera_enabled(self._camera_name, False)
            await self.coordinator.async_request_refresh()
        except Exception as err:
            _LOGGER.error("Failed to disable camera %s: %s", self._camera_name, err)
            raise HomeAssistantError(f"Failed to disable camera: {err}") from err
```

### 4.9 Services

**File:** `homeassistant/integration/custom_components/homesec/services.yaml`

```yaml
add_camera:
  name: Add Camera
  description: Add a new camera to HomeSec
  fields:
    name:
      name: Name
      description: Unique identifier for the camera
      required: true
      example: "front_door"
      selector:
        text:
    source_backend:
      name: Source Backend
      description: Camera source backend type
      required: true
      example: "rtsp"
      selector:
        select:
          options:
            - "rtsp"
            - "ftp"
            - "local_folder"
    rtsp_url:
      name: RTSP URL
      description: RTSP stream URL (for rtsp backend)
      example: "rtsp://192.168.1.100:554/stream"
      selector:
        text:

remove_camera:
  name: Remove Camera
  description: Remove a camera from HomeSec
  target:
    device:
      integration: homesec

# TODO v2: Add these services (require additional API endpoints)
# set_alert_policy:
#   name: Set Alert Policy
#   description: Configure alert policy override for a camera
#   (requires PUT /api/v1/config/alert_policy endpoint)
#
# test_camera:
#   name: Test Camera
#   description: Test camera connection
#   (requires POST /api/v1/cameras/{name}/test endpoint)
```

### 4.10 Translations

**File:** `homeassistant/integration/custom_components/homesec/translations/en.json`

```json
{
  "config": {
    "step": {
      "user": {
        "title": "Connect to HomeSec",
        "description": "Enter the connection details for your HomeSec instance.",
        "data": {
          "host": "Host",
          "port": "Port",
          "api_key": "API Key (optional)",
          "verify_ssl": "Verify SSL certificate"
        }
      },
      "cameras": {
        "title": "Select Cameras",
        "description": "Found {camera_count} cameras. Select which ones to add to Home Assistant.",
        "data": {
          "cameras": "Cameras"
        }
      }
    },
    "error": {
      "cannot_connect": "Failed to connect to HomeSec",
      "invalid_auth": "Invalid API key",
      "unknown": "Unexpected error"
    },
    "abort": {
      "already_configured": "This HomeSec instance is already configured"
    }
  },
  "options": {
    "step": {
      "init": {
        "title": "HomeSec Options",
        "data": {
          "cameras": "Enabled cameras",
          "scan_interval": "Update interval (seconds)"
        }
      }
    }
  },
  "entity": {
    "sensor": {
      "last_activity": {
        "name": "Last Activity"
      },
      "risk_level": {
        "name": "Risk Level"
      },
      "health": {
        "name": "Health"
      },
      "clips_today": {
        "name": "Clips Today"
      }
    },
    "binary_sensor": {
      "motion": {
        "name": "Motion"
      },
      "person": {
        "name": "Person Detected"
      },
      "online": {
        "name": "Online"
      }
    }
  },
  "services": {
    "add_camera": {
      "name": "Add Camera",
      "description": "Add a new camera to HomeSec"
    },
    "remove_camera": {
      "name": "Remove Camera",
      "description": "Remove a camera from HomeSec"
    }
  }
}
```

### 4.11 Acceptance Criteria

- [ ] Config flow auto-detects add-on and connects to HomeSec
- [ ] Config flow falls back to manual setup for standalone users
- [ ] All entity platforms create entities correctly
- [ ] DataUpdateCoordinator fetches data at correct intervals
- [ ] HA Events subscription triggers refresh on alerts (homesec_alert, homesec_camera_health)
- [ ] Motion sensor auto-resets after configurable timeout (default 30s)
- [ ] Camera switch enables/disables camera (requires `enabled` field prerequisite)
- [ ] Config version tracking prevents conflicts (409 on stale version)
- [ ] Services work correctly (add_camera, remove_camera)
- [ ] Options flow allows reconfiguration
- [ ] Diagnostics provide useful debug information
- [ ] All strings are translatable
- [ ] HACS installation works
- [ ] Error handling shows user-friendly messages on API failures

---

## Phase 5: Advanced Features

**Goal:** Premium features for power users.

**Estimated Effort:** 5-7 days

### 5.1 Custom Lovelace Card (Optional)

**File:** `homeassistant/integration/custom_components/homesec/www/homesec-camera-card.js`

```javascript
class HomesecCameraCard extends HTMLElement {
  set hass(hass) {
    if (!this.content) {
      this.innerHTML = `
        <ha-card header="HomeSec Camera">
          <div class="card-content">
            <div class="camera-container">
              <img id="camera-image" style="width: 100%;" />
            </div>
            <div class="detections">
              <span class="badge" id="motion-badge">Motion</span>
              <span class="badge" id="person-badge">Person</span>
            </div>
            <div class="info">
              <div id="activity"></div>
              <div id="risk"></div>
            </div>
          </div>
        </ha-card>
        <style>
          .badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin-right: 8px;
          }
          .badge.active {
            background-color: var(--primary-color);
            color: white;
          }
          .badge.inactive {
            background-color: var(--disabled-color);
            color: var(--disabled-text-color);
          }
        </style>
      `;
      this.content = true;
    }

    const config = this._config;
    const cameraEntity = config.entity;
    const state = hass.states[cameraEntity];

    if (state) {
      // Update camera image
      const img = this.querySelector("#camera-image");
      img.src = `/api/camera_proxy/${cameraEntity}?token=${state.attributes.access_token}`;

      // Update detection badges
      const motionEntity = cameraEntity.replace("camera.", "binary_sensor.") + "_motion";
      const personEntity = cameraEntity.replace("camera.", "binary_sensor.") + "_person";

      const motionBadge = this.querySelector("#motion-badge");
      const personBadge = this.querySelector("#person-badge");

      motionBadge.className = `badge ${hass.states[motionEntity]?.state === "on" ? "active" : "inactive"}`;
      personBadge.className = `badge ${hass.states[personEntity]?.state === "on" ? "active" : "inactive"}`;

      // Update activity info
      const activityEntity = cameraEntity.replace("camera.", "sensor.") + "_last_activity";
      const activityState = hass.states[activityEntity];

      if (activityState) {
        this.querySelector("#activity").textContent = `Activity: ${activityState.state}`;
        this.querySelector("#risk").textContent = `Risk: ${activityState.attributes.risk_level || "N/A"}`;
      }
    }
  }

  setConfig(config) {
    if (!config.entity) {
      throw new Error("You need to define an entity");
    }
    this._config = config;
  }

  getCardSize() {
    return 4;
  }
}

customElements.define("homesec-camera-card", HomesecCameraCard);
```

### 5.2 Event Timeline Panel (Optional)

Create a custom panel for viewing event history with timeline visualization.

### 5.3 Acceptance Criteria

- [ ] Snapshot images work
- [ ] Custom Lovelace card displays detection overlays
- [ ] Event timeline shows historical data

---

## Testing Strategy

All new tests must include Given/When/Then comments (per `TESTING.md`). Prefer behavioral
style tests that assert observable outcomes, not internal state.

### Unit Tests

```
tests/
├── unit/
│   ├── api/
│   │   ├── test_routes_cameras.py
│   │   ├── test_routes_clips.py
│   │   └── test_routes_events.py
│   ├── plugins/
│   │   └── notifiers/
│   │       └── test_mqtt_discovery.py
│   └── models/
│       └── test_config_mqtt.py
├── integration/
│   ├── test_mqtt_ha_integration.py
│   ├── test_api_full_flow.py
│   └── test_ha_addon.py
└── e2e/
    └── test_ha_config_flow.py
```

### Integration Tests

1. **API Tests:**
   - Full CRUD cycle for cameras
   - Authentication flows
   - Returns 503 when Postgres is unavailable

3. **HA Notifier Tests:**
   - Supervisor mode detection (SUPERVISOR_TOKEN)
   - Standalone mode requires url_env + token_env
   - Events fire correctly: homesec_alert, homesec_camera_health, homesec_clip_recorded
   - HA unreachable does not stall clip processing (best-effort)

4. **Add-on Tests:**
   - Installation on HA OS
   - SUPERVISOR_TOKEN injected automatically
   - HA Events API works (homesec_alert reaches HA)
   - Ingress access to API

### Manual Testing Checklist

- [ ] Install add-on from repository
- [ ] Configure via HA UI
- [ ] Add/remove cameras
- [ ] Verify entities update on alerts (via HA Events)
- [ ] Test automations with HomeSec triggers
- [ ] Verify snapshots and links (if enabled) in dashboard

### Optional: MQTT Discovery Tests (Phase 1)

If user enables MQTT Discovery:
- [ ] Publish discovery → verify entities appear in HA
- [ ] HA restart → verify discovery republishes
- [ ] Camera add → verify new entities created

---

## Migration Guide

### From Standalone to Add-on

1. Export current `config.yaml`
2. Install HomeSec add-on
3. Copy config to `/config/homesec/config.yaml`
4. Create `/data/overrides.yaml` for HA-managed config
5. Update database URL if using external Postgres
6. Start add-on

### From MQTT-only to Full Integration

1. Keep existing MQTT configuration
2. Install custom integration via HACS
3. Configure integration with HomeSec URL
4. Entities will be created alongside MQTT entities
5. Optionally disable MQTT discovery to avoid duplicates

---

## Appendix: File Change Summary

### Phase 2: REST API
- `src/homesec/api/` - New package (server.py, routes/*, dependencies.py)
- `src/homesec/api/routes/stats.py` - Stats endpoint for hub entities
- `src/homesec/api/routes/health.py` - Health endpoint with response schema
- `src/homesec/config/manager.py` - Config persistence + restart signaling
- `src/homesec/models/config.py` - Add FastAPIServerConfig
- `src/homesec/config/loader.py` - Support multiple YAML files + merge semantics
- `src/homesec/cli.py` - Accept repeated `--config` flags (order matters)
- `src/homesec/app.py` - Integrate API server
- `pyproject.toml` - Add fastapi, uvicorn dependencies

### Phase 2.5: Home Assistant Notifier
- `src/homesec/models/config.py` - Add HomeAssistantNotifierConfig
- `src/homesec/plugins/notifiers/home_assistant.py` - New file (HA Events API notifier)
- `config/example.yaml` - Add home_assistant notifier example
- `tests/unit/plugins/notifiers/test_home_assistant.py` - New tests

### Phase 4: Integration (in `homeassistant/integration/`)
- `homeassistant/integration/hacs.json` - HACS configuration
- `homeassistant/integration/custom_components/homesec/__init__.py` - Setup entry
- `homeassistant/integration/custom_components/homesec/manifest.json` - Metadata
- `homeassistant/integration/custom_components/homesec/const.py` - Constants
- `homeassistant/integration/custom_components/homesec/config_flow.py` - Add-on auto-discovery
- `homeassistant/integration/custom_components/homesec/coordinator.py` - Data + event subscriptions + motion timers
- `homeassistant/integration/custom_components/homesec/entity.py` - Base entity class
- `homeassistant/integration/custom_components/homesec/sensor.py` - Sensors
- `homeassistant/integration/custom_components/homesec/binary_sensor.py` - Motion (30s auto-reset)
- `homeassistant/integration/custom_components/homesec/switch.py` - Camera enable/disable (stops RTSP)
- `homeassistant/integration/custom_components/homesec/services.yaml` - Service definitions
- `homeassistant/integration/custom_components/homesec/translations/en.json` - Strings

### Phase 3: Add-on (in `homeassistant/addon/`)
- `repository.json` - Add-on repository manifest (at repo root, points to `homeassistant/addon/homesec/`)
- `homeassistant/addon/homesec/config.yaml` - Add-on manifest (homeassistant_api: true)
- `homeassistant/addon/homesec/Dockerfile` - Container build with PostgreSQL 16
- `homeassistant/addon/homesec/rootfs/` - s6-overlay services (postgres-init, postgres, homesec)
- `homeassistant/addon/homesec/rootfs/etc/nginx/` - Ingress config

### Phase 1: MQTT Discovery (Optional)
- `src/homesec/models/config.py` - Add MQTTDiscoveryConfig
- `src/homesec/plugins/notifiers/mqtt.py` - Enhance with discovery
- `src/homesec/plugins/notifiers/mqtt_discovery.py` - New file
- `src/homesec/app.py` - Register cameras with notifier
- `config/example.yaml` - Add discovery example
- `tests/unit/plugins/notifiers/test_mqtt_discovery.py` - New tests

### Phase 5: Advanced
- `homeassistant/integration/custom_components/homesec/www/` - Lovelace cards

---

## Timeline Summary

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 2: REST API | 5-7 days | None |
| Phase 2.5: HA Notifier | 2-3 days | None |
| Phase 4: Integration | 7-10 days | Phase 2, 2.5 |
| Phase 3: Add-on | 3-4 days | Phase 2, 2.5 |
| Phase 1: MQTT Discovery (optional) | 2-3 days | None |
| Phase 5: Advanced | 5-7 days | Phase 4 |

**Total: 20-27 days** (excluding optional MQTT Discovery)

Execution order: Phase 2 → Phase 2.5 → Phase 4 → Phase 3 → Phase 5. Phase 1 (MQTT Discovery) is optional.

Key benefit: No MQTT broker required. Add-on users get zero-config real-time events via `SUPERVISOR_TOKEN`.
