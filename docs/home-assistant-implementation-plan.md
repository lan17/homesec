# HomeSec + Home Assistant: Detailed Implementation Plan

This document provides a comprehensive, step-by-step implementation plan for integrating HomeSec with Home Assistant. Each phase includes specific file changes, code examples, testing strategies, and acceptance criteria.

---

## Table of Contents

1. [Phase 1: MQTT Discovery Enhancement](#phase-1-mqtt-discovery-enhancement)
2. [Phase 2: REST API for Configuration](#phase-2-rest-api-for-configuration)
3. [Phase 3: Home Assistant Add-on](#phase-3-home-assistant-add-on)
4. [Phase 4: Native Home Assistant Integration](#phase-4-native-home-assistant-integration)
5. [Phase 5: Advanced Features](#phase-5-advanced-features)
6. [Testing Strategy](#testing-strategy)
7. [Migration Guide](#migration-guide)

---

## Phase 1: MQTT Discovery Enhancement

**Goal:** Auto-create Home Assistant entities from HomeSec without requiring manual HA configuration.

**Estimated Effort:** 2-3 days

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
    username: str | None = None
    password_env: str | None = None
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

Extend the existing MQTT notifier to support discovery:

```python
"""Enhanced MQTT Notifier with Home Assistant Discovery support."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

import aiomqtt

from homesec.interfaces import Notifier
from homesec.models.alert import Alert
from homesec.models.config import MQTTConfig
from homesec.plugins.registry import plugin, PluginType

from .mqtt_discovery import MQTTDiscoveryBuilder

if TYPE_CHECKING:
    from homesec.models.clip import Clip

logger = logging.getLogger(__name__)


@plugin(plugin_type=PluginType.NOTIFIER, name="mqtt")
class MQTTNotifier(Notifier):
    """MQTT notifier with Home Assistant discovery support."""

    config_cls = MQTTConfig

    def __init__(self, config: MQTTConfig, version: str = "1.0.0"):
        self.config = config
        self.version = version
        self._client: aiomqtt.Client | None = None
        self._discovery_builder: MQTTDiscoveryBuilder | None = None
        self._cameras: set[str] = set()
        self._discovery_published: bool = False

        if config.discovery.enabled:
            self._discovery_builder = MQTTDiscoveryBuilder(
                config.discovery, version
            )

    async def start(self) -> None:
        """Start the MQTT client and publish discovery if enabled."""
        self._client = aiomqtt.Client(
            hostname=self.config.host,
            port=self.config.port,
            username=self.config.username,
            password=self._get_password(),
        )
        await self._client.__aenter__()

        if self.config.discovery.enabled and self.config.discovery.subscribe_to_birth:
            # Subscribe to HA birth topic to republish discovery on HA restart
            await self._client.subscribe(self.config.discovery.birth_topic)
            asyncio.create_task(self._listen_for_birth())

    async def _listen_for_birth(self) -> None:
        """Listen for Home Assistant birth messages to republish discovery."""
        async for message in self._client.messages:
            if message.topic.matches(self.config.discovery.birth_topic):
                if message.payload.decode() == "online":
                    logger.info("Home Assistant came online, republishing discovery")
                    await self._publish_discovery()

    async def register_camera(self, camera_name: str) -> None:
        """Register a camera for discovery."""
        self._cameras.add(camera_name)
        if self.config.discovery.enabled and self._discovery_published:
            # Publish discovery for new camera immediately
            await self._publish_camera_discovery(camera_name)

    async def _publish_discovery(self) -> None:
        """Publish all discovery messages."""
        if not self._discovery_builder or not self._client:
            return

        # Publish hub entities
        for entity in self._discovery_builder.build_hub_entities():
            topic = self._discovery_builder.get_discovery_topic(entity)
            payload = self._discovery_builder.get_discovery_payload(entity)
            await self._client.publish(topic, payload, retain=True)
            logger.debug(f"Published discovery: {topic}")

        # Publish camera entities
        for camera_name in self._cameras:
            await self._publish_camera_discovery(camera_name)

        self._discovery_published = True
        logger.info(f"Published MQTT discovery for {len(self._cameras)} cameras")

    async def _publish_camera_discovery(self, camera_name: str) -> None:
        """Publish discovery messages for a single camera."""
        if not self._discovery_builder or not self._client:
            return

        for entity in self._discovery_builder.build_camera_entities(camera_name):
            topic = self._discovery_builder.get_discovery_topic(entity)
            payload = self._discovery_builder.get_discovery_payload(entity)
            await self._client.publish(topic, payload, retain=True)

    async def notify(self, alert: Alert) -> None:
        """Send alert and update entity states."""
        if not self._client:
            raise RuntimeError("MQTT client not started")

        camera = alert.camera_name
        base_topic = f"homesec/{camera}"

        # Original alert topic (backwards compatible)
        alert_topic = self.config.topic_template.format(camera_name=camera)
        await self._client.publish(
            alert_topic,
            alert.model_dump_json(),
            qos=self.config.qos,
            retain=self.config.retain,
        )

        # If discovery is enabled, also publish to state topics
        if self.config.discovery.enabled:
            # Motion state
            await self._client.publish(
                f"{base_topic}/motion",
                "ON",
                retain=True,
            )

            # Person detection (if detected)
            if alert.analysis and "person" in (alert.analysis.detected_objects or []):
                await self._client.publish(
                    f"{base_topic}/person",
                    "ON",
                    retain=True,
                )

            # Activity details
            activity_payload = {
                "activity_type": alert.activity_type or "unknown",
                "summary": alert.summary,
                "risk_level": alert.risk_level.value if alert.risk_level else None,
                "timestamp": alert.ts.isoformat(),
                "clip_id": alert.clip_id,
            }
            await self._client.publish(
                f"{base_topic}/activity",
                json.dumps(activity_payload),
                retain=True,
            )

            # Risk level
            await self._client.publish(
                f"{base_topic}/risk",
                alert.risk_level.value if alert.risk_level else "UNKNOWN",
                retain=True,
            )

            # Clip info
            clip_payload = {
                "clip_id": alert.clip_id,
                "view_url": alert.view_url,
                "storage_uri": alert.storage_uri,
                "timestamp": alert.ts.isoformat(),
            }
            await self._client.publish(
                f"{base_topic}/clip",
                json.dumps(clip_payload),
                retain=True,
            )

            # Device trigger for automations
            await self._client.publish(
                f"{base_topic}/alert",
                json.dumps({"event_type": "alert", **activity_payload}),
            )

            # Schedule motion reset after 30 seconds
            asyncio.create_task(self._reset_motion_state(camera))

    async def _reset_motion_state(self, camera_name: str, delay: float = 30.0) -> None:
        """Reset motion binary sensor after delay."""
        await asyncio.sleep(delay)
        if self._client:
            await self._client.publish(
                f"homesec/{camera_name}/motion",
                "OFF",
                retain=True,
            )
            await self._client.publish(
                f"homesec/{camera_name}/person",
                "OFF",
                retain=True,
            )

    async def publish_health(self, camera_name: str, health: str) -> None:
        """Publish camera health state."""
        if self._client and self.config.discovery.enabled:
            await self._client.publish(
                f"homesec/{camera_name}/health",
                health,
                retain=True,
            )

    async def publish_hub_stats(self, clips_today: int, alerts_today: int) -> None:
        """Publish hub statistics."""
        if self._client and self.config.discovery.enabled:
            await self._client.publish(
                "homesec/hub/status",
                "online",
                retain=True,
            )
            await self._client.publish(
                "homesec/hub/stats",
                json.dumps({
                    "clips_today": clips_today,
                    "alerts_today": alerts_today,
                }),
                retain=True,
            )

    async def stop(self) -> None:
        """Stop the MQTT client."""
        if self._client:
            # Publish offline status
            if self.config.discovery.enabled:
                await self._client.publish(
                    "homesec/hub/status",
                    "offline",
                    retain=True,
                )
            await self._client.__aexit__(None, None, None)
            self._client = None
```

### 1.4 Application Integration

**File:** `src/homesec/app.py`

Update the application to register cameras with the MQTT notifier:

```python
# In HomesecApp.start() method, after initializing sources:

# Register cameras with MQTT notifier for discovery
for notifier in self.notifiers:
    if hasattr(notifier, 'register_camera'):
        for source in self.sources:
            await notifier.register_camera(source.camera_name)

        # Publish initial discovery
        if hasattr(notifier, '_publish_discovery'):
            await notifier._publish_discovery()
```

### 1.5 Configuration Example

**File:** `config/example.yaml` (add section)

```yaml
notifiers:
  - type: mqtt
    host: localhost
    port: 1883
    username: homeassistant
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

from .routes import cameras, clips, config, events, health, websocket

if TYPE_CHECKING:
    from homesec.app import HomesecApp

logger = logging.getLogger(__name__)


def create_app(homesec_app: HomesecApp) -> FastAPI:
    """Create the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Store reference to HomeSec app
        app.state.homesec = homesec_app
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
    app.include_router(websocket.router, prefix="/api/v1", tags=["websocket"])

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

### 2.2 API Routes

**New File:** `src/homesec/api/routes/cameras.py`

```python
"""Camera management API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..dependencies import get_homesec_app

router = APIRouter(prefix="/cameras")


class CameraCreate(BaseModel):
    """Request model for creating a camera."""
    name: str
    type: str  # rtsp, ftp, local_folder
    config: dict  # Type-specific configuration


class CameraUpdate(BaseModel):
    """Request model for updating a camera."""
    config: dict | None = None
    alert_policy: dict | None = None


class CameraResponse(BaseModel):
    """Response model for camera."""
    name: str
    type: str
    healthy: bool
    last_heartbeat: float | None
    config: dict
    alert_policy: dict | None


class CameraListResponse(BaseModel):
    """Response model for camera list."""
    cameras: list[CameraResponse]
    total: int


@router.get("", response_model=CameraListResponse)
async def list_cameras(app=Depends(get_homesec_app)):
    """List all configured cameras."""
    cameras = []
    for source in app.sources:
        cameras.append(CameraResponse(
            name=source.camera_name,
            type=source.source_type,
            healthy=source.is_healthy(),
            last_heartbeat=source.last_heartbeat(),
            config=source.get_config(),
            alert_policy=app.get_camera_alert_policy(source.camera_name),
        ))
    return CameraListResponse(cameras=cameras, total=len(cameras))


@router.get("/{camera_name}", response_model=CameraResponse)
async def get_camera(camera_name: str, app=Depends(get_homesec_app)):
    """Get a specific camera's configuration."""
    source = app.get_source(camera_name)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{camera_name}' not found",
        )
    return CameraResponse(
        name=source.camera_name,
        type=source.source_type,
        healthy=source.is_healthy(),
        last_heartbeat=source.last_heartbeat(),
        config=source.get_config(),
        alert_policy=app.get_camera_alert_policy(camera_name),
    )


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(camera: CameraCreate, app=Depends(get_homesec_app)):
    """Add a new camera."""
    if app.get_source(camera.name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Camera '{camera.name}' already exists",
        )

    try:
        source = await app.add_camera(
            name=camera.name,
            source_type=camera.type,
            config=camera.config,
        )
        return CameraResponse(
            name=source.camera_name,
            type=source.source_type,
            healthy=source.is_healthy(),
            last_heartbeat=source.last_heartbeat(),
            config=source.get_config(),
            alert_policy=None,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.put("/{camera_name}", response_model=CameraResponse)
async def update_camera(
    camera_name: str,
    update: CameraUpdate,
    app=Depends(get_homesec_app),
):
    """Update a camera's configuration."""
    source = app.get_source(camera_name)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{camera_name}' not found",
        )

    if update.config:
        await app.update_camera_config(camera_name, update.config)

    if update.alert_policy:
        await app.update_camera_alert_policy(camera_name, update.alert_policy)

    source = app.get_source(camera_name)
    return CameraResponse(
        name=source.camera_name,
        type=source.source_type,
        healthy=source.is_healthy(),
        last_heartbeat=source.last_heartbeat(),
        config=source.get_config(),
        alert_policy=app.get_camera_alert_policy(camera_name),
    )


@router.delete("/{camera_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(camera_name: str, app=Depends(get_homesec_app)):
    """Remove a camera."""
    source = app.get_source(camera_name)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{camera_name}' not found",
        )

    await app.remove_camera(camera_name)


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

**New File:** `src/homesec/api/routes/websocket.py`

```python
"""WebSocket routes for real-time updates."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from homesec.app import HomesecApp

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass  # Connection might be closed


manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time events."""
    await manager.connect(websocket)

    app: HomesecApp = websocket.app.state.homesec

    # Subscribe to app events
    event_queue = asyncio.Queue()

    async def event_handler(event_type: str, data: dict):
        await event_queue.put({"type": event_type, "data": data})

    app.subscribe_events(event_handler)

    try:
        while True:
            # Wait for either client message or app event
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(websocket.receive_text()),
                    asyncio.create_task(event_queue.get()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                result = task.result()

                if isinstance(result, str):
                    # Message from client
                    try:
                        msg = json.loads(result)
                        if msg.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                    except json.JSONDecodeError:
                        pass
                else:
                    # Event from app
                    await websocket.send_json(result)

            for task in pending:
                task.cancel()

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        app.unsubscribe_events(event_handler)
```

### 2.3 API Configuration

**File:** `src/homesec/models/config.py` (add)

```python
class APIConfig(BaseModel):
    """Configuration for the REST API."""

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: list[str] = ["*"]
    # Authentication (optional)
    auth_enabled: bool = False
    api_key_env: str | None = None
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
```

### 2.4 OpenAPI Documentation

The FastAPI app automatically generates OpenAPI docs at `/api/v1/docs` (Swagger UI) and `/api/v1/redoc` (ReDoc).

### 2.5 Acceptance Criteria

- [ ] All CRUD operations for cameras work
- [ ] Clip listing with filtering works
- [ ] Event history API works
- [ ] WebSocket broadcasts real-time events
- [ ] OpenAPI documentation is accurate
- [ ] CORS works for Home Assistant frontend
- [ ] API authentication (optional) works

---

## Phase 3: Home Assistant Add-on

**Goal:** Provide one-click installation for Home Assistant OS/Supervised users.

**Estimated Effort:** 3-4 days

### 3.1 Add-on Repository Structure

**New Repository:** `homesec-ha-addons`

```
homesec-ha-addons/
├── README.md
├── repository.json
└── homesec/
    ├── config.yaml           # Add-on manifest
    ├── Dockerfile            # Container build
    ├── build.yaml            # Build configuration
    ├── run.sh                # Startup script
    ├── DOCS.md               # Documentation
    ├── CHANGELOG.md          # Version history
    ├── icon.png              # Add-on icon (512x512)
    ├── logo.png              # Add-on logo (256x256)
    └── translations/
        └── en.yaml           # UI strings
```

### 3.2 Add-on Manifest

**File:** `homesec/config.yaml`

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
  8554/tcp: 8554    # RTSP proxy (if implemented)

# Volume mappings
map:
  - config:rw        # /config - HA config directory
  - media:rw         # /media - Media storage
  - share:rw         # /share - Shared data

# Services
services:
  - mqtt:want        # Use HA's MQTT broker

# Options schema
schema:
  config_path: str?
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
  openai_model: str?
  # MQTT Discovery
  mqtt_discovery: bool?

# Default options
options:
  config_path: /config/homesec/config.yaml
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

**File:** `homesec/Dockerfile`

```dockerfile
# syntax=docker/dockerfile:1
ARG BUILD_FROM=ghcr.io/hassio-addons/base:15.0.8
FROM ${BUILD_FROM}

# Install runtime dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    ffmpeg \
    postgresql-client \
    opencv \
    && rm -rf /var/cache/apk/*

# Install HomeSec
ARG HOMESEC_VERSION=1.2.2
RUN pip3 install --no-cache-dir homesec==${HOMESEC_VERSION}

# Copy root filesystem
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

### 3.4 Startup Script

**File:** `homesec/rootfs/etc/services.d/homesec/run`

```bash
#!/usr/bin/with-contenv bashio
# ==============================================================================
# HomeSec Add-on Startup Script
# ==============================================================================

# Read options
CONFIG_PATH=$(bashio::config 'config_path')
LOG_LEVEL=$(bashio::config 'log_level')
DATABASE_URL=$(bashio::config 'database_url')
STORAGE_TYPE=$(bashio::config 'storage_type')
STORAGE_PATH=$(bashio::config 'storage_path')
VLM_ENABLED=$(bashio::config 'vlm_enabled')
MQTT_DISCOVERY=$(bashio::config 'mqtt_discovery')

# Get MQTT credentials from HA if available
if bashio::services.available "mqtt"; then
    MQTT_HOST=$(bashio::services mqtt "host")
    MQTT_PORT=$(bashio::services mqtt "port")
    MQTT_USER=$(bashio::services mqtt "username")
    MQTT_PASS=$(bashio::services mqtt "password")

    export MQTT_HOST MQTT_PORT MQTT_USER MQTT_PASS
    bashio::log.info "Using Home Assistant MQTT broker at ${MQTT_HOST}:${MQTT_PORT}"
fi

# Create config directory if needed
mkdir -p "$(dirname "${CONFIG_PATH}")"

# Generate config if it doesn't exist
if [[ ! -f "${CONFIG_PATH}" ]]; then
    bashio::log.info "Generating initial configuration at ${CONFIG_PATH}"
    cat > "${CONFIG_PATH}" << EOF
version: 1

cameras: []

storage:
  type: ${STORAGE_TYPE}
  path: ${STORAGE_PATH}

state_store:
  type: postgres
  url_env: DATABASE_URL

notifiers:
  - type: mqtt
    host_env: MQTT_HOST
    port_env: MQTT_PORT
    username_env: MQTT_USER
    password_env: MQTT_PASS
    discovery:
      enabled: ${MQTT_DISCOVERY}

health:
  enabled: true
  port: 8080

api:
  enabled: true
  port: 8080
EOF
fi

# Set up database if using local PostgreSQL
if [[ -z "${DATABASE_URL}" ]]; then
    bashio::log.info "No database URL specified, using SQLite fallback"
    export DATABASE_URL="sqlite:///config/homesec/homesec.db"
fi

# Export secrets from config
if bashio::config.has_value 'dropbox_token'; then
    export DROPBOX_TOKEN=$(bashio::config 'dropbox_token')
fi

if bashio::config.has_value 'openai_api_key'; then
    export OPENAI_API_KEY=$(bashio::config 'openai_api_key')
fi

# Create storage directory
mkdir -p "${STORAGE_PATH}"

bashio::log.info "Starting HomeSec..."

# Run HomeSec
exec python3 -m homesec.cli run \
    --config "${CONFIG_PATH}" \
    --log-level "${LOG_LEVEL}"
```

### 3.5 Ingress Configuration

**File:** `homesec/rootfs/etc/nginx/includes/ingress.conf`

```nginx
# Proxy to HomeSec API
location / {
    proxy_pass http://127.0.0.1:8080;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # WebSocket support
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";

    # Timeouts for long-running connections
    proxy_connect_timeout 60s;
    proxy_send_timeout 60s;
    proxy_read_timeout 60s;
}
```

### 3.6 Acceptance Criteria

- [ ] Add-on installs successfully from repository
- [ ] Auto-configures MQTT from Home Assistant
- [ ] Ingress provides access to API/UI
- [ ] Configuration options work correctly
- [ ] Watchdog restarts on failure
- [ ] Logs are accessible in HA
- [ ] Works on both amd64 and aarch64

---

## Phase 4: Native Home Assistant Integration

**Goal:** Full UI-based configuration and deep entity integration in Home Assistant.

**Estimated Effort:** 7-10 days

### 4.1 Integration Structure

**Directory:** `custom_components/homesec/`

```
custom_components/homesec/
├── __init__.py           # Setup and entry points
├── manifest.json         # Integration metadata
├── const.py              # Constants
├── config_flow.py        # UI configuration flow
├── coordinator.py        # Data update coordinator
├── entity.py             # Base entity class
├── camera.py             # Camera platform
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

**File:** `custom_components/homesec/manifest.json`

```json
{
  "domain": "homesec",
  "name": "HomeSec",
  "codeowners": ["@lan17"],
  "config_flow": true,
  "dependencies": ["mqtt"],
  "documentation": "https://github.com/lan17/homesec",
  "integration_type": "hub",
  "iot_class": "local_push",
  "issue_tracker": "https://github.com/lan17/homesec/issues",
  "requirements": ["aiohttp>=3.8.0"],
  "version": "1.0.0"
}
```

### 4.3 Constants

**File:** `custom_components/homesec/const.py`

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

# Platforms
PLATFORMS: Final = [
    "binary_sensor",
    "camera",
    "image",
    "select",
    "sensor",
    "switch",
]

# Entity categories
DIAGNOSTIC_SENSORS: Final = ["health", "last_heartbeat"]

# Update intervals
SCAN_INTERVAL_SECONDS: Final = 30
WEBSOCKET_RECONNECT_DELAY: Final = 5

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

**File:** `custom_components/homesec/config_flow.py`

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

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
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
                # Check if already configured
                await self.async_set_unique_id(f"homesec_{self._host}_{self._port}")
                self._abort_if_unique_id_configured()

                return await self.async_step_cameras()

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
        )

    async def async_step_cameras(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle camera configuration step."""
        if user_input is not None:
            # Create the config entry
            return self.async_create_entry(
                title=f"HomeSec ({self._host})",
                data={
                    CONF_HOST: self._host,
                    CONF_PORT: self._port,
                    CONF_API_KEY: self._api_key,
                    CONF_VERIFY_SSL: self._verify_ssl,
                },
                options={
                    "cameras": user_input.get("cameras", []),
                },
            )

        # Build camera selection schema
        camera_names = [c["name"] for c in self._cameras]
        schema = vol.Schema(
            {
                vol.Optional(
                    "cameras",
                    default=camera_names,
                ): vol.All(
                    vol.Coerce(list),
                    [vol.In(camera_names)],
                ),
            }
        )

        return self.async_show_form(
            step_id="cameras",
            data_schema=schema,
            description_placeholders={
                "camera_count": str(len(self._cameras)),
            },
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

        # Fetch current cameras from HomeSec
        coordinator = self.hass.data[DOMAIN][self.config_entry.entry_id]
        camera_names = [c["name"] for c in coordinator.data.get("cameras", [])]
        current_cameras = self.config_entry.options.get("cameras", camera_names)

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        "cameras",
                        default=current_cameras,
                    ): vol.All(
                        vol.Coerce(list),
                        [vol.In(camera_names)],
                    ),
                    vol.Optional(
                        "scan_interval",
                        default=self.config_entry.options.get("scan_interval", 30),
                    ): vol.All(vol.Coerce(int), vol.Range(min=10, max=300)),
                }
            ),
        )


class CannotConnect(Exception):
    """Error to indicate we cannot connect."""


class InvalidAuth(Exception):
    """Error to indicate there is invalid auth."""
```

### 4.5 Data Coordinator

**File:** `custom_components/homesec/coordinator.py`

```python
"""Data coordinator for HomeSec integration."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Any

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.update_coordinator import (
    DataUpdateCoordinator,
    UpdateFailed,
)

from .const import DOMAIN, SCAN_INTERVAL_SECONDS, WEBSOCKET_RECONNECT_DELAY

_LOGGER = logging.getLogger(__name__)


class HomesecCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator to manage HomeSec data updates."""

    def __init__(
        self,
        hass: HomeAssistant,
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
        self._session = async_get_clientsession(hass, verify_ssl=verify_ssl)
        self._ws_task: asyncio.Task | None = None
        self._event_callbacks: list[callable] = []

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

                return {
                    "health": health,
                    "cameras": cameras_data.get("cameras", []),
                    "connected": True,
                }

        except asyncio.TimeoutError as err:
            raise UpdateFailed("Timeout connecting to HomeSec") from err
        except aiohttp.ClientError as err:
            raise UpdateFailed(f"Error communicating with HomeSec: {err}") from err

    async def async_start_websocket(self) -> None:
        """Start WebSocket connection for real-time updates."""
        if self._ws_task is not None:
            return
        self._ws_task = asyncio.create_task(self._websocket_loop())

    async def async_stop_websocket(self) -> None:
        """Stop WebSocket connection."""
        if self._ws_task is not None:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None

    async def _websocket_loop(self) -> None:
        """Maintain WebSocket connection and handle events."""
        ws_url = f"ws://{self.host}:{self.port}/api/v1/ws"

        while True:
            try:
                async with self._session.ws_connect(ws_url) as ws:
                    _LOGGER.info("Connected to HomeSec WebSocket")

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = msg.json()
                            await self._handle_ws_event(data)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            _LOGGER.error("WebSocket error: %s", ws.exception())
                            break

            except aiohttp.ClientError as err:
                _LOGGER.error("WebSocket connection error: %s", err)

            except asyncio.CancelledError:
                _LOGGER.info("WebSocket task cancelled")
                return

            _LOGGER.info(
                "WebSocket disconnected, reconnecting in %s seconds",
                WEBSOCKET_RECONNECT_DELAY,
            )
            await asyncio.sleep(WEBSOCKET_RECONNECT_DELAY)

    async def _handle_ws_event(self, data: dict[str, Any]) -> None:
        """Handle incoming WebSocket event."""
        event_type = data.get("type")
        event_data = data.get("data", {})

        _LOGGER.debug("Received WebSocket event: %s", event_type)

        # Trigger immediate data refresh for certain events
        if event_type in ["alert", "clip_recorded", "camera_status_changed"]:
            await self.async_request_refresh()

        # Notify registered callbacks
        for callback in self._event_callbacks:
            try:
                await callback(event_type, event_data)
            except Exception:
                _LOGGER.exception("Error in event callback")

    def register_event_callback(self, callback: callable) -> callable:
        """Register a callback for WebSocket events."""
        self._event_callbacks.append(callback)

        def remove():
            self._event_callbacks.remove(callback)

        return remove

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
        camera_type: str,
        config: dict,
    ) -> dict:
        """Add a new camera."""
        async with self._session.post(
            f"{self.base_url}/cameras",
            headers=self._headers,
            json={"name": name, "type": camera_type, "config": config},
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def async_update_camera(
        self,
        camera_name: str,
        config: dict | None = None,
        alert_policy: dict | None = None,
    ) -> dict:
        """Update camera configuration."""
        payload = {}
        if config is not None:
            payload["config"] = config
        if alert_policy is not None:
            payload["alert_policy"] = alert_policy

        async with self._session.put(
            f"{self.base_url}/cameras/{camera_name}",
            headers=self._headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def async_delete_camera(self, camera_name: str) -> None:
        """Delete a camera."""
        async with self._session.delete(
            f"{self.base_url}/cameras/{camera_name}",
            headers=self._headers,
        ) as response:
            response.raise_for_status()

    async def async_test_camera(self, camera_name: str) -> dict:
        """Test camera connection."""
        async with self._session.post(
            f"{self.base_url}/cameras/{camera_name}/test",
            headers=self._headers,
        ) as response:
            response.raise_for_status()
            return await response.json()
```

### 4.6 Entity Platforms

**File:** `custom_components/homesec/sensor.py`

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

**File:** `custom_components/homesec/binary_sensor.py`

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
            return camera.get("motion_detected", False)
        elif key == "person":
            return camera.get("person_detected", False)
        elif key == "online":
            return camera.get("healthy", False)

        return None
```

### 4.7 Services

**File:** `custom_components/homesec/services.yaml`

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
    type:
      name: Type
      description: Camera source type
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
      description: RTSP stream URL (for RTSP type)
      example: "rtsp://192.168.1.100:554/stream"
      selector:
        text:

remove_camera:
  name: Remove Camera
  description: Remove a camera from HomeSec
  target:
    device:
      integration: homesec

set_alert_policy:
  name: Set Alert Policy
  description: Configure alert policy for a camera
  target:
    device:
      integration: homesec
  fields:
    min_risk_level:
      name: Minimum Risk Level
      description: Minimum risk level to trigger alerts
      required: true
      selector:
        select:
          options:
            - "LOW"
            - "MEDIUM"
            - "HIGH"
            - "CRITICAL"
    activity_types:
      name: Activity Types
      description: Activity types that trigger alerts
      selector:
        select:
          multiple: true
          options:
            - "person"
            - "vehicle"
            - "animal"
            - "package"
            - "suspicious"

test_camera:
  name: Test Camera
  description: Test camera connection
  target:
    device:
      integration: homesec
```

### 4.8 Translations

**File:** `custom_components/homesec/translations/en.json`

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
    },
    "set_alert_policy": {
      "name": "Set Alert Policy",
      "description": "Configure alert policy for a camera"
    },
    "test_camera": {
      "name": "Test Camera",
      "description": "Test camera connection"
    }
  }
}
```

### 4.9 Acceptance Criteria

- [ ] Config flow connects to HomeSec and discovers cameras
- [ ] All entity platforms create entities correctly
- [ ] DataUpdateCoordinator fetches data at correct intervals
- [ ] WebSocket connection provides real-time updates
- [ ] Services work correctly (add/remove camera, set policy, test)
- [ ] Options flow allows reconfiguration
- [ ] Diagnostics provide useful debug information
- [ ] All strings are translatable
- [ ] HACS installation works

---

## Phase 5: Advanced Features

**Goal:** Premium features for power users.

**Estimated Effort:** 5-7 days

### 5.1 Camera Entity with Live Stream

**File:** `custom_components/homesec/camera.py`

```python
"""Camera platform for HomeSec integration."""

from __future__ import annotations

from homeassistant.components.camera import Camera, CameraEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .coordinator import HomesecCoordinator
from .entity import HomesecEntity


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up HomeSec cameras."""
    coordinator: HomesecCoordinator = hass.data[DOMAIN][entry.entry_id]

    entities = [
        HomesecCameraEntity(coordinator, camera["name"])
        for camera in coordinator.data.get("cameras", [])
    ]

    async_add_entities(entities)


class HomesecCameraEntity(HomesecEntity, Camera):
    """Representation of a HomeSec camera."""

    _attr_supported_features = CameraEntityFeature.STREAM

    def __init__(
        self,
        coordinator: HomesecCoordinator,
        camera_name: str,
    ) -> None:
        """Initialize the camera."""
        HomesecEntity.__init__(self, coordinator, camera_name)
        Camera.__init__(self)
        self._attr_unique_id = f"{camera_name}_camera"
        self._attr_name = None  # Use device name

    async def stream_source(self) -> str | None:
        """Return the source of the stream."""
        camera = self._get_camera_data()
        if not camera:
            return None

        # Get RTSP URL from camera config
        config = camera.get("config", {})
        rtsp_url = config.get("rtsp_url")

        if rtsp_url:
            return rtsp_url

        # Fallback to HomeSec RTSP proxy if available
        return f"rtsp://{self.coordinator.host}:8554/{self._camera_name}"

    async def async_camera_image(
        self, width: int | None = None, height: int | None = None
    ) -> bytes | None:
        """Return a still image from the camera."""
        camera = self._get_camera_data()
        if not camera:
            return None

        # Request snapshot from HomeSec API
        try:
            async with self.coordinator._session.get(
                f"{self.coordinator.base_url}/cameras/{self._camera_name}/snapshot",
                headers=self.coordinator._headers,
            ) as response:
                if response.status == 200:
                    return await response.read()
        except Exception:
            pass

        return None
```

### 5.2 Custom Lovelace Card (Optional)

**File:** `custom_components/homesec/www/homesec-camera-card.js`

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

### 5.3 Event Timeline Panel (Optional)

Create a custom panel for viewing event history with timeline visualization.

### 5.4 Acceptance Criteria

- [ ] Camera entities show live streams
- [ ] Snapshot images work
- [ ] Custom Lovelace card displays detection overlays
- [ ] Event timeline shows historical data

---

## Testing Strategy

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

1. **MQTT Discovery Tests:**
   - Publish discovery → verify entities appear in HA
   - HA restart → verify discovery republishes
   - Camera add → verify new entities created

2. **API Tests:**
   - Full CRUD cycle for cameras
   - WebSocket event delivery
   - Authentication flows

3. **Add-on Tests:**
   - Installation on HA OS
   - MQTT auto-configuration
   - Ingress access

### Manual Testing Checklist

- [ ] Install add-on from repository
- [ ] Configure via HA UI
- [ ] Add/remove cameras
- [ ] Verify entities update on alerts
- [ ] Test automations with HomeSec triggers
- [ ] Verify camera streams in dashboard

---

## Migration Guide

### From Standalone to Add-on

1. Export current `config.yaml`
2. Install HomeSec add-on
3. Copy config to `/config/homesec/config.yaml`
4. Update database URL if using external Postgres
5. Start add-on

### From MQTT-only to Full Integration

1. Keep existing MQTT configuration
2. Install custom integration via HACS
3. Configure integration with HomeSec URL
4. Entities will be created alongside MQTT entities
5. Optionally disable MQTT discovery to avoid duplicates

---

## Appendix: File Change Summary

### Phase 1: MQTT Discovery
- `src/homesec/models/config.py` - Add MQTTDiscoveryConfig
- `src/homesec/plugins/notifiers/mqtt.py` - Enhance with discovery
- `src/homesec/plugins/notifiers/mqtt_discovery.py` - New file
- `src/homesec/app.py` - Register cameras with notifier
- `config/example.yaml` - Add discovery example
- `tests/unit/plugins/notifiers/test_mqtt_discovery.py` - New tests

### Phase 2: REST API
- `src/homesec/api/` - New package (server.py, routes/*, dependencies.py)
- `src/homesec/models/config.py` - Add APIConfig
- `src/homesec/app.py` - Integrate API server
- `pyproject.toml` - Add fastapi, uvicorn dependencies

### Phase 3: Add-on
- New repository: `homesec-ha-addons/`
- `homesec/config.yaml` - Add-on manifest
- `homesec/Dockerfile` - Container build
- `homesec/rootfs/` - Startup scripts, nginx config

### Phase 4: Integration
- `custom_components/homesec/` - Full integration package
- HACS repository configuration

### Phase 5: Advanced
- `custom_components/homesec/camera.py` - Camera platform
- `custom_components/homesec/www/` - Lovelace cards

---

## Timeline Summary

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: MQTT Discovery | 2-3 days | None |
| Phase 2: REST API | 5-7 days | None |
| Phase 3: Add-on | 3-4 days | Phase 2 |
| Phase 4: Integration | 7-10 days | Phase 2 |
| Phase 5: Advanced | 5-7 days | Phase 4 |

**Total: 22-31 days**

Phases 1 and 2 can run in parallel. Phase 3 and 4 can run in parallel after Phase 2.
