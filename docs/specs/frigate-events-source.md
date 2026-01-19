# Frigate Events Source Plugin Specification

## Overview

The `frigate_events` source plugin ingests clips from [Frigate NVR](https://frigate.video/) to add HomeSec's structured VLM intelligence (risk levels, activity types, entity timelines) on top of Frigate's detection capabilities.

### Value Proposition

| Frigate Provides | HomeSec Adds |
|------------------|--------------|
| Real-time object detection (YOLO) | Structured VLM analysis |
| Event clips with thumbnails | Risk level classification (LOW → CRITICAL) |
| MQTT event notifications | Activity type categorization |
| Zone-based filtering | Entity timelines with movement tracking |
| Hardware acceleration | Unified multi-camera alerting |

**Key Insight**: Frigate excels at "what" detection (person, car, dog), while HomeSec's VLM adds "why" analysis (delivery, suspicious behavior, loitering).

---

## Architecture

### Data Flow

```
Frigate NVR
    │
    ├──[MQTT]──► frigate/events topic
    │                 │
    │                 ▼
    │         FrigateEventsSource (subscribe)
    │                 │
    │                 │ [event.type == "end"]
    │                 ▼
    ├──[HTTP]──► GET /api/events/{event_id}/clip
    │                 │
    │                 ▼
    │           Download clip to local_path
    │                 │
    │                 ▼
    │           Create HomeSec Clip
    │                 │
    │                 ▼
    │           callback(clip) → Pipeline
    │
    ▼
┌─────────────────────────────────────────────────┐
│              HomeSec Pipeline                   │
│                                                 │
│  [Upload] ──► [Filter*] ──► [VLM] ──► [Alert]  │
│                                                 │
│  * Filter stage can be bypassed since Frigate  │
│    already performed object detection          │
└─────────────────────────────────────────────────┘
```

### Integration Points

1. **MQTT Subscription**: Listen to `frigate/events` for real-time event notifications
2. **HTTP API**: Fetch clips and event metadata from Frigate's REST API
3. **Optional Filter Bypass**: Skip HomeSec's filter stage since Frigate already detected objects

---

## Configuration Model

```python
# src/homesec/models/source.py (addition)

class FrigateEventsConfig(BaseModel):
    """Configuration for Frigate events source."""

    # Frigate connection
    frigate_url: str = Field(
        description="Frigate HTTP API base URL (e.g., http://frigate:5000)"
    )
    frigate_url_env: str | None = Field(
        default=None,
        description="Environment variable containing Frigate URL"
    )

    # MQTT connection
    mqtt_host: str = Field(
        default="localhost",
        description="MQTT broker hostname"
    )
    mqtt_port: int = Field(
        default=1883,
        ge=1,
        le=65535,
        description="MQTT broker port"
    )
    mqtt_username: str | None = Field(
        default=None,
        description="MQTT username (optional)"
    )
    mqtt_password_env: str | None = Field(
        default=None,
        description="Environment variable containing MQTT password"
    )
    mqtt_topic_prefix: str = Field(
        default="frigate",
        description="Frigate MQTT topic prefix"
    )
    mqtt_client_id: str = Field(
        default="homesec-frigate",
        description="MQTT client identifier"
    )

    # Event filtering
    cameras: list[str] | None = Field(
        default=None,
        description="List of Frigate camera names to monitor (None = all)"
    )
    labels: list[str] = Field(
        default=["person"],
        description="Object labels to process (e.g., person, car, dog)"
    )
    min_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum detection confidence score"
    )
    zones: list[str] | None = Field(
        default=None,
        description="Only process events from these zones (None = all)"
    )
    min_duration_s: float = Field(
        default=1.0,
        ge=0.0,
        description="Minimum event duration to process"
    )

    # Clip handling
    output_dir: Path = Field(
        default=Path("./frigate_clips"),
        description="Local directory for downloaded clips"
    )
    clip_format: Literal["mp4", "original"] = Field(
        default="mp4",
        description="Clip format: 'mp4' for re-encoded, 'original' for raw"
    )
    retain_clips: bool = Field(
        default=True,
        description="Keep clips after pipeline completion"
    )
    download_timeout_s: float = Field(
        default=30.0,
        ge=5.0,
        description="HTTP timeout for clip downloads"
    )

    # Pipeline integration
    bypass_filter: bool = Field(
        default=True,
        description="Skip HomeSec filter stage (Frigate already detected)"
    )
    include_frigate_metadata: bool = Field(
        default=True,
        description="Include Frigate detection data in clip metadata"
    )

    # Reconnection
    reconnect_delay_s: float = Field(
        default=5.0,
        ge=1.0,
        description="Delay before reconnecting after disconnect"
    )
    max_reconnect_attempts: int = Field(
        default=10,
        ge=0,
        description="Max reconnection attempts (0 = unlimited)"
    )

    @model_validator(mode="after")
    def validate_frigate_url(self) -> "FrigateEventsConfig":
        """Ensure frigate_url is set via direct value or env var."""
        if not self.frigate_url and not self.frigate_url_env:
            raise ValueError("Either frigate_url or frigate_url_env must be set")
        return self
```

### Example YAML Configuration

```yaml
# config/frigate-example.yaml

version: 1

cameras:
  - name: frigate_all  # Logical name for HomeSec
    source:
      type: frigate_events
      config:
        frigate_url_env: FRIGATE_URL
        mqtt_host: mqtt.local
        mqtt_port: 1883
        mqtt_username: homesec
        mqtt_password_env: MQTT_PASSWORD

        # Only process person events from front cameras
        cameras:
          - front_door
          - driveway
          - porch
        labels:
          - person
          - car
        min_score: 0.75
        zones:
          - entrance
          - street

        # Clip handling
        output_dir: ./frigate_clips
        bypass_filter: true
        include_frigate_metadata: true

# Since Frigate already detected objects, we can skip the filter
# or use it for additional validation
filter:
  plugin: yolo
  max_workers: 2
  config:
    model_path: "yolo11n.pt"
    classes: [person, car]
    min_confidence: 0.5

# VLM adds structured intelligence on top of Frigate's detection
vlm:
  backend: openai
  trigger_classes: [person]  # Ignored if bypass_filter=true
  max_workers: 2
  llm:
    api_key_env: OPENAI_API_KEY
    model: gpt-4o

# Alert based on VLM analysis
alert_policy:
  backend: default
  config:
    min_risk_level: medium
    notify_on_activity_types:
      - suspicious_behavior
      - person_at_door
      - unknown

# Per-Frigate-camera overrides (uses Frigate camera names)
per_camera_alert:
  front_door:
    min_risk_level: low
    notify_on_activity_types: [person_at_door, delivery]
  driveway:
    min_risk_level: high
```

---

## Implementation

### Source Plugin Class

```python
# src/homesec/plugins/sources/frigate_events.py

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import aiomqtt
import httpx

from homesec.interfaces import ClipSource
from homesec.models.clip import Clip
from homesec.models.source import FrigateEventsConfig
from homesec.plugins.registry import PluginType, plugin

logger = logging.getLogger(__name__)


@plugin(plugin_type=PluginType.SOURCE, name="frigate_events")
class FrigateEventsSource(ClipSource):
    """
    Clip source that ingests events from Frigate NVR.

    Subscribes to Frigate's MQTT events and downloads clips via HTTP API
    when events complete. Designed to add HomeSec's VLM intelligence
    on top of Frigate's object detection.
    """

    config_cls = FrigateEventsConfig

    def __init__(self, config: FrigateEventsConfig, camera_name: str) -> None:
        self._config = config
        self._camera_name = camera_name
        self._callback: Callable[[Clip], None] | None = None
        self._task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()
        self._last_heartbeat: datetime | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._frigate_url = self._resolve_frigate_url()

    def _resolve_frigate_url(self) -> str:
        """Resolve Frigate URL from config or environment."""
        import os
        if self._config.frigate_url_env:
            url = os.environ.get(self._config.frigate_url_env)
            if url:
                return url.rstrip("/")
        return self._config.frigate_url.rstrip("/")

    @classmethod
    def create(cls, config: FrigateEventsConfig, camera_name: str) -> FrigateEventsSource:
        return cls(config, camera_name)

    def register_callback(self, callback: Callable[[Clip], None]) -> None:
        """Register callback for new clips."""
        self._callback = callback

    async def start(self) -> None:
        """Start the MQTT subscription and event processing."""
        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        self._http_client = httpx.AsyncClient(
            timeout=self._config.download_timeout_s,
            follow_redirects=True,
        )
        self._task = asyncio.create_task(self._run_mqtt_loop())
        logger.info(
            "FrigateEventsSource started for %s (cameras=%s, labels=%s)",
            self._camera_name,
            self._config.cameras or "all",
            self._config.labels,
        )

    async def shutdown(self, timeout: float = 10.0) -> None:
        """Stop the source gracefully."""
        self._shutdown_event.set()
        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        if self._http_client:
            await self._http_client.aclose()
        logger.info("FrigateEventsSource stopped for %s", self._camera_name)

    def is_healthy(self) -> bool:
        """Check if source is operational."""
        if self._task is None or self._task.done():
            return False
        if self._last_heartbeat is None:
            return True  # Just started, no heartbeat yet
        age = (datetime.now(timezone.utc) - self._last_heartbeat).total_seconds()
        return age < 300  # Healthy if heartbeat within 5 minutes

    def last_heartbeat(self) -> datetime | None:
        """Return last successful operation timestamp."""
        return self._last_heartbeat

    async def ping(self) -> bool:
        """Async health check - verify Frigate API is reachable."""
        try:
            if not self._http_client:
                return False
            response = await self._http_client.get(f"{self._frigate_url}/api/version")
            return response.status_code == 200
        except Exception:
            return False

    async def _run_mqtt_loop(self) -> None:
        """Main MQTT subscription loop with reconnection."""
        attempts = 0

        while not self._shutdown_event.is_set():
            try:
                await self._mqtt_session()
                attempts = 0  # Reset on successful session
            except aiomqtt.MqttError as e:
                attempts += 1
                max_attempts = self._config.max_reconnect_attempts
                if max_attempts > 0 and attempts >= max_attempts:
                    logger.error(
                        "Max reconnection attempts (%d) reached, stopping",
                        max_attempts,
                    )
                    break
                logger.warning(
                    "MQTT connection lost (attempt %d/%s): %s",
                    attempts,
                    max_attempts or "∞",
                    e,
                )
                await asyncio.sleep(self._config.reconnect_delay_s)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Unexpected error in MQTT loop: %s", e)
                await asyncio.sleep(self._config.reconnect_delay_s)

    async def _mqtt_session(self) -> None:
        """Single MQTT session - subscribe and process events."""
        import os

        password = None
        if self._config.mqtt_password_env:
            password = os.environ.get(self._config.mqtt_password_env)

        async with aiomqtt.Client(
            hostname=self._config.mqtt_host,
            port=self._config.mqtt_port,
            username=self._config.mqtt_username,
            password=password,
            identifier=self._config.mqtt_client_id,
        ) as client:
            topic = f"{self._config.mqtt_topic_prefix}/events"
            await client.subscribe(topic)
            logger.info("Subscribed to MQTT topic: %s", topic)

            async for message in client.messages:
                self._last_heartbeat = datetime.now(timezone.utc)
                try:
                    await self._handle_mqtt_message(message)
                except Exception as e:
                    logger.exception("Error handling MQTT message: %s", e)

    async def _handle_mqtt_message(self, message: aiomqtt.Message) -> None:
        """Process a single MQTT message from Frigate."""
        import json

        payload = json.loads(message.payload)
        event_type = payload.get("type")

        # Only process "end" events (clip is complete)
        if event_type != "end":
            return

        before = payload.get("before", {})
        after = payload.get("after", {})

        # Use "after" data for completed events
        event_data = after if after else before

        # Apply filters
        if not self._should_process_event(event_data):
            return

        # Download clip and create HomeSec Clip object
        clip = await self._create_clip_from_event(event_data)
        if clip and self._callback:
            self._callback(clip)

    def _should_process_event(self, event: dict[str, Any]) -> bool:
        """Check if event passes configured filters."""
        # Camera filter
        camera = event.get("camera")
        if self._config.cameras and camera not in self._config.cameras:
            logger.debug("Skipping event: camera %s not in filter", camera)
            return False

        # Label filter
        label = event.get("label")
        if label not in self._config.labels:
            logger.debug("Skipping event: label %s not in filter", label)
            return False

        # Score filter
        score = event.get("top_score", event.get("score", 0))
        if score < self._config.min_score:
            logger.debug("Skipping event: score %.2f below threshold", score)
            return False

        # Zone filter
        zones = event.get("zones", [])
        if self._config.zones:
            if not any(z in self._config.zones for z in zones):
                logger.debug("Skipping event: zones %s not in filter", zones)
                return False

        # Duration filter
        start_time = event.get("start_time", 0)
        end_time = event.get("end_time", 0)
        duration = end_time - start_time
        if duration < self._config.min_duration_s:
            logger.debug("Skipping event: duration %.1fs below threshold", duration)
            return False

        return True

    async def _create_clip_from_event(self, event: dict[str, Any]) -> Clip | None:
        """Download clip from Frigate and create HomeSec Clip object."""
        event_id = event.get("id")
        camera = event.get("camera")

        if not event_id or not camera:
            logger.warning("Event missing id or camera: %s", event)
            return None

        # Download clip
        clip_url = f"{self._frigate_url}/api/events/{event_id}/clip.mp4"
        try:
            response = await self._http_client.get(clip_url)
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("Failed to download clip %s: %s", event_id, e)
            return None

        # Save to local file
        start_time = event.get("start_time", 0)
        start_dt = datetime.fromtimestamp(start_time, tz=timezone.utc)
        timestamp_str = start_dt.strftime("%Y%m%d_%H%M%S")
        filename = f"{camera}_{timestamp_str}_{event_id}.mp4"
        local_path = self._config.output_dir / filename

        local_path.write_bytes(response.content)
        logger.info("Downloaded clip: %s (%.1f KB)", local_path, len(response.content) / 1024)

        # Calculate timestamps
        end_time = event.get("end_time", start_time)
        end_dt = datetime.fromtimestamp(end_time, tz=timezone.utc)
        duration = end_time - start_time

        # Build metadata with Frigate detection info
        metadata: dict[str, Any] = {}
        if self._config.include_frigate_metadata:
            metadata = {
                "frigate_event_id": event_id,
                "frigate_camera": camera,
                "frigate_label": event.get("label"),
                "frigate_score": event.get("top_score", event.get("score")),
                "frigate_zones": event.get("zones", []),
                "frigate_region": event.get("region"),
                "frigate_has_clip": event.get("has_clip", True),
                "frigate_has_snapshot": event.get("has_snapshot", False),
                # Detection box (for potential visualization)
                "frigate_box": event.get("box"),
                # Pre-detection filter result (bypass HomeSec filter)
                "bypass_filter": self._config.bypass_filter,
                "predetected_classes": [event.get("label")] if event.get("label") else [],
            }

        return Clip(
            clip_id=f"{camera}_{timestamp_str}_{event_id[:8]}",
            camera_name=camera,  # Use Frigate camera name for per-camera alerts
            local_path=local_path,
            start_ts=start_dt,
            end_ts=end_dt,
            duration_s=duration,
            source_type="frigate_events",
            metadata=metadata,
        )
```

### Pipeline Integration: Filter Bypass

The `bypass_filter` metadata flag allows skipping HomeSec's filter stage when Frigate has already performed object detection.

```python
# src/homesec/pipeline/core.py (modification to _process_clip)

async def _process_clip(self, clip: Clip) -> None:
    """Process a single clip through the pipeline."""

    # ... existing initialization ...

    # Stage 1: Upload + Filter (parallel)
    # Check for filter bypass (e.g., Frigate already detected)
    bypass_filter = clip.metadata.get("bypass_filter", False) if clip.metadata else False
    predetected_classes = clip.metadata.get("predetected_classes", []) if clip.metadata else []

    if bypass_filter and predetected_classes:
        # Create synthetic FilterResult from pre-detection
        filter_result = FilterResult(
            clip_id=clip.clip_id,
            detected_classes=predetected_classes,
            detections=[
                Detection(
                    class_name=cls,
                    confidence=clip.metadata.get("frigate_score", 0.9),
                    frame_number=0,
                    bbox=clip.metadata.get("frigate_box"),
                )
                for cls in predetected_classes
            ],
            frame_count=1,
            duration_ms=0,  # Not measured
            source="frigate_predetection",
        )
        logger.info(
            "Bypassing filter for %s (predetected: %s)",
            clip.clip_id,
            predetected_classes,
        )
    else:
        # Run normal filter stage
        filter_result = await self._filter_stage(clip)

    # ... rest of pipeline unchanged ...
```

---

## Extended Clip Model

Add optional metadata field to support Frigate enrichment:

```python
# src/homesec/models/clip.py (modification)

class Clip(BaseModel):
    """Represents a video clip from any source."""

    clip_id: str
    camera_name: str
    local_path: Path
    start_ts: datetime
    end_ts: datetime
    duration_s: float
    source_type: str

    # Optional metadata from source (e.g., Frigate detection data)
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Source-specific metadata (e.g., Frigate event data)"
    )
```

---

## Frigate MQTT Event Schema

For reference, here's the Frigate MQTT event structure:

```json
{
  "type": "end",
  "before": { ... },
  "after": {
    "id": "1234567890.123456-abc123",
    "camera": "front_door",
    "frame_time": 1234567890.123,
    "snapshot_time": 1234567890.456,
    "label": "person",
    "sub_label": null,
    "top_score": 0.89453125,
    "false_positive": false,
    "start_time": 1234567890.0,
    "end_time": 1234567910.0,
    "score": 0.87890625,
    "box": [0.1, 0.2, 0.3, 0.4],
    "area": 12345,
    "ratio": 0.75,
    "region": [0.0, 0.1, 0.5, 0.6],
    "stationary": false,
    "motionless_count": 0,
    "position_changes": 5,
    "current_zones": ["entrance"],
    "entered_zones": ["entrance", "driveway"],
    "has_clip": true,
    "has_snapshot": true,
    "attributes": {},
    "current_attributes": []
  }
}
```

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
frigate = [
    "aiomqtt>=2.0.0",
    "httpx>=0.25.0",
]

# Or add to main dependencies if Frigate is a core feature
```

---

## Testing Strategy

### Unit Tests

```python
# tests/plugins/sources/test_frigate_events.py

import pytest
from unittest.mock import AsyncMock, MagicMock
from homesec.plugins.sources.frigate_events import FrigateEventsSource
from homesec.models.source import FrigateEventsConfig


@pytest.fixture
def frigate_config():
    return FrigateEventsConfig(
        frigate_url="http://frigate:5000",
        mqtt_host="localhost",
        cameras=["front_door"],
        labels=["person"],
        min_score=0.7,
    )


@pytest.fixture
def sample_frigate_event():
    return {
        "type": "end",
        "after": {
            "id": "1234567890.123456-abc123",
            "camera": "front_door",
            "label": "person",
            "top_score": 0.85,
            "start_time": 1234567890.0,
            "end_time": 1234567910.0,
            "zones": ["entrance"],
            "has_clip": True,
        }
    }


class TestFrigateEventsSource:
    def test_should_process_event_passes_valid(self, frigate_config, sample_frigate_event):
        source = FrigateEventsSource(frigate_config, "test")
        assert source._should_process_event(sample_frigate_event["after"])

    def test_should_process_event_filters_wrong_camera(self, frigate_config, sample_frigate_event):
        source = FrigateEventsSource(frigate_config, "test")
        event = sample_frigate_event["after"].copy()
        event["camera"] = "backyard"
        assert not source._should_process_event(event)

    def test_should_process_event_filters_low_score(self, frigate_config, sample_frigate_event):
        source = FrigateEventsSource(frigate_config, "test")
        event = sample_frigate_event["after"].copy()
        event["top_score"] = 0.5
        assert not source._should_process_event(event)

    @pytest.mark.asyncio
    async def test_create_clip_from_event(self, frigate_config, sample_frigate_event, tmp_path):
        frigate_config.output_dir = tmp_path
        source = FrigateEventsSource(frigate_config, "test")

        # Mock HTTP client
        source._http_client = AsyncMock()
        source._http_client.get.return_value = MagicMock(
            content=b"fake video data",
            raise_for_status=MagicMock(),
        )

        clip = await source._create_clip_from_event(sample_frigate_event["after"])

        assert clip is not None
        assert clip.camera_name == "front_door"
        assert clip.source_type == "frigate_events"
        assert clip.metadata["frigate_label"] == "person"
        assert clip.metadata["bypass_filter"] is True
```

### Integration Test

```python
# tests/integration/test_frigate_integration.py

@pytest.mark.integration
@pytest.mark.asyncio
async def test_frigate_to_vlm_pipeline(
    frigate_source: FrigateEventsSource,
    mock_vlm_analyzer: VLMAnalyzer,
    pipeline: ClipPipeline,
):
    """End-to-end test: Frigate event → HomeSec VLM analysis."""
    clips_processed = []

    def on_clip_done(clip_id: str, result: PipelineResult):
        clips_processed.append((clip_id, result))

    pipeline.on_clip_completed = on_clip_done

    # Simulate Frigate event
    await frigate_source._handle_mqtt_message(
        MagicMock(payload=json.dumps(SAMPLE_FRIGATE_EVENT))
    )

    # Wait for pipeline
    await asyncio.sleep(1)

    assert len(clips_processed) == 1
    clip_id, result = clips_processed[0]

    # VLM should have been called (filter bypassed)
    assert result.analysis_result is not None
    assert result.analysis_result.risk_level is not None
```

---

## Operational Considerations

### Health Monitoring

The source exposes health via standard `ClipSource` interface:
- `is_healthy()`: Returns `True` if MQTT connected and heartbeat within 5 minutes
- `ping()`: Async check against Frigate HTTP API `/api/version`

### Graceful Degradation

| Failure Mode | Behavior |
|--------------|----------|
| MQTT disconnect | Auto-reconnect with backoff (configurable attempts) |
| Frigate API down | Log error, skip clip download, continue listening |
| Clip download fails | Log error, don't create Clip, continue processing other events |
| Frigate clip missing | Frigate may delete clips on its retention policy - check `has_clip` |

### Logging

```python
# Example log output
INFO  - FrigateEventsSource started for frigate_all (cameras=['front_door', 'driveway'], labels=['person', 'car'])
INFO  - Subscribed to MQTT topic: frigate/events
INFO  - Downloaded clip: ./frigate_clips/front_door_20240115_143022_abc123.mp4 (1.2 MB)
DEBUG - Skipping event: camera backyard not in filter
WARN  - MQTT connection lost (attempt 1/10): Connection refused
INFO  - FrigateEventsSource stopped for frigate_all
```

---

## Future Enhancements

1. **Sub-labels**: Support Frigate's sub-label feature (e.g., "person" → "delivery_driver")
2. **Snapshot integration**: Optionally fetch snapshots for thumbnail generation
3. **Zone-aware alerting**: Map Frigate zones to HomeSec alert policy overrides
4. **Two-way integration**: Publish HomeSec analysis back to Frigate via MQTT
5. **Frigate+ labels**: Support Frigate+ model labels when available
6. **WebSocket API**: Alternative to MQTT using Frigate's WebSocket endpoint
7. **Clip retention sync**: Coordinate HomeSec clip deletion with Frigate retention policy

---

## Summary

The `frigate_events` source bridges Frigate's excellent real-time detection with HomeSec's structured VLM analysis. Key design decisions:

1. **MQTT for events, HTTP for clips**: Leverages Frigate's standard integration points
2. **Filter bypass option**: Avoids redundant object detection when Frigate already detected
3. **Metadata preservation**: Carries Frigate detection data through pipeline for enrichment
4. **Per-camera alerts**: Uses Frigate camera names for alert policy overrides
5. **Graceful degradation**: Continues operating through transient failures

This creates a powerful combination: Frigate handles the "what" (person detected at front door), while HomeSec's VLM answers the "why" (delivery driver, suspicious loitering, family member returning home).
