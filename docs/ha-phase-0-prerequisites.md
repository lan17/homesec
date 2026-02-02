# Phase 0: Core Prerequisites

**Goal**: Prepare the HomeSec core codebase for HA integration by adding required fields and interfaces.

**Estimated Effort**: 2-3 days

**Dependencies**: None

---

## Overview

Before implementing the HA integration, these changes are needed in the core HomeSec codebase:

1. Add `enabled` field to `CameraConfig`
2. Add `publish_camera_health()` to Notifier interface
3. Implement camera health monitoring loop in Application
4. Add stats methods to ClipRepository

---

## 0.1 Add `enabled` Field to CameraConfig

**File**: `src/homesec/models/config.py`

### Interface

```python
class CameraConfig(BaseModel):
    """Camera configuration and clip source selection."""

    name: str
    enabled: bool = True  # NEW: Allow disabling camera via API
    source: CameraSourceConfig
```

### Constraints

- Default is `True` (backwards compatible)
- When `enabled=False`, Application must not start the source
- API can toggle this field; requires restart to take effect

---

## 0.2 Add `publish_camera_health()` to Notifier Interface

**File**: `src/homesec/interfaces.py`

### Interface

```python
class Notifier(Protocol):
    """Notification service interface."""

    async def start(self) -> None: ...
    async def shutdown(self) -> None: ...
    async def notify(self, alert: Alert) -> None: ...

    # NEW: Camera health status change
    async def publish_camera_health(self, camera_name: str, healthy: bool) -> None:
        """Publish camera health status change.

        Called by Application when a camera transitions between healthy/unhealthy.
        Implementations should be best-effort (don't raise on failure).
        """
        ...
```

### Constraints

- Must be async
- Must be best-effort (failures don't crash pipeline)
- Called on health state transitions, not on every health check
- Existing notifiers (MQTT, SendGrid) can implement as no-op initially

---

## 0.3 Update MultiplexNotifier

**File**: `src/homesec/notifiers/multiplex.py`

### Interface

```python
class MultiplexNotifier:
    """Routes notifications to multiple backends."""

    async def publish_camera_health(self, camera_name: str, healthy: bool) -> None:
        """Broadcast camera health to all notifiers."""
        # Fan out to all notifiers, log errors but don't raise
```

### Constraints

- Must fan out to all configured notifiers
- Must handle individual notifier failures gracefully
- Should log errors but continue to other notifiers

---

## 0.4 Camera Health Monitoring in Application

**File**: `src/homesec/app.py`

### Interface

```python
class Application:
    """Main application orchestrator."""

    async def _monitor_camera_health(self) -> None:
        """Background task that monitors camera health and publishes changes.

        Runs every N seconds (configurable), checks is_healthy() on each source,
        and calls notifier.publish_camera_health() when state changes.
        """
        ...

    async def _start_sources(self) -> None:
        """Start all enabled camera sources.

        Must check camera.enabled before starting each source.
        """
        ...
```

### Constraints

- Health check interval should be configurable (default: 30s)
- Only publish on state transitions (healthy→unhealthy or unhealthy→healthy)
- Track previous health state per camera
- Must respect `CameraConfig.enabled` - don't start disabled cameras
- Health monitoring task should be cancelled on shutdown

---

## 0.5 Add Stats Methods to ClipRepository

**File**: `src/homesec/repository.py` (or appropriate location)

### Interface

```python
class ClipRepository(Protocol):
    """Repository for clip and event data."""

    # Existing methods...

    # NEW: Stats methods for API
    async def count_clips_since(self, since: datetime) -> int:
        """Count clips created since the given timestamp."""
        ...

    async def count_alerts_since(self, since: datetime) -> int:
        """Count alert events (notification_sent) since the given timestamp."""
        ...
```

### Constraints

- Must use async SQLAlchemy
- Should be efficient (use COUNT, not fetch all)
- `count_alerts_since` counts events where `event_type='notification_sent'`

---

## File Changes Summary

| File | Change |
|------|--------|
| `src/homesec/models/config.py` | Add `enabled: bool = True` to `CameraConfig` |
| `src/homesec/interfaces.py` | Add `publish_camera_health()` to `Notifier` protocol |
| `src/homesec/notifiers/multiplex.py` | Implement `publish_camera_health()` fan-out |
| `src/homesec/plugins/notifiers/mqtt.py` | Add no-op `publish_camera_health()` |
| `src/homesec/plugins/notifiers/sendgrid_email.py` | Add no-op `publish_camera_health()` |
| `src/homesec/app.py` | Add health monitoring loop, respect `enabled` field |
| `src/homesec/repository.py` | Add `count_clips_since()`, `count_alerts_since()` |

---

## Test Expectations

### Fixtures Needed

- `camera_config_enabled` - CameraConfig with enabled=True
- `camera_config_disabled` - CameraConfig with enabled=False
- `mock_notifier` - Notifier that records calls to `publish_camera_health()`
- `mock_source_healthy` - ClipSource where `is_healthy()` returns True
- `mock_source_unhealthy` - ClipSource where `is_healthy()` returns False

### Test Cases

**CameraConfig.enabled**
- Given a config with `enabled=False`, when Application starts, then source is not started
- Given a config with `enabled=True` (or missing), when Application starts, then source is started

**publish_camera_health**
- Given a camera transitions from healthy to unhealthy, when health monitor runs, then `publish_camera_health(camera, False)` is called
- Given a camera stays healthy, when health monitor runs, then `publish_camera_health` is NOT called
- Given MultiplexNotifier with 3 notifiers and one fails, when `publish_camera_health` is called, then other notifiers still receive the call

**count_clips_since / count_alerts_since**
- Given 5 clips created today and 10 yesterday, when `count_clips_since(today_start)` is called, then returns 5
- Given 0 alerts, when `count_alerts_since(any_date)` is called, then returns 0

---

## Verification

```bash
# Run unit tests
pytest tests/unit/test_app.py -v -k "health"
pytest tests/unit/models/test_config.py -v -k "enabled"
pytest tests/unit/notifiers/test_multiplex.py -v -k "camera_health"

# Verify CameraConfig accepts enabled field
python -c "from homesec.models.config import CameraConfig; print(CameraConfig(name='test', enabled=False, source={'backend': 'rtsp', 'config': {}}))"
```

---

## Definition of Done

- [ ] `CameraConfig` has `enabled: bool = True` field
- [ ] `Notifier` protocol includes `publish_camera_health()`
- [ ] `MultiplexNotifier` fans out `publish_camera_health()` to all notifiers
- [ ] Existing notifiers (MQTT, SendGrid) have no-op implementations
- [ ] Application skips starting sources where `enabled=False`
- [ ] Application monitors camera health and calls `publish_camera_health()` on transitions
- [ ] `ClipRepository` has `count_clips_since()` and `count_alerts_since()`
- [ ] All tests pass
- [ ] Existing functionality unchanged (backwards compatible)
