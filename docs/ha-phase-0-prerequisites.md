# Phase 0: Core Prerequisites

**Goal**: Prepare the HomeSec core codebase for HA integration by adding required fields and interfaces.

**Estimated Effort**: 2-3 days

**Dependencies**: None

---

## Overview

Before implementing the HA integration, these changes are needed in the core HomeSec codebase:

1. Add `enabled` field to `CameraConfig`
2. Add `publish_camera_health()` to `Notifier` interface
3. Implement `publish_camera_health()` in `MultiplexNotifier`
4. Add no-op implementations to existing notifiers
5. Add camera health monitoring loop in `Application`
6. Add health monitoring configuration

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

### Current Interface

```python
class Notifier(Shutdownable, ABC):
    """Sends notifications (e.g., MQTT, email, SMS)."""

    @abstractmethod
    async def send(self, alert: Alert) -> None:
        """Send notification. Raises on failure."""
        raise NotImplementedError

    @abstractmethod
    async def ping(self) -> bool:
        """Health check. Returns True if notifier is reachable."""
        raise NotImplementedError
```

### Add This Method

```python
    # NEW: Camera health status change (add to Notifier class)
    async def publish_camera_health(self, camera_name: str, healthy: bool) -> None:
        """Publish camera health status change.

        Called by Application when a camera transitions between healthy/unhealthy.
        Default implementation is no-op. Override in subclasses that support it.

        Implementations should be best-effort (don't raise on failure).
        """
        pass  # Default no-op
```

### Constraints

- Must be async
- Default implementation is no-op (not abstract - existing notifiers don't break)
- Must be best-effort (failures don't crash pipeline)
- Called on health state transitions only, not on every health check

---

## 0.3 Update MultiplexNotifier

**File**: `src/homesec/notifiers/multiplex.py`

### Add This Method

```python
async def publish_camera_health(self, camera_name: str, healthy: bool) -> None:
    """Broadcast camera health to all notifiers.

    Fan out to all notifiers, log errors but don't raise.
    """
    if self._shutdown_called:
        return

    results = await self._call_all(
        lambda notifier: notifier.publish_camera_health(camera_name, healthy)
    )

    for entry, result in results:
        match result:
            case BaseException() as err:
                logger.warning(
                    "Notifier publish_camera_health failed: notifier=%s error=%s",
                    entry.name,
                    err,
                )
```

### Constraints

- Must fan out to all configured notifiers
- Must handle individual notifier failures gracefully (log, don't raise)
- Should continue to other notifiers even if one fails

---

## 0.4 Add Health Monitoring Configuration

**File**: `src/homesec/models/config.py`

### Interface

```python
class HealthMonitorConfig(BaseModel):
    """Configuration for camera health monitoring."""

    enabled: bool = True
    check_interval_s: int = Field(default=30, ge=5, le=300)
```

Add to `Config` class:
```python
class Config(BaseModel):
    # ... existing fields ...
    health_monitor: HealthMonitorConfig = Field(default_factory=HealthMonitorConfig)
```

### Constraints

- Default interval is 30 seconds
- Minimum 5 seconds, maximum 300 seconds
- Can be disabled entirely with `enabled: false`

---

## 0.5 Camera Health Monitoring in Application

**File**: `src/homesec/app.py`

### Interface

```python
class Application:
    """Main application orchestrator."""

    def __init__(self, ...):
        # ... existing init ...
        self._camera_health_state: dict[str, bool] = {}  # Track previous health per camera
        self._health_monitor_task: asyncio.Task | None = None

    async def _start_sources(self) -> None:
        """Start all enabled camera sources.

        Must check camera.enabled before starting each source.
        Skip sources where enabled=False.
        """
        for camera_config in self._config.cameras:
            if not camera_config.enabled:
                logger.info("Skipping disabled camera: %s", camera_config.name)
                continue
            # ... start source ...

    async def _monitor_camera_health(self) -> None:
        """Background task that monitors camera health and publishes changes.

        Runs every check_interval_s seconds, checks is_healthy() on each source,
        and calls notifier.publish_camera_health() only when state changes.
        """
        interval = self._config.health_monitor.check_interval_s

        while True:
            await asyncio.sleep(interval)

            for camera_name, source in self._sources.items():
                current_healthy = source.is_healthy()
                previous_healthy = self._camera_health_state.get(camera_name)

                # Only publish on state transitions
                if previous_healthy is not None and current_healthy != previous_healthy:
                    await self._notifier.publish_camera_health(camera_name, current_healthy)

                self._camera_health_state[camera_name] = current_healthy

    async def run(self) -> None:
        """Run the application."""
        # ... existing startup ...

        # Start health monitoring if enabled
        if self._config.health_monitor.enabled:
            self._health_monitor_task = asyncio.create_task(self._monitor_camera_health())

        # ... rest of run ...

    async def shutdown(self) -> None:
        """Shutdown the application."""
        # Cancel health monitor
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass

        # ... rest of shutdown ...
```

### Constraints

- Only publish on state transitions (healthy→unhealthy or unhealthy→healthy)
- Track previous health state per camera
- Must respect `CameraConfig.enabled` - don't start disabled cameras
- Health monitoring task should be cancelled on shutdown
- First health check after startup doesn't trigger publish (no previous state)

---

## File Changes Summary

| File | Change |
|------|--------|
| `src/homesec/models/config.py` | Add `enabled: bool = True` to `CameraConfig`, add `HealthMonitorConfig` |
| `src/homesec/interfaces.py` | Add `publish_camera_health()` with default no-op to `Notifier` |
| `src/homesec/notifiers/multiplex.py` | Implement `publish_camera_health()` fan-out |
| `src/homesec/app.py` | Add health monitoring loop, respect `enabled` field |

**Note**: Existing notifiers (MQTT, SendGrid) inherit the default no-op implementation and don't need changes.

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
- Given a camera transitions from healthy to unhealthy, when health monitor runs, then `publish_camera_health(camera, False)` is called once
- Given a camera stays healthy across checks, when health monitor runs, then `publish_camera_health` is NOT called
- Given a camera is checked for the first time, when health monitor runs, then `publish_camera_health` is NOT called (no previous state)
- Given MultiplexNotifier with 3 notifiers and one fails, when `publish_camera_health` is called, then other notifiers still receive the call

**HealthMonitorConfig**
- Given `health_monitor.enabled=False`, when Application starts, then no health monitor task is created
- Given `health_monitor.check_interval_s=10`, when Application runs, then health checks happen every 10 seconds

---

## Verification

```bash
# Run unit tests
pytest tests/unit/test_app.py -v -k "health"
pytest tests/unit/models/test_config.py -v -k "enabled"
pytest tests/unit/notifiers/test_multiplex.py -v -k "camera_health"

# Verify CameraConfig accepts enabled field
python -c "from homesec.models.config import CameraConfig, CameraSourceConfig; print(CameraConfig(name='test', enabled=False, source=CameraSourceConfig(backend='rtsp', config={})))"

# Verify HealthMonitorConfig
python -c "from homesec.models.config import HealthMonitorConfig; print(HealthMonitorConfig(check_interval_s=60))"
```

---

## Definition of Done

- [ ] `CameraConfig` has `enabled: bool = True` field
- [ ] `HealthMonitorConfig` exists with `enabled` and `check_interval_s` fields
- [ ] `Notifier` interface includes `publish_camera_health()` with default no-op
- [ ] `MultiplexNotifier` fans out `publish_camera_health()` to all notifiers
- [ ] Application skips starting sources where `enabled=False`
- [ ] Application monitors camera health and calls `publish_camera_health()` on transitions only
- [ ] Health monitoring respects configuration (enabled, interval)
- [ ] All tests pass
- [ ] Existing functionality unchanged (backwards compatible)
