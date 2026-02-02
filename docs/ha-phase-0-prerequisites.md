# Phase 0: Core Prerequisites

**Goal**: Prepare the HomeSec core codebase for HA integration by adding the `enabled` field to CameraConfig.

**Estimated Effort**: 0.5 days

**Dependencies**: None

---

## Overview

Before implementing the HA integration, one small change is needed in the core HomeSec codebase:

1. Add `enabled` field to `CameraConfig`

**Note**: Camera health monitoring uses a **pull-based** architecture. The HA Integration (Phase 4) will poll the REST API (Phase 1) to get camera health status. No push-based health monitoring is needed in the core.

---

## 0.1 Add `enabled` Field to CameraConfig

**File**: `src/homesec/models/config.py`

### Interface

```python
class CameraConfig(BaseModel):
    """Camera configuration and clip source selection."""

    name: str
    enabled: bool = True  # Allow disabling camera via API
    source: CameraSourceConfig
```

### Constraints

- Default is `True` (backwards compatible)
- When `enabled=False`, Application must not start the source
- API can toggle this field; requires restart to take effect

---

## 0.2 Respect `enabled` in Application

**File**: `src/homesec/app.py`

### Interface

```python
def _create_sources(self, config: Config) -> list[ClipSource]:
    """Create clip sources based on config using plugin registry.

    Respects CameraConfig.enabled - skips disabled cameras.
    """
    sources: list[ClipSource] = []

    for camera in config.cameras:
        if not camera.enabled:
            logger.info("Skipping disabled camera: %s", camera.name)
            continue

        source_cfg = camera.source
        source = load_source_plugin(
            source_backend=source_cfg.backend,
            config=source_cfg.config,
            camera_name=camera.name,
        )
        sources.append(source)
        self._sources_by_name[camera.name] = source

    return sources
```

### Constraints

- Skip sources where `enabled=False`
- Log which cameras are skipped
- Maintain `_sources_by_name` mapping for API queries (Phase 1)

---

## File Changes Summary

| File | Change |
|------|--------|
| `src/homesec/models/config.py` | Add `enabled: bool = True` to `CameraConfig` |
| `src/homesec/app.py` | Skip disabled cameras, add `_sources_by_name` mapping |

---

## Health Monitoring Architecture Note

Camera health uses a **pull-based** architecture:

1. **ClipSource** already has `is_healthy()` method
2. **Phase 1 REST API** exposes `GET /api/v1/cameras/{name}/status` (includes health)
3. **Phase 4 HA Integration** polls that endpoint every 30-60s via DataUpdateCoordinator

This keeps the HomeSec core simple and stateless. The HA Integration handles converting pull to entity updates.

---

## Test Expectations

### Test Cases

**CameraConfig.enabled**
- Given a config with `enabled=False`, when Application starts, then source is not started
- Given a config with `enabled=True` (or missing), when Application starts, then source is started
- Given two cameras (one enabled, one disabled), when Application starts, then only enabled camera source is created

---

## Verification

```bash
# Run unit tests
make check

# Verify CameraConfig accepts enabled field
python -c "from homesec.models.config import CameraConfig, CameraSourceConfig; print(CameraConfig(name='test', enabled=False, source=CameraSourceConfig(backend='rtsp', config={})))"
```

---

## Definition of Done

- [x] `CameraConfig` has `enabled: bool = True` field
- [x] Application skips starting sources where `enabled=False`
- [x] `_sources_by_name` mapping maintained for future API use
- [ ] All tests pass
- [x] Existing functionality unchanged (backwards compatible)
