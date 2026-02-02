# Phase 2: Home Assistant Notifier Plugin

**Goal**: Enable real-time event push from HomeSec to Home Assistant without requiring MQTT.

**Estimated Effort**: 2-3 days

**Dependencies**: None (can be done in parallel with Phase 1)

---

## Overview

This phase adds a new notifier plugin that pushes events directly to Home Assistant via the HA Events API. When running as an add-on, it uses `SUPERVISOR_TOKEN` for zero-config authentication.

---

## 2.1 Configuration Model

**File**: `src/homesec/models/config.py`

### Interface

```python
class HomeAssistantNotifierConfig(BaseModel):
    """Configuration for Home Assistant notifier.

    When running as HA add-on, SUPERVISOR_TOKEN is used automatically.
    For standalone mode, url_env and token_env must be provided.
    """

    # Standalone mode only (ignored when SUPERVISOR_TOKEN present)
    url_env: str | None = None      # e.g., "HA_URL" -> http://homeassistant.local:8123
    token_env: str | None = None    # e.g., "HA_TOKEN" -> long-lived access token

    # Event configuration
    event_prefix: str = "homesec"   # Events: homesec_alert, homesec_camera_health, etc.
```

### Constraints

- `url_env` and `token_env` store env var names, not actual values
- When `SUPERVISOR_TOKEN` env var exists, use supervisor mode automatically
- In supervisor mode, url_env and token_env are ignored

---

## 2.2 Notifier Implementation

**File**: `src/homesec/plugins/notifiers/home_assistant.py`

### Interface

```python
@plugin(plugin_type=PluginType.NOTIFIER, name="home_assistant")
class HomeAssistantNotifier(Notifier):
    """Push events directly to Home Assistant via Events API."""

    config_cls = HomeAssistantNotifierConfig

    def __init__(self, config: HomeAssistantNotifierConfig): ...

    async def start(self) -> None:
        """Initialize HTTP session and detect supervisor mode.

        Supervisor mode: SUPERVISOR_TOKEN env var exists
        Standalone mode: requires url_env and token_env in config

        Raises:
            ValueError: If standalone mode and url_env/token_env missing
        """
        ...

    async def shutdown(self) -> None:
        """Close HTTP session."""
        ...

    async def notify(self, alert: Alert) -> None:
        """Send alert to Home Assistant as homesec_alert event.

        Event data includes:
        - camera: str
        - clip_id: str
        - activity_type: str
        - risk_level: str
        - summary: str
        - view_url: str | None
        - storage_uri: str | None
        - timestamp: ISO8601 string
        - detected_objects: list[str] (if analysis present)
        """
        ...

    async def publish_camera_health(self, camera_name: str, healthy: bool) -> None:
        """Send camera health event to Home Assistant.

        Event: homesec_camera_health
        Data: {camera: str, healthy: bool, status: "healthy"|"unhealthy"}
        """
        ...

    async def publish_clip_recorded(self, clip_id: str, camera_name: str) -> None:
        """Send clip recorded event to Home Assistant.

        Event: homesec_clip_recorded
        Data: {clip_id: str, camera: str}
        """
        ...
```

### Internal Methods

```python
def _get_url_and_headers(self, event_type: str) -> tuple[str, dict[str, str]]:
    """Get the URL and headers for the HA Events API.

    Supervisor mode:
        URL: http://supervisor/core/api/events/{prefix}_{event_type}
        Auth: Bearer {SUPERVISOR_TOKEN}

    Standalone mode:
        URL: {resolved_url}/api/events/{prefix}_{event_type}
        Auth: Bearer {resolved_token}
    """
    ...
```

### Constraints

- All event publishing is best-effort (log errors, don't raise)
- Use aiohttp for HTTP requests
- Events API endpoint: `POST /api/events/{event_type}`
- Supervisor URL: `http://supervisor/core/api/events/...`
- Event names: `{event_prefix}_alert`, `{event_prefix}_camera_health`, `{event_prefix}_clip_recorded`

---

## 2.3 Events Reference

### homesec_alert

Fired when an alert is generated after VLM analysis.

```json
{
  "camera": "front_door",
  "clip_id": "abc123",
  "activity_type": "person_at_door",
  "risk_level": "medium",
  "summary": "Person approached front door and rang doorbell",
  "view_url": "https://dropbox.com/...",
  "storage_uri": "dropbox:///clips/abc123.mp4",
  "timestamp": "2026-02-01T10:30:00Z",
  "detected_objects": ["person"]
}
```

### homesec_camera_health

Fired when camera health status changes.

```json
{
  "camera": "front_door",
  "healthy": false,
  "status": "unhealthy"
}
```

### homesec_clip_recorded

Fired when a new clip is recorded (before analysis).

```json
{
  "clip_id": "abc123",
  "camera": "front_door"
}
```

---

## 2.4 Configuration Example

**File**: `config/example.yaml` (add section)

```yaml
notifiers:
  # Home Assistant notifier (recommended for HA users)
  - backend: home_assistant
    config:
      # When running as HA add-on, no configuration needed (uses SUPERVISOR_TOKEN)
      # For standalone mode, provide HA URL and token:
      # url_env: HA_URL           # http://homeassistant.local:8123
      # token_env: HA_TOKEN       # Long-lived access token from HA
      event_prefix: homesec       # Optional, default is "homesec"
```

---

## File Changes Summary

| File | Change |
|------|--------|
| `src/homesec/models/config.py` | Add `HomeAssistantNotifierConfig` |
| `src/homesec/plugins/notifiers/home_assistant.py` | New notifier plugin |
| `config/example.yaml` | Add home_assistant notifier example |

---

## Test Expectations

### Fixtures Needed

- `ha_notifier_config` - HomeAssistantNotifierConfig with url_env and token_env
- `ha_notifier` - Initialized HomeAssistantNotifier
- `sample_alert` - Alert with all fields populated
- `mock_aiohttp_session` - Mocked aiohttp.ClientSession

### Test Cases

**Supervisor Mode Detection**
- Given SUPERVISOR_TOKEN env var is set, when notifier starts, then supervisor mode is enabled
- Given SUPERVISOR_TOKEN not set and config has url_env/token_env, when notifier starts, then standalone mode is enabled
- Given SUPERVISOR_TOKEN not set and config missing url_env, when notifier starts, then ValueError raised

**Event Sending**
- Given notifier in standalone mode, when notify() called with alert, then POST to {url}/api/events/homesec_alert
- Given notifier in supervisor mode, when notify() called, then POST to http://supervisor/core/api/events/homesec_alert
- Given HA returns 401, when notify() called, then error logged but no exception raised
- Given HA unreachable, when notify() called, then error logged but no exception raised

**Camera Health**
- Given notifier started, when publish_camera_health("front", False), then POST homesec_camera_health event with healthy=false

---

## Verification

```bash
# Run notifier tests
pytest tests/unit/plugins/notifiers/test_home_assistant.py -v

# Manual testing (requires HA instance)
# 1. Start HomeSec with home_assistant notifier configured
# 2. Trigger a clip recording
# 3. Check HA Developer Tools > Events for homesec_* events

# Test with HA dev tools
# In HA: Developer Tools > Events > Listen to event > "homesec_alert"
```

---

## Definition of Done

- [ ] Notifier auto-detects supervisor mode via `SUPERVISOR_TOKEN`
- [ ] Zero-config works when running as HA add-on
- [ ] Standalone mode requires `url_env` and `token_env`
- [ ] Events fired: `homesec_alert`, `homesec_camera_health`, `homesec_clip_recorded`
- [ ] Notification failures don't crash the pipeline (best-effort)
- [ ] Events contain all required metadata
- [ ] Config example added
- [ ] All tests pass
