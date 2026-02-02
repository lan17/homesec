# HomeSec + Home Assistant: Implementation Plan

This document provides an overview of the Home Assistant integration for HomeSec. Detailed implementation for each phase is in separate documents.

---

## Decision Snapshot (2026-02-01)

- **Approach**: Add-on + native integration with HomeSec as the runtime
- **API stack**: FastAPI, async endpoints only, async SQLAlchemy only
- **Config storage**: Override YAML file is source of truth for dynamic config. Base YAML is bootstrap-only.
- **Config merge**: Multiple YAML files loaded left → right; rightmost wins. Dicts deep-merge (recursive), lists merge (union).
- **Single instance**: HA integration assumes one HomeSec instance (`single_config_entry`)
- **Secrets**: Never stored in HomeSec config; only env var names are persisted
- **Repository pattern**: API reads/writes go through `ClipRepository` (no direct `StateStore`/`EventStore` access)
- **Tests**: Given/When/Then comments required for all new tests
- **P0 priority**: Recording + uploading must keep working even if Postgres is down
- **Real-time events**: Use HA Events API (not MQTT). Add-on gets `SUPERVISOR_TOKEN` automatically
- **No MQTT required**: Integration uses HA Events API directly. Existing MQTT notifier remains for Node-RED/other systems
- **Last write wins**: No optimistic concurrency in v1 (single HA instance assumption)
- **API during Postgres outage**: Return 503 Service Unavailable
- **Restart acceptable**: API writes validated config to disk and returns `restart_required`; HA can trigger restart

---

## Execution Order

| Phase | Document | Dependencies | Estimated Effort |
|-------|----------|--------------|------------------|
| 1 | [REST API](./ha-phase-1-rest-api.md) | None | 5-7 days |
| 2 | [HA Notifier](./ha-phase-2-ha-notifier.md) | None | 2-3 days |
| 3 | [Add-on](./ha-phase-3-addon.md) | Phase 1, 2 | 3-4 days |
| 4 | [HA Integration](./ha-phase-4-ha-integration.md) | Phase 1, 2 | 7-10 days |
| 5 | [Advanced Features](./ha-phase-5-advanced.md) | Phase 4 | TBD |

**Total: 17-24 days** (v1 core)

**Parallel work possible**: Phase 1 and Phase 2 can be done in parallel.

**Note**: Phase 0 (Prerequisites) was merged into Phase 1. Phase 5 scope TBD.

---

## Repository Structure

All code lives in the main `homesec` monorepo:

```
homesec/
├── repository.json                     # HA Add-on repo manifest (at repo root)
├── src/homesec/                        # Main Python package (PyPI)
│   ├── api/                            # Phase 1: REST API
│   │   ├── __init__.py
│   │   ├── server.py
│   │   ├── dependencies.py
│   │   └── routes/
│   ├── config/                         # Phase 1: Config management
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── manager.py
│   ├── plugins/
│   │   └── notifiers/
│   │       └── home_assistant.py       # Phase 2: HA Events API notifier
│   └── ...existing code...
│
├── homeassistant/                      # All HA-specific code
│   ├── integration/                    # Phase 4: Custom component (HACS)
│   │   └── custom_components/homesec/
│   └── addon/                          # Phase 3: Add-on (HA Supervisor)
│       └── homesec/
│
└── tests/
    ├── unit/
    │   ├── api/
    │   └── plugins/notifiers/
    └── integration/
```

**Distribution:**
- **PyPI**: `src/homesec/` published as `homesec` package
- **HACS**: Users point to `homeassistant/integration/`
- **Add-on Store**: Users add repo URL (repository.json at root points to `homeassistant/addon/homesec/`)

---

## Key Interfaces

These interfaces are defined across phases. See individual phase docs for details.

### CameraConfig (Phase 1)
- `enabled: bool = True` field (allows disabling cameras via API)

### ConfigManager (Phase 1)
- `get_config() -> Config`
- `update_camera(...) -> ConfigUpdateResult`
- `add_camera(...) -> ConfigUpdateResult`
- `remove_camera(...) -> ConfigUpdateResult`

### ClipRepository Extensions (Phase 1)
- `get_clip(clip_id) -> Clip | None`
- `list_clips(...) -> tuple[list[Clip], int]` (supports `alerted`, `risk_level`, `activity_type` filters)
- `delete_clip(clip_id) -> None`
- `count_clips_since(since: datetime) -> int`
- `count_alerts_since(since: datetime) -> int`

---

## Migration Guide

### From Standalone to Add-on

1. Export current `config.yaml`
2. Install HomeSec add-on
3. Copy config to `/config/homesec/config.yaml`
4. Create `/data/overrides.yaml` for HA-managed config
5. Update database URL if using external Postgres
6. Start add-on

### From MQTT Notifier to Native Integration

1. Keep existing MQTT notifier configuration (for Node-RED, etc.)
2. Install custom integration via HACS
3. Configure integration with HomeSec URL
4. Native entities will be created
5. Optionally disable MQTT notifier if no longer needed
