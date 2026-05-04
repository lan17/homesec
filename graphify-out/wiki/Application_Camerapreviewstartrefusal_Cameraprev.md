# Application Camerapreviewstartrefusal Cameraprev

> 149 nodes · cohesion 0.03

## Key Concepts

- **Config** (106 connections) — `src/homesec/models/config.py`
- **Application** (105 connections) — `src/homesec/app.py`
- **CameraPreviewStartRefusal** (47 connections) — `src/homesec/runtime/models.py`
- **CameraPreviewStopResult** (46 connections) — `src/homesec/runtime/models.py`
- **_StubRuntimeController** (43 connections) — `tests/homesec/test_app.py`
- **_StubStorage** (43 connections) — `tests/homesec/test_app.py`
- **_StubRuntimeManagerStatus** (38 connections) — `tests/homesec/test_app.py`
- **_FakeProcess** (37 connections) — `tests/homesec/test_app.py`
- **_StubStateStore** (37 connections) — `tests/homesec/test_app.py`
- **PluginType** (37 connections) — `src/homesec/plugins/registry.py`
- **_RecordingEventStore** (36 connections) — `tests/homesec/test_app.py`
- **SubprocessRuntimeHandle** (36 connections) — `src/homesec/runtime/subprocess_controller.py`
- **_StubPostgresBackupManager** (35 connections) — `tests/homesec/test_app.py`
- **RuntimeController** (35 connections) — `src/homesec/runtime/controller.py`
- **RuntimeStatusSnapshot** (34 connections) — `src/homesec/runtime/models.py`
- **ConfigError** (33 connections) — `src/homesec/config/loader.py`
- **_CameraSourceSnapshot** (31 connections) — `src/homesec/app.py`
- **RuntimeState** (31 connections) — `src/homesec/runtime/models.py`
- **test_app.py** (29 connections) — `tests/homesec/test_app.py`
- **RuntimeCameraStatus** (26 connections) — `src/homesec/runtime/models.py`
- **ConfigErrorCode** (24 connections) — `src/homesec/config/loader.py`
- **_make_config()** (22 connections) — `tests/homesec/test_app.py`
- **models.py** (16 connections) — `src/homesec/runtime/models.py`
- **test_camera_health_degrades_when_runtime_heartbeat_is_stale()** (10 connections) — `tests/homesec/test_app.py`
- **test_get_runtime_status_preserves_reloading_when_heartbeat_is_stale()** (10 connections) — `tests/homesec/test_app.py`
- *... and 124 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/api/routes/health.py`
- `src/homesec/api/routes/runtime.py`
- `src/homesec/app.py`
- `src/homesec/config/loader.py`
- `src/homesec/models/config.py`
- `src/homesec/plugins/registry.py`
- `src/homesec/runtime/controller.py`
- `src/homesec/runtime/models.py`
- `src/homesec/runtime/subprocess_controller.py`
- `tests/homesec/test_app.py`
- `tests/homesec/test_config.py`
- `tests/homesec/test_plugin_registration.py`

## Audit Trail

- EXTRACTED: 400 (33%)
- INFERRED: 822 (67%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*