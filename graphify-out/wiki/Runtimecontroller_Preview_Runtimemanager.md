# Runtimecontroller Preview Runtimemanager

> 28 nodes · cohesion 0.08

## Key Concepts

- **RuntimeController** (9 connections) — `src/homesec/runtime/controller.py`
- **RuntimeController** (5 connections) — `src/homesec/runtime/manager.py`
- **RuntimeController.build_candidate** (3 connections) — `src/homesec/runtime/controller.py`
- **RuntimeController.ensure_preview_active** (3 connections) — `src/homesec/runtime/controller.py`
- **ManagedRuntime** (3 connections) — `src/homesec/runtime/controller.py`
- **RuntimeManager.ensure_preview_active** (3 connections) — `src/homesec/runtime/manager.py`
- **RuntimeManager** (3 connections) — `src/homesec/runtime/manager.py`
- **RuntimeManager.start_initial_runtime** (3 connections) — `src/homesec/runtime/manager.py`
- **CameraPreviewStatus** (2 connections) — `src/homesec/runtime/controller.py`
- **RuntimeController.force_stop_preview** (2 connections) — `src/homesec/runtime/controller.py`
- **RuntimeController.get_preview_status** (2 connections) — `src/homesec/runtime/controller.py`
- **RuntimeController.shutdown_runtime** (2 connections) — `src/homesec/runtime/controller.py`
- **RuntimeController.start_runtime** (2 connections) — `src/homesec/runtime/controller.py`
- **CameraPreviewStatus** (2 connections) — `src/homesec/runtime/manager.py`
- **RuntimeManager.get_preview_status** (2 connections) — `src/homesec/runtime/manager.py`
- **RuntimeStatusSnapshot** (2 connections) — `src/homesec/runtime/manager.py`
- **sanitize_runtime_error** (2 connections) — `src/homesec/runtime/manager.py`
- **RuntimeController** (2 connections) — `src/homesec/runtime/subprocess_controller.py`
- **CameraPreviewStartRefusal** (1 connections) — `src/homesec/runtime/controller.py`
- **CameraPreviewStopResult** (1 connections) — `src/homesec/runtime/controller.py`
- **Config** (1 connections) — `src/homesec/runtime/controller.py`
- **RuntimeController.note_preview_viewer_activity** (1 connections) — `src/homesec/runtime/controller.py`
- **sanitize_runtime_error** (1 connections) — `src/homesec/runtime/errors.py`
- **CameraPreviewStartRefusal** (1 connections) — `src/homesec/runtime/manager.py`
- **RuntimeManager.get_status** (1 connections) — `src/homesec/runtime/manager.py`
- *... and 3 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/runtime/controller.py`
- `src/homesec/runtime/errors.py`
- `src/homesec/runtime/manager.py`
- `src/homesec/runtime/subprocess_controller.py`

## Audit Trail

- EXTRACTED: 62 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*