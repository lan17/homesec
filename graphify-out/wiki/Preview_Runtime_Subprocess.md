# Preview Runtime Subprocess

> 79 nodes · cohesion 0.05

## Key Concepts

- **SubprocessRuntimeController** (54 connections) — `src/homesec/runtime/subprocess_controller.py`
- **_FakeProcess** (30 connections) — `tests/homesec/test_runtime_subprocess_controller.py`
- **PreviewRuntimeUnavailableError** (26 connections) — `src/homesec/runtime/errors.py`
- **PreviewCameraNotFoundError** (22 connections) — `src/homesec/runtime/errors.py`
- **_make_config()** (21 connections) — `tests/homesec/test_runtime_subprocess_controller.py`
- **test_runtime_subprocess_controller.py** (14 connections) — `tests/homesec/test_runtime_subprocess_controller.py`
- **PreviewSessionResponse** (12 connections) — `src/homesec/api/routes/preview.py`
- **PreviewStatusResponse** (12 connections) — `src/homesec/api/routes/preview.py`
- **PreviewStopResponse** (12 connections) — `src/homesec/api/routes/preview.py`
- **.force_stop_preview()** (12 connections) — `src/homesec/runtime/subprocess_controller.py`
- **.get_preview_status()** (12 connections) — `src/homesec/runtime/subprocess_controller.py`
- **subprocess_controller.py** (12 connections) — `src/homesec/runtime/subprocess_controller.py`
- **sanitize_runtime_error()** (11 connections) — `src/homesec/runtime/errors.py`
- **.ensure_preview_active()** (10 connections) — `src/homesec/runtime/subprocess_controller.py`
- **.note_preview_viewer_activity()** (10 connections) — `src/homesec/runtime/subprocess_controller.py`
- **_require_handle()** (7 connections) — `src/homesec/runtime/subprocess_controller.py`
- **._send_command()** (7 connections) — `src/homesec/runtime/subprocess_controller.py`
- **._spawn_and_wait_started()** (7 connections) — `src/homesec/runtime/subprocess_controller.py`
- **._stop_handle()** (7 connections) — `src/homesec/runtime/subprocess_controller.py`
- **RuntimePreviewError** (6 connections) — `src/homesec/runtime/errors.py`
- **_runtime_preview_status()** (6 connections) — `src/homesec/runtime/subprocess_controller.py`
- **.build_candidate()** (6 connections) — `src/homesec/runtime/subprocess_controller.py`
- **._stale_preview_status()** (6 connections) — `src/homesec/runtime/subprocess_controller.py`
- **.start_runtime()** (6 connections) — `src/homesec/runtime/subprocess_controller.py`
- **errors.py** (6 connections) — `src/homesec/runtime/errors.py`
- *... and 54 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/api/routes/preview.py`
- `src/homesec/runtime/errors.py`
- `src/homesec/runtime/models.py`
- `src/homesec/runtime/subprocess_controller.py`
- `tests/homesec/test_runtime_subprocess_controller.py`

## Audit Trail

- EXTRACTED: 286 (61%)
- INFERRED: 181 (39%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*