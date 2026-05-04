# When Repository Clips

> 25 nodes · cohesion 0.11

## Key Concepts

- **_StubApp** (99 connections) — `tests/homesec/test_api_routes.py`
- **.delete()** (11 connections) — `tests/homesec/test_api_routes.py`
- **test_delete_camera_apply_changes_triggers_runtime_reload()** (9 connections) — `tests/homesec/test_api_routes.py`
- **test_delete_clip_storage_failure_returns_500()** (9 connections) — `tests/homesec/test_api_routes.py`
- **test_delete_camera()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_delete_camera_apply_changes_skips_reload_when_mutation_fails()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_delete_camera_missing_returns_404()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_delete_clip_missing_returns_404()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_delete_clip_restores_storage_uri_when_repository_clears_it()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_delete_clip_returns_404_when_repository_delete_races()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_delete_clip_returns_500_when_repository_delete_fails_unexpectedly()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_delete_clip_success_removes_storage()** (8 connections) — `tests/homesec/test_api_routes.py`
- **DELETE /clips/{id} should return 500 if storage deletion fails.** (1 connections) — `tests/homesec/test_api_routes.py`
- **DELETE /clips/{id} should return 404 when missing.** (1 connections) — `tests/homesec/test_api_routes.py`
- **DELETE /clips/{id} should return clip data on success.** (1 connections) — `tests/homesec/test_api_routes.py`
- **DELETE /clips/{id} should return 404 when repository delete fails after initial** (1 connections) — `tests/homesec/test_api_routes.py`
- **DELETE /clips/{id} should map unexpected repository failures to canonical 500.** (1 connections) — `tests/homesec/test_api_routes.py`
- **DELETE /clips/{id} should preserve storage URI in response when repository omits** (1 connections) — `tests/homesec/test_api_routes.py`
- **DELETE /cameras/{name} should remove a camera.** (1 connections) — `tests/homesec/test_api_routes.py`
- **DELETE /cameras/{name} should return 404 when missing.** (1 connections) — `tests/homesec/test_api_routes.py`
- **DELETE /cameras should optionally request runtime reload when apply_changes is t** (1 connections) — `tests/homesec/test_api_routes.py`
- **DELETE /cameras should avoid runtime reload calls when camera deletion fails.** (1 connections) — `tests/homesec/test_api_routes.py`
- **.activate_setup_config()** (1 connections) — `tests/homesec/test_api_routes.py`
- **.request_restart()** (1 connections) — `tests/homesec/test_api_routes.py`
- **.request_runtime_reload()** (1 connections) — `tests/homesec/test_api_routes.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/test_api_routes.py`

## Audit Trail

- EXTRACTED: 185 (90%)
- INFERRED: 20 (10%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*