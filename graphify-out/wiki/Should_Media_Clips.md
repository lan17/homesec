# Should Media Clips

> 19 nodes · cohesion 0.11

## Key Concepts

- **_write_config()** (80 connections) — `tests/homesec/test_api_routes.py`
- **test_auth_env_missing_returns_500()** (9 connections) — `tests/homesec/test_api_routes.py`
- **test_create_clip_media_token_auth_disabled_returns_direct_media_url()** (9 connections) — `tests/homesec/test_api_routes.py`
- **test_get_clip_media_storage_failure_returns_502()** (9 connections) — `tests/homesec/test_api_routes.py`
- **test_list_clips_forwards_alerted_filter_to_repository()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_list_clips_rejects_inverted_time_window()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_update_camera_apply_changes_propagates_runtime_config_error()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_create_camera()** (7 connections) — `tests/homesec/test_api_routes.py`
- **test_system_restart_endpoint_removed_returns_404()** (7 connections) — `tests/homesec/test_api_routes.py`
- **test_update_camera()** (7 connections) — `tests/homesec/test_api_routes.py`
- **GET /clips should reject when since is greater than until.** (1 connections) — `tests/homesec/test_api_routes.py`
- **GET /clips should forward the alerted filter to the repository.** (1 connections) — `tests/homesec/test_api_routes.py`
- **GET /clips/{id}/media should return 502 when storage fetch fails.** (1 connections) — `tests/homesec/test_api_routes.py`
- **POST /clips/{id}/media-token should return direct media path when auth is disabl** (1 connections) — `tests/homesec/test_api_routes.py`
- **POST /system/restart should be removed from the API surface.** (1 connections) — `tests/homesec/test_api_routes.py`
- **Auth should return 500 when API key is not configured.** (1 connections) — `tests/homesec/test_api_routes.py`
- **POST /cameras should create a camera.** (1 connections) — `tests/homesec/test_api_routes.py`
- **PATCH /cameras/{name} should update camera fields.** (1 connections) — `tests/homesec/test_api_routes.py`
- **PATCH /cameras should map runtime config errors when apply_changes reload fails.** (1 connections) — `tests/homesec/test_api_routes.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/test_api_routes.py`

## Audit Trail

- EXTRACTED: 155 (96%)
- INFERRED: 6 (4%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*