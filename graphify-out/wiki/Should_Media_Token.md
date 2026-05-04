# Should Media Token

> 19 nodes · cohesion 0.11

## Key Concepts

- **_client()** (78 connections) — `tests/homesec/test_api_routes.py`
- **create_app()** (15 connections) — `src/homesec/api/server.py`
- **test_create_clip_media_token_rejects_token_only_auth()** (9 connections) — `tests/homesec/test_api_routes.py`
- **test_get_clip_media_proxy_success_returns_inline_video()** (9 connections) — `tests/homesec/test_api_routes.py`
- **test_create_camera_apply_changes_triggers_runtime_reload()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_health_endpoints_report_setup_required_in_bootstrap_mode()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_list_cameras_redacts_sensitive_source_config()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_update_camera_apply_changes_conflict_returns_409()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_create_camera_duplicate_returns_409()** (7 connections) — `tests/homesec/test_api_routes.py`
- **test_update_camera_returns_404_when_camera_removed_after_update()** (7 connections) — `tests/homesec/test_api_routes.py`
- **Create the FastAPI application.** (1 connections) — `src/homesec/api/server.py`
- **GET /clips/{id}/media should proxy media inline for playback.** (1 connections) — `tests/homesec/test_api_routes.py`
- **POST /clips/{id}/media-token should reject token-only access to prevent token ch** (1 connections) — `tests/homesec/test_api_routes.py`
- **Health endpoints should stay available and report setup mode during bootstrap.** (1 connections) — `tests/homesec/test_api_routes.py`
- **POST /cameras should return 409 for duplicate names.** (1 connections) — `tests/homesec/test_api_routes.py`
- **POST /cameras should optionally apply runtime reload in the same request.** (1 connections) — `tests/homesec/test_api_routes.py`
- **GET /cameras should redact sensitive source_config fields.** (1 connections) — `tests/homesec/test_api_routes.py`
- **PATCH /cameras/{name} should return 404 when camera disappears after update.** (1 connections) — `tests/homesec/test_api_routes.py`
- **PATCH /cameras should surface runtime-reload conflict when apply_changes is requ** (1 connections) — `tests/homesec/test_api_routes.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/api/server.py`
- `tests/homesec/test_api_routes.py`

## Audit Trail

- EXTRACTED: 149 (90%)
- INFERRED: 17 (10%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*