# Should Serving Routes

> 20 nodes · cohesion 0.11

## Key Concepts

- **test_api_routes.py** (92 connections) — `tests/homesec/test_api_routes.py`
- **test_ui_serving_serves_spa_shell_without_shadowing_api()** (10 connections) — `tests/homesec/test_api_routes.py`
- **test_get_clip_media_success_cleans_temp_directory()** (9 connections) — `tests/homesec/test_api_routes.py`
- **test_ui_serving_catch_all_rejects_path_traversal_input()** (9 connections) — `tests/homesec/test_api_routes.py`
- **test_db_health_probe_rechecks_after_ttl_and_detects_outage()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_list_clips_forwards_status_alias_to_repository()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_ui_serving_fails_fast_when_assets_directory_missing()** (8 connections) — `tests/homesec/test_api_routes.py`
- **_write_ui_dist()** (3 connections) — `tests/homesec/test_api_routes.py`
- **bootstrap_mode()** (1 connections) — `tests/homesec/test_api_routes.py`
- **config()** (1 connections) — `tests/homesec/test_api_routes.py`
- **pipeline_running()** (1 connections) — `tests/homesec/test_api_routes.py`
- **Tests for FastAPI camera and clip routes.** (1 connections) — `tests/homesec/test_api_routes.py`
- **UI serving should return SPA shell while preserving API and docs routes.** (1 connections) — `tests/homesec/test_api_routes.py`
- **App creation should fail fast when assets directory is missing.** (1 connections) — `tests/homesec/test_api_routes.py`
- **SPA catch-all should reject traversal-like full_path values.** (1 connections) — `tests/homesec/test_api_routes.py`
- **GET /clips should map query param status to repository status argument.** (1 connections) — `tests/homesec/test_api_routes.py`
- **GET /clips/{id}/media should clean temporary media files after response.** (1 connections) — `tests/homesec/test_api_routes.py`
- **DB-backed endpoints should re-check health after TTL and fail closed when DB dro** (1 connections) — `tests/homesec/test_api_routes.py`
- **server_config()** (1 connections) — `tests/homesec/test_api_routes.py`
- **setup_test_connection_lock()** (1 connections) — `tests/homesec/test_api_routes.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/test_api_routes.py`

## Audit Trail

- EXTRACTED: 154 (97%)
- INFERRED: 5 (3%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*