# When Should Missing

> 21 nodes · cohesion 0.10

## Key Concepts

- **_StubStorage** (96 connections) — `tests/homesec/test_api_routes.py`
- **test_create_clip_media_token_missing_storage_uri_returns_409()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_get_config_returns_empty_mapping_when_redaction_result_is_not_mapping()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_ui_serving_fails_fast_when_dist_missing()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_cors_disables_credentials_for_wildcard_origins()** (7 connections) — `tests/homesec/test_api_routes.py`
- **test_create_camera_apply_changes_does_not_request_reload_when_mutation_fails()** (7 connections) — `tests/homesec/test_api_routes.py`
- **test_create_clip_media_token_missing_clip_returns_404()** (7 connections) — `tests/homesec/test_api_routes.py`
- **test_update_camera_invalid_config_returns_400()** (7 connections) — `tests/homesec/test_api_routes.py`
- **test_update_camera_missing_returns_404()** (7 connections) — `tests/homesec/test_api_routes.py`
- **test_update_camera_supports_source_backend_switch()** (7 connections) — `tests/homesec/test_api_routes.py`
- **GET /config should fallback to empty object when redaction returns non-mapping.** (1 connections) — `tests/homesec/test_api_routes.py`
- **CORS should disable credentials when wildcard origins are configured.** (1 connections) — `tests/homesec/test_api_routes.py`
- **App creation should fail fast when configured UI dist is missing.** (1 connections) — `tests/homesec/test_api_routes.py`
- **POST /clips/{id}/media-token should return 404 when clip is missing.** (1 connections) — `tests/homesec/test_api_routes.py`
- **POST /clips/{id}/media-token should return 409 when media is unavailable.** (1 connections) — `tests/homesec/test_api_routes.py`
- **POST /cameras should avoid runtime reload calls when mutation fails early.** (1 connections) — `tests/homesec/test_api_routes.py`
- **PATCH /cameras/{name} should return 400 for invalid config.** (1 connections) — `tests/homesec/test_api_routes.py`
- **PATCH /cameras/{name} should return 404 when missing.** (1 connections) — `tests/homesec/test_api_routes.py`
- **PATCH /cameras/{name} should support changing source_backend with valid config.** (1 connections) — `tests/homesec/test_api_routes.py`
- **.__init__()** (1 connections) — `tests/homesec/test_api_routes.py`
- **.ping()** (1 connections) — `tests/homesec/test_api_routes.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/test_api_routes.py`

## Audit Trail

- EXTRACTED: 156 (90%)
- INFERRED: 17 (10%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*