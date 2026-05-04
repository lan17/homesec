# Returns Clips Media

> 26 nodes · cohesion 0.08

## Key Concepts

- **.get()** (48 connections) — `tests/homesec/test_api_routes.py`
- **test_get_clip_media_accepts_valid_media_token_when_auth_enabled()** (10 connections) — `tests/homesec/test_api_routes.py`
- **test_auth_required_when_enabled()** (9 connections) — `tests/homesec/test_api_routes.py`
- **test_get_clip_media_defaults_filename_suffix_when_storage_uri_has_no_extension()** (9 connections) — `tests/homesec/test_api_routes.py`
- **test_get_clip_media_missing_storage_uri_returns_409()** (9 connections) — `tests/homesec/test_api_routes.py`
- **test_db_health_probe_is_cached_for_burst_requests()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_db_unavailable_returns_503()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_get_camera()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_get_camera_missing_returns_404()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_get_clip_missing_returns_404()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_get_config_redacts_sensitive_fields()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_list_clips_forwards_detected_filter_to_repository()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_list_clips_invalid_query_returns_canonical_validation_error()** (8 connections) — `tests/homesec/test_api_routes.py`
- **.get_source()** (2 connections) — `tests/homesec/test_api_routes.py`
- **GET /config should redact direct secret values while preserving *_env references** (1 connections) — `tests/homesec/test_api_routes.py`
- **GET /clips should forward the detected filter to the repository.** (1 connections) — `tests/homesec/test_api_routes.py`
- **GET /clips should return canonical envelope for query validation failures.** (1 connections) — `tests/homesec/test_api_routes.py`
- **GET /clips/{id} should return 404 when missing.** (1 connections) — `tests/homesec/test_api_routes.py`
- **GET /clips/{id}/media should return 409 when media is unavailable.** (1 connections) — `tests/homesec/test_api_routes.py`
- **GET /clips/{id}/media should default filename suffix to .mp4 when storage URI la** (1 connections) — `tests/homesec/test_api_routes.py`
- **GET /clips/{id}/media should allow token-authenticated browser playback.** (1 connections) — `tests/homesec/test_api_routes.py`
- **DB-backed endpoints should return 503 when DB is down.** (1 connections) — `tests/homesec/test_api_routes.py`
- **DB-backed endpoints should reuse a recent health probe result.** (1 connections) — `tests/homesec/test_api_routes.py`
- **Auth should be enforced for non-public endpoints.** (1 connections) — `tests/homesec/test_api_routes.py`
- **GET /cameras/{name} should return a camera.** (1 connections) — `tests/homesec/test_api_routes.py`
- *... and 1 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/test_api_routes.py`

## Audit Trail

- EXTRACTED: 157 (96%)
- INFERRED: 6 (4%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*