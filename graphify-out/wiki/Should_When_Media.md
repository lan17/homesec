# Should When Media

> 21 nodes · cohesion 0.10

## Key Concepts

- **_StubRepository** (98 connections) — `tests/homesec/test_api_routes.py`
- **test_get_clip_includes_analysis_and_alert_details()** (12 connections) — `tests/homesec/test_api_routes.py`
- **test_get_clip_media_accepts_api_key_when_auth_enabled()** (10 connections) — `tests/homesec/test_api_routes.py`
- **test_cameras_available_when_db_unavailable()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_cors_allows_credentials_for_explicit_origins()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_get_clip_media_missing_returns_404()** (8 connections) — `tests/homesec/test_api_routes.py`
- **test_update_camera_null_patch_value_clears_optional_source_field()** (7 connections) — `tests/homesec/test_api_routes.py`
- **test_update_camera_rejects_redacted_source_config_patch()** (7 connections) — `tests/homesec/test_api_routes.py`
- **.delete_clip()** (2 connections) — `tests/homesec/test_api_routes.py`
- **.get_clip()** (2 connections) — `tests/homesec/test_api_routes.py`
- **CORS should allow credentials when origins are explicit.** (1 connections) — `tests/homesec/test_api_routes.py`
- **GET /clips/{id} should include analysis, detection, and alert fields.** (1 connections) — `tests/homesec/test_api_routes.py`
- **GET /clips/{id}/media should return 404 when clip is missing.** (1 connections) — `tests/homesec/test_api_routes.py`
- **GET /clips/{id}/media should allow API-key authenticated playback.** (1 connections) — `tests/homesec/test_api_routes.py`
- **Camera config endpoint should remain available when DB is down.** (1 connections) — `tests/homesec/test_api_routes.py`
- **PATCH /cameras/{name} should reject redacted placeholder source_config values.** (1 connections) — `tests/homesec/test_api_routes.py`
- **PATCH /cameras/{name} should clear optional source fields when null is provided.** (1 connections) — `tests/homesec/test_api_routes.py`
- **.count_alerts_since()** (1 connections) — `tests/homesec/test_api_routes.py`
- **.count_clips_since()** (1 connections) — `tests/homesec/test_api_routes.py`
- **.__init__()** (1 connections) — `tests/homesec/test_api_routes.py`
- **.ping()** (1 connections) — `tests/homesec/test_api_routes.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/test_api_routes.py`

## Audit Trail

- EXTRACTED: 151 (87%)
- INFERRED: 22 (13%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*