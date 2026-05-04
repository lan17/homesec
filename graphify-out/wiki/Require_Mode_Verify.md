# Require Mode Verify

> 22 nodes · cohesion 0.16

## Key Concepts

- **dependencies.py** (14 connections) — `src/homesec/api/dependencies.py`
- **_DatabaseProbeCache** (8 connections) — `src/homesec/api/dependencies.py`
- **verify_media_access()** (6 connections) — `src/homesec/api/dependencies.py`
- **verify_preview_access()** (6 connections) — `src/homesec/api/dependencies.py`
- **_get_server_config()** (5 connections) — `src/homesec/api/dependencies.py`
- **_require_api_key_value()** (5 connections) — `src/homesec/api/dependencies.py`
- **verify_api_key()** (5 connections) — `src/homesec/api/dependencies.py`
- **_is_bootstrap_mode()** (4 connections) — `src/homesec/api/dependencies.py`
- **_parse_bearer_token()** (4 connections) — `src/homesec/api/dependencies.py`
- **require_database()** (4 connections) — `src/homesec/api/dependencies.py`
- **_get_database_probe_cache()** (3 connections) — `src/homesec/api/dependencies.py`
- **require_bootstrap_mode()** (3 connections) — `src/homesec/api/dependencies.py`
- **require_normal_mode()** (3 connections) — `src/homesec/api/dependencies.py`
- **get_homesec_app()** (2 connections) — `src/homesec/api/dependencies.py`
- **FastAPI dependency helpers.** (1 connections) — `src/homesec/api/dependencies.py`
- **Authorize preview playback requests with API key or short-lived preview token.** (1 connections) — `src/homesec/api/dependencies.py`
- **Ensure the database is reachable for data endpoints.** (1 connections) — `src/homesec/api/dependencies.py`
- **Ensure runtime-dependent routes are unavailable in bootstrap mode.** (1 connections) — `src/homesec/api/dependencies.py`
- **Ensure setup-finalize route is only available during bootstrap mode.** (1 connections) — `src/homesec/api/dependencies.py`
- **Get the HomeSec Application instance from request state.** (1 connections) — `src/homesec/api/dependencies.py`
- **Verify API key if authentication is enabled.** (1 connections) — `src/homesec/api/dependencies.py`
- **Authorize media playback requests with API key or short-lived media token.** (1 connections) — `src/homesec/api/dependencies.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/api/dependencies.py`

## Audit Trail

- EXTRACTED: 72 (90%)
- INFERRED: 8 (10%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*