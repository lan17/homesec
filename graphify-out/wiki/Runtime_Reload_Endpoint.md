# Runtime Reload Endpoint

> 22 nodes · cohesion 0.16

## Key Concepts

- **RuntimeReloadRequest** (34 connections) — `src/homesec/runtime/models.py`
- **_StubRuntimeApp** (15 connections) — `tests/homesec/test_api_runtime_routes.py`
- **test_api_runtime_routes.py** (11 connections) — `tests/homesec/test_api_runtime_routes.py`
- **_client()** (7 connections) — `tests/homesec/test_api_runtime_routes.py`
- **test_runtime_reload_endpoint_returns_400_for_invalid_config()** (7 connections) — `tests/homesec/test_api_runtime_routes.py`
- **test_runtime_reload_endpoint_returns_422_for_unprocessable_config()** (7 connections) — `tests/homesec/test_api_runtime_routes.py`
- **test_runtime_reload_endpoint_returns_busy_conflict()** (6 connections) — `tests/homesec/test_api_runtime_routes.py`
- **test_runtime_reload_endpoint_returns_success_payload()** (6 connections) — `tests/homesec/test_api_runtime_routes.py`
- **test_runtime_status_endpoint_returns_manager_snapshot()** (6 connections) — `tests/homesec/test_api_runtime_routes.py`
- **bootstrap_mode()** (1 connections) — `tests/homesec/test_api_runtime_routes.py`
- **config()** (1 connections) — `tests/homesec/test_api_runtime_routes.py`
- **Tests for runtime control-plane API routes.** (1 connections) — `tests/homesec/test_api_runtime_routes.py`
- **POST /runtime/reload should reject when a reload is already in progress.** (1 connections) — `tests/homesec/test_api_runtime_routes.py`
- **POST /runtime/reload should return 400 for invalid reload config payloads.** (1 connections) — `tests/homesec/test_api_runtime_routes.py`
- **POST /runtime/reload should return 422 for semantically invalid config.** (1 connections) — `tests/homesec/test_api_runtime_routes.py`
- **GET /runtime/status should return runtime state fields.** (1 connections) — `tests/homesec/test_api_runtime_routes.py`
- **POST /runtime/reload should return acceptance payload without waiting.** (1 connections) — `tests/homesec/test_api_runtime_routes.py`
- **server_config()** (1 connections) — `tests/homesec/test_api_runtime_routes.py`
- **.get_runtime_status()** (1 connections) — `tests/homesec/test_api_runtime_routes.py`
- **.request_runtime_reload()** (1 connections) — `tests/homesec/test_api_runtime_routes.py`
- **.wait_for_runtime_reload()** (1 connections) — `tests/homesec/test_api_runtime_routes.py`
- **Result of requesting a runtime reload.** (1 connections) — `src/homesec/runtime/models.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/runtime/models.py`
- `tests/homesec/test_api_runtime_routes.py`

## Audit Trail

- EXTRACTED: 62 (55%)
- INFERRED: 50 (45%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*