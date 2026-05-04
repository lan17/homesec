# Canonical Error Exception

> 10 nodes · cohesion 0.24

## Key Concepts

- **register_exception_handlers()** (6 connections) — `src/homesec/api/errors.py`
- **test_api_errors.py** (4 connections) — `tests/homesec/test_api_errors.py`
- **test_http_exception_dict_detail_is_normalized_to_canonical_envelope()** (3 connections) — `tests/homesec/test_api_errors.py`
- **test_missing_homesec_app_maps_to_app_not_initialized_error()** (3 connections) — `tests/homesec/test_api_errors.py`
- **test_unhandled_exception_maps_to_internal_server_error_envelope()** (3 connections) — `tests/homesec/test_api_errors.py`
- **Register canonical API error handlers.** (1 connections) — `src/homesec/api/errors.py`
- **Tests for canonical API error handlers.** (1 connections) — `tests/homesec/test_api_errors.py`
- **HTTPException payloads with dict detail should map to canonical envelope.** (1 connections) — `tests/homesec/test_api_errors.py`
- **Unhandled route exceptions should return canonical internal server error.** (1 connections) — `tests/homesec/test_api_errors.py`
- **Dependency failure for missing app state should map to canonical 503 payload.** (1 connections) — `tests/homesec/test_api_errors.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/api/errors.py`
- `tests/homesec/test_api_errors.py`

## Audit Trail

- EXTRACTED: 17 (71%)
- INFERRED: 7 (29%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*