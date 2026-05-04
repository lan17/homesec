# App Exception Envelope

> 12 nodes · cohesion 0.24

## Key Concepts

- **register_exception_handlers** (4 connections) — `tests/homesec/test_api_errors.py`
- **test_http_exception_dict_detail_is_normalized_to_canonical_envelope** (4 connections) — `tests/homesec/test_api_errors.py`
- **test_missing_homesec_app_maps_to_app_not_initialized_error** (4 connections) — `tests/homesec/test_api_errors.py`
- **test_unhandled_exception_maps_to_internal_server_error_envelope** (4 connections) — `tests/homesec/test_api_errors.py`
- **FastAPI** (3 connections) — `tests/homesec/test_api_errors.py`
- **_needs_app** (3 connections) — `tests/homesec/test_api_errors.py`
- **TestClient** (3 connections) — `tests/homesec/test_api_errors.py`
- **_boom** (2 connections) — `tests/homesec/test_api_errors.py`
- **HTTPException** (2 connections) — `tests/homesec/test_api_errors.py`
- **_crash** (1 connections) — `tests/homesec/test_api_errors.py`
- **Depends** (1 connections) — `tests/homesec/test_api_errors.py`
- **get_homesec_app** (1 connections) — `tests/homesec/test_api_errors.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/test_api_errors.py`

## Audit Trail

- EXTRACTED: 28 (88%)
- INFERRED: 4 (12%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*