# Probe Error Discover

> 34 nodes · cohesion 0.08

## Key Concepts

- **test_api_onvif.py** (25 connections) — `tests/homesec/test_api_onvif.py`
- **_client()** (17 connections) — `tests/homesec/test_api_onvif.py`
- **test_discover_returns_found_cameras()** (4 connections) — `tests/homesec/test_api_onvif.py`
- **test_onvif_routes_require_api_key_when_auth_enabled()** (4 connections) — `tests/homesec/test_api_onvif.py`
- **test_discover_maps_runtime_failures_to_canonical_error()** (3 connections) — `tests/homesec/test_api_onvif.py`
- **test_discover_returns_empty_list_when_none_found()** (3 connections) — `tests/homesec/test_api_onvif.py`
- **test_discover_validates_attempt_bounds()** (3 connections) — `tests/homesec/test_api_onvif.py`
- **test_probe_close_failure_does_not_mask_primary_probe_error()** (3 connections) — `tests/homesec/test_api_onvif.py`
- **test_probe_close_timeout_does_not_extend_request_timeout()** (3 connections) — `tests/homesec/test_api_onvif.py`
- **test_probe_maps_client_constructor_failure_to_canonical_error()** (3 connections) — `tests/homesec/test_api_onvif.py`
- **test_probe_rejects_blank_credentials()** (3 connections) — `tests/homesec/test_api_onvif.py`
- **test_probe_rejects_blank_host()** (3 connections) — `tests/homesec/test_api_onvif.py`
- **test_probe_returns_device_info_and_profiles()** (3 connections) — `tests/homesec/test_api_onvif.py`
- **test_probe_returns_error_on_connection_failure()** (3 connections) — `tests/homesec/test_api_onvif.py`
- **test_probe_times_out_with_canonical_timeout_error()** (3 connections) — `tests/homesec/test_api_onvif.py`
- **test_probe_trims_credentials_before_client_initialization()** (3 connections) — `tests/homesec/test_api_onvif.py`
- **config()** (1 connections) — `tests/homesec/test_api_onvif.py`
- **pipeline_running()** (1 connections) — `tests/homesec/test_api_onvif.py`
- **Tests for ONVIF API endpoints (discover + probe).** (1 connections) — `tests/homesec/test_api_onvif.py`
- **POST /onvif/discover should return empty list when no cameras found.** (1 connections) — `tests/homesec/test_api_onvif.py`
- **POST /onvif/discover should map discovery runtime failures to ONVIF_DISCOVER_FAI** (1 connections) — `tests/homesec/test_api_onvif.py`
- **POST /onvif/discover should reject invalid attempt bounds via request validation** (1 connections) — `tests/homesec/test_api_onvif.py`
- **ONVIF endpoints should require API key auth when server auth is enabled.** (1 connections) — `tests/homesec/test_api_onvif.py`
- **POST /onvif/probe should return device info and merged profiles with stream URIs** (1 connections) — `tests/homesec/test_api_onvif.py`
- **POST /onvif/probe should return ONVIF_PROBE_FAILED when camera is unreachable.** (1 connections) — `tests/homesec/test_api_onvif.py`
- *... and 9 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/test_api_onvif.py`

## Audit Trail

- EXTRACTED: 101 (97%)
- INFERRED: 3 (3%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*