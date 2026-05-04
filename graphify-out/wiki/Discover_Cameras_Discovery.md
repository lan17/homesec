# Discover Cameras Discovery

> 29 nodes · cohesion 0.10

## Key Concepts

- **discover_cameras()** (11 connections) — `src/homesec/onvif/discovery.py`
- **discovery.py** (10 connections) — `src/homesec/onvif/discovery.py`
- **test_onvif_discovery.py** (8 connections) — `tests/homesec/test_onvif_discovery.py`
- **_parse_discovery_services()** (7 connections) — `src/homesec/onvif/discovery.py`
- **test_discover_cameras_probes_each_type_separately_and_deduplicates()** (4 connections) — `tests/homesec/test_onvif_discovery.py`
- **test_discover_cameras_logs_warning_when_stop_raises()** (3 connections) — `tests/homesec/test_onvif_discovery.py`
- **test_discover_cameras_rejects_invalid_attempt_count()** (3 connections) — `tests/homesec/test_onvif_discovery.py`
- **test_discover_cameras_runs_all_attempts()** (3 connections) — `tests/homesec/test_onvif_discovery.py`
- **test_discover_cameras_stops_discovery_when_search_fails()** (3 connections) — `tests/homesec/test_onvif_discovery.py`
- **_build_onvif_type_sets()** (3 connections) — `src/homesec/onvif/discovery.py`
- **_search_services()** (3 connections) — `src/homesec/onvif/discovery.py`
- **_suppress_wsdiscovery_interface_warnings()** (3 connections) — `src/homesec/onvif/discovery.py`
- **test_parse_discovery_services_tolerates_missing_methods_and_invalid_xaddr()** (2 connections) — `tests/homesec/test_onvif_discovery.py`
- **test_search_services_falls_back_to_legacy_wsdiscovery_signatures()** (2 connections) — `tests/homesec/test_onvif_discovery.py`
- **_as_iterable()** (2 connections) — `src/homesec/onvif/discovery.py`
- **_extract_ip()** (2 connections) — `src/homesec/onvif/discovery.py`
- **_safe_call()** (2 connections) — `src/homesec/onvif/discovery.py`
- **Tests for ONVIF WS-Discovery helpers.** (1 connections) — `tests/homesec/test_onvif_discovery.py`
- **discover_cameras should run every attempt round, not stop early.** (1 connections) — `tests/homesec/test_onvif_discovery.py`
- **discover_cameras should validate attempt count for deterministic behavior.** (1 connections) — `tests/homesec/test_onvif_discovery.py`
- **discover_cameras should always stop discovery even when search raises.** (1 connections) — `tests/homesec/test_onvif_discovery.py`
- **_search_services should retry with older WSDiscovery call signatures.** (1 connections) — `tests/homesec/test_onvif_discovery.py`
- **discover_cameras should tolerate stop() failures and emit cleanup warning.** (1 connections) — `tests/homesec/test_onvif_discovery.py`
- **_parse_discovery_services should safely ignore malformed WS-Discovery payloads.** (1 connections) — `tests/homesec/test_onvif_discovery.py`
- **discover_cameras should probe Device and NVT types separately and merge results.** (1 connections) — `tests/homesec/test_onvif_discovery.py`
- *... and 4 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/onvif/discovery.py`
- `tests/homesec/test_onvif_discovery.py`

## Audit Trail

- EXTRACTED: 70 (84%)
- INFERRED: 13 (16%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*