# Setup Connection Probe

> 44 nodes · cohesion 0.05

## Key Concepts

- **test_local_folder_camera_connection** (8 connections) — `src/homesec/sources/local_folder_setup_probe.py`
- **test_connection** (5 connections) — `src/homesec/api/routes/setup.py`
- **load_builtin_setup_probes** (4 connections) — `src/homesec/services/setup_probes.py`
- **_RegisteredSetupProbe** (4 connections) — `src/homesec/services/setup_probes.py`
- **SetupProbeFn** (4 connections) — `src/homesec/services/setup_probes.py`
- **_test_camera_connection** (4 connections) — `src/homesec/services/setup.py`
- **finalize_setup_endpoint** (3 connections) — `src/homesec/api/routes/setup.py`
- **get_setup_probe** (3 connections) — `src/homesec/services/setup.py`
- **build_test_connection_response** (3 connections) — `src/homesec/services/setup_probe_support.py`
- **get_setup_probe** (3 connections) — `src/homesec/services/setup_probes.py`
- **get_setup_probe_backends** (3 connections) — `src/homesec/services/setup_probes.py`
- **get_setup_probe_timeout** (3 connections) — `src/homesec/services/setup_probes.py`
- **SetupProbeRegistry.register** (3 connections) — `src/homesec/services/setup_probes.py`
- **build_test_connection_response** (2 connections) — `src/homesec/sources/local_folder_setup_probe.py`
- **format_validation_error** (2 connections) — `src/homesec/sources/local_folder_setup_probe.py`
- **setup_probe** (2 connections) — `src/homesec/sources/local_folder_setup_probe.py`
- **SETUP_TEST_CAMERA_NAME** (2 connections) — `src/homesec/sources/local_folder_setup_probe.py`
- **APIError** (2 connections) — `src/homesec/api/routes/setup.py`
- **build_test_connection_response** (2 connections) — `src/homesec/services/setup.py`
- **format_validation_error** (2 connections) — `src/homesec/services/setup_probe_support.py`
- **SetupProbeRegistry.get** (2 connections) — `src/homesec/services/setup_probes.py`
- **setup_probe** (2 connections) — `src/homesec/services/setup_probes.py`
- **SetupProbeRegistry** (2 connections) — `src/homesec/services/setup_probes.py`
- **test_setup_connection_endpoint** (2 connections) — `src/homesec/api/routes/setup.py`
- **_test_storage_connection** (2 connections) — `src/homesec/services/setup.py`
- *... and 19 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/api/routes/setup.py`
- `src/homesec/models/setup.py`
- `src/homesec/services/setup.py`
- `src/homesec/services/setup_probe_support.py`
- `src/homesec/services/setup_probes.py`
- `src/homesec/sources/local_folder_setup_probe.py`

## Audit Trail

- EXTRACTED: 88 (94%)
- INFERRED: 4 (4%)
- AMBIGUOUS: 2 (2%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*