# Connection Setup Probe

> 18 nodes · cohesion 0.16

## Key Concepts

- **build_test_connection_response()** (11 connections) — `src/homesec/services/setup_probe_support.py`
- **_test_plugin_ping_connection()** (9 connections) — `src/homesec/services/setup.py`
- **get_plugin_names()** (8 connections) — `src/homesec/plugins/registry.py`
- **_run_registered_setup_probe()** (8 connections) — `src/homesec/services/setup.py`
- **test_connection()** (7 connections) — `src/homesec/services/setup.py`
- **_test_camera_connection()** (6 connections) — `src/homesec/services/setup.py`
- **_camera_backend_names()** (5 connections) — `src/homesec/services/setup.py`
- **_ensure_known_backend()** (4 connections) — `src/homesec/services/setup.py`
- **_test_analyzer_connection()** (4 connections) — `src/homesec/services/setup.py`
- **_test_notifier_connection()** (4 connections) — `src/homesec/services/setup.py`
- **_test_storage_connection()** (4 connections) — `src/homesec/services/setup.py`
- **setup_probe_support.py** (3 connections) — `src/homesec/services/setup_probe_support.py`
- **format_validation_error()** (2 connections) — `src/homesec/services/setup_probe_support.py`
- **Get list of registered plugin names for a given type.** (1 connections) — `src/homesec/plugins/registry.py`
- **Shared helpers for setup-only probe implementations.** (1 connections) — `src/homesec/services/setup_probe_support.py`
- **Return a concise single-error validation message for setup UX.** (1 connections) — `src/homesec/services/setup_probe_support.py`
- **Build a setup test-connection response with optional latency metadata.** (1 connections) — `src/homesec/services/setup_probe_support.py`
- **Validate and probe connectivity for setup-configured integrations.** (1 connections) — `src/homesec/services/setup.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/plugins/registry.py`
- `src/homesec/services/setup.py`
- `src/homesec/services/setup_probe_support.py`

## Audit Trail

- EXTRACTED: 55 (69%)
- INFERRED: 25 (31%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*