# Setup Probe Probes

> 34 nodes · cohesion 0.08

## Key Concepts

- **SetupProbeRegistry** (11 connections) — `src/homesec/services/setup_probes.py`
- **setup_probes.py** (8 connections) — `src/homesec/services/setup_probes.py`
- **get_setup_probe()** (5 connections) — `src/homesec/services/setup_probes.py`
- **get_setup_probe_backends()** (5 connections) — `src/homesec/services/setup_probes.py`
- **get_setup_probe_timeout()** (5 connections) — `src/homesec/services/setup_probes.py`
- **load_builtin_setup_probes()** (5 connections) — `src/homesec/services/setup_probes.py`
- **test_setup_probes.py** (5 connections) — `tests/homesec/test_setup_probes.py`
- **.get()** (4 connections) — `src/homesec/services/setup_probes.py`
- **.get_timeout()** (4 connections) — `src/homesec/services/setup_probes.py`
- **test_setup_probe_registry_registers_and_looks_up_probe()** (3 connections) — `tests/homesec/test_setup_probes.py`
- **test_setup_probe_registry_rejects_duplicate_registration()** (3 connections) — `tests/homesec/test_setup_probes.py`
- **test_setup_probe_registry_tracks_custom_timeout_budget()** (3 connections) — `tests/homesec/test_setup_probes.py`
- **setup_probe()** (3 connections) — `src/homesec/services/setup_probes.py`
- **.get_backends()** (3 connections) — `src/homesec/services/setup_probes.py`
- **.register()** (3 connections) — `src/homesec/services/setup_probes.py`
- **test_load_builtin_setup_probes_registers_backend_adjacent_modules()** (2 connections) — `tests/homesec/test_setup_probes.py`
- **_RegisteredSetupProbe** (2 connections) — `src/homesec/services/setup_probes.py`
- **Tests for setup-only probe registry helpers.** (1 connections) — `tests/homesec/test_setup_probes.py`
- **Registry should register probes by target/backend and normalize lookup keys.** (1 connections) — `tests/homesec/test_setup_probes.py`
- **Registry should reject duplicate target/backend registrations.** (1 connections) — `tests/homesec/test_setup_probes.py`
- **Registry should preserve explicit timeout metadata for registered probes.** (1 connections) — `tests/homesec/test_setup_probes.py`
- **Builtin loader should expose built-in camera and storage probe backends.** (1 connections) — `tests/homesec/test_setup_probes.py`
- **Setup-only probe registry for onboarding test-connection flows.** (1 connections) — `src/homesec/services/setup_probes.py`
- **Decorator for registering a setup-only probe handler.** (1 connections) — `src/homesec/services/setup_probes.py`
- **Look up a setup-only probe handler.** (1 connections) — `src/homesec/services/setup_probes.py`
- *... and 9 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/services/setup_probes.py`
- `tests/homesec/test_setup_probes.py`

## Audit Trail

- EXTRACTED: 80 (88%)
- INFERRED: 11 (12%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*