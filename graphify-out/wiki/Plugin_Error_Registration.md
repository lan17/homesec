# Plugin Error Registration

> 17 nodes · cohesion 0.13

## Key Concepts

- **test_plugin_registration.py** (9 connections) — `tests/homesec/test_plugin_registration.py`
- **test_register_and_load_plugin()** (5 connections) — `tests/homesec/test_plugin_registration.py`
- **plugin()** (5 connections) — `src/homesec/plugins/registry.py`
- **test_validation_error()** (4 connections) — `tests/homesec/test_plugin_registration.py`
- **DummyPlugin** (3 connections) — `tests/homesec/test_plugin_registration.py`
- **test_duplicate_registration_error()** (3 connections) — `tests/homesec/test_plugin_registration.py`
- **test_unknown_plugin_error()** (3 connections) — `tests/homesec/test_plugin_registration.py`
- **clean_registry()** (2 connections) — `tests/homesec/test_plugin_registration.py`
- **create()** (1 connections) — `tests/homesec/test_plugin_registration.py`
- **.__init__()** (1 connections) — `tests/homesec/test_plugin_registration.py`
- **Tests for plugin registration and discovery mechanisms.** (1 connections) — `tests/homesec/test_plugin_registration.py`
- **Ensure registry is clean for tests.** (1 connections) — `tests/homesec/test_plugin_registration.py`
- **Test basic registration and loading.** (1 connections) — `tests/homesec/test_plugin_registration.py`
- **Test config validation error.** (1 connections) — `tests/homesec/test_plugin_registration.py`
- **Test registering same name twice raises error.** (1 connections) — `tests/homesec/test_plugin_registration.py`
- **Test loading unknown plugin raises ValueError.** (1 connections) — `tests/homesec/test_plugin_registration.py`
- **Decorator to register a class as a plugin.      Args:         plugin_type: The c** (1 connections) — `src/homesec/plugins/registry.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/plugins/registry.py`
- `tests/homesec/test_plugin_registration.py`

## Audit Trail

- EXTRACTED: 32 (74%)
- INFERRED: 11 (26%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*