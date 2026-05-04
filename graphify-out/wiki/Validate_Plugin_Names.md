# Validate Plugin Names

> 24 nodes · cohesion 0.11

## Key Concepts

- **discover_all_plugins()** (12 connections) — `src/homesec/plugins/__init__.py`
- **validate_config()** (11 connections) — `src/homesec/config/validation.py`
- **validate_plugin_names()** (8 connections) — `src/homesec/config/validation.py`
- **.validate()** (7 connections) — `src/homesec/cli.py`
- **validate_camera_references()** (6 connections) — `src/homesec/config/validation.py`
- **validate_plugin_configs()** (6 connections) — `src/homesec/config/validation.py`
- **validation.py** (6 connections) — `src/homesec/config/validation.py`
- **validate_preview_camera_names()** (5 connections) — `src/homesec/config/validation.py`
- **test_validate_config_rejects_unknown_camera_override()** (5 connections) — `tests/homesec/test_config.py`
- **test_validate_config_rejects_unknown_plugin_backend()** (5 connections) — `tests/homesec/test_config.py`
- **test_validate_plugin_names_valid()** (5 connections) — `tests/homesec/test_config.py`
- **__init__.py** (2 connections) — `src/homesec/plugins/__init__.py`
- **Custom configuration validation helpers.** (1 connections) — `src/homesec/config/validation.py`
- **Validate that per-camera config keys reference valid camera names.      Args:** (1 connections) — `src/homesec/config/validation.py`
- **Validate plugin configs against registered plugin config models.** (1 connections) — `src/homesec/config/validation.py`
- **Validate config boundaries and plugin configs.** (1 connections) — `src/homesec/config/validation.py`
- **Validate preview-enabled configs do not alias camera artifact paths.** (1 connections) — `src/homesec/config/validation.py`
- **Validate that plugin names are recognized.      Args:         config: Config ins** (1 connections) — `src/homesec/config/validation.py`
- **Validate config file without running.          Args:             config: Path to** (1 connections) — `src/homesec/cli.py`
- **Validate_config should reject overrides for unknown cameras.** (1 connections) — `tests/homesec/test_config.py`
- **validate_config should surface unknown plugin backends.** (1 connections) — `tests/homesec/test_config.py`
- **Test validation passes when plugin names are valid.** (1 connections) — `tests/homesec/test_config.py`
- **Unified plugin discovery for all plugin types.** (1 connections) — `src/homesec/plugins/__init__.py`
- **Discover and register all plugins (built-in and external).      Built-in plugins** (1 connections) — `src/homesec/plugins/__init__.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/cli.py`
- `src/homesec/config/validation.py`
- `src/homesec/plugins/__init__.py`
- `tests/homesec/test_config.py`

## Audit Trail

- EXTRACTED: 49 (54%)
- INFERRED: 41 (46%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*