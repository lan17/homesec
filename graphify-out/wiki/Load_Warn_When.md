# Load Warn When

> 20 nodes · cohesion 0.10

## Key Concepts

- **load_config()** (18 connections) — `src/homesec/config/loader.py`
- **_warn_if_permissive_config_mode()** (3 connections) — `src/homesec/config/loader.py`
- **.get_config()** (3 connections) — `src/homesec/config/manager.py`
- **test_load_config_does_not_warn_when_permissions_are_restrictive()** (3 connections) — `tests/homesec/test_config.py`
- **test_load_config_empty_file()** (3 connections) — `tests/homesec/test_config.py`
- **test_load_config_file_not_found()** (3 connections) — `tests/homesec/test_config.py`
- **test_load_config_from_yaml_file()** (3 connections) — `tests/homesec/test_config.py`
- **test_load_config_invalid_yaml()** (3 connections) — `tests/homesec/test_config.py`
- **test_load_config_warns_when_permissions_are_too_open()** (3 connections) — `tests/homesec/test_config.py`
- **test_load_example_config()** (3 connections) — `tests/homesec/test_config.py`
- **Warn when config file mode exposes secrets to group/other users.** (1 connections) — `src/homesec/config/loader.py`
- **Load and validate configuration from YAML file.      Args:         path: Path to** (1 connections) — `src/homesec/config/loader.py`
- **Get the current configuration.** (1 connections) — `src/homesec/config/manager.py`
- **Test loading config from YAML file.** (1 connections) — `tests/homesec/test_config.py`
- **Loading config should warn when file permissions expose secrets.** (1 connections) — `tests/homesec/test_config.py`
- **Loading config should not warn when file permissions are already restrictive.** (1 connections) — `tests/homesec/test_config.py`
- **Test that missing file raises ConfigError.** (1 connections) — `tests/homesec/test_config.py`
- **Test that invalid YAML raises ConfigError.** (1 connections) — `tests/homesec/test_config.py`
- **Test that empty file raises ConfigError.** (1 connections) — `tests/homesec/test_config.py`
- **Test that the example config file loads successfully.** (1 connections) — `tests/homesec/test_config.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/config/loader.py`
- `src/homesec/config/manager.py`
- `tests/homesec/test_config.py`

## Audit Trail

- EXTRACTED: 34 (62%)
- INFERRED: 21 (38%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*