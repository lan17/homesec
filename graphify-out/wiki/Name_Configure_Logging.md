# Name Configure Logging

> 14 nodes · cohesion 0.18

## Key Concepts

- **configure_logging()** (19 connections) — `src/homesec/logging_setup.py`
- **TestConfigureLogging** (6 connections) — `tests/homesec/test_logging_setup.py`
- **set_camera_name()** (4 connections) — `src/homesec/logging_setup.py`
- **.test_injects_camera_name()** (4 connections) — `tests/homesec/test_logging_setup.py`
- **.test_configures_console_handler()** (3 connections) — `tests/homesec/test_logging_setup.py`
- **.test_skips_db_handler_when_disabled()** (3 connections) — `tests/homesec/test_logging_setup.py`
- **.test_suppresses_third_party_loggers()** (3 connections) — `tests/homesec/test_logging_setup.py`
- **Set the `camera_name` value injected into log records.** (1 connections) — `src/homesec/logging_setup.py`
- **Configure root logging with a consistent format.      Format includes `camera_na** (1 connections) — `src/homesec/logging_setup.py`
- **Tests for configure_logging function.** (1 connections) — `tests/homesec/test_logging_setup.py`
- **Configures console handler on root logger.** (1 connections) — `tests/homesec/test_logging_setup.py`
- **Sets third-party loggers to WARNING level.** (1 connections) — `tests/homesec/test_logging_setup.py`
- **Doesn't add DB handler when DB_DSN not set.** (1 connections) — `tests/homesec/test_logging_setup.py`
- **Camera name is injected into log records.** (1 connections) — `tests/homesec/test_logging_setup.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/logging_setup.py`
- `tests/homesec/test_logging_setup.py`

## Audit Trail

- EXTRACTED: 29 (59%)
- INFERRED: 20 (41%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*