# Logging Recording Reset

> 16 nodes · cohesion 0.12

## Key Concepts

- **test_logging_setup.py** (7 connections) — `tests/homesec/test_logging_setup.py`
- **TestLoggingInjection** (5 connections) — `tests/homesec/test_logging_setup.py`
- **TestDbHandlerFilter** (4 connections) — `tests/homesec/test_logging_setup.py`
- **.test_injects_recording_id()** (4 connections) — `tests/homesec/test_logging_setup.py`
- **set_recording_id()** (3 connections) — `src/homesec/logging_setup.py`
- **reset_logging_root()** (3 connections) — `tests/homesec/test_logging_setup.py`
- **.test_db_handler_filters_events_and_levels()** (3 connections) — `tests/homesec/test_logging_setup.py`
- **reset_logging_state()** (2 connections) — `tests/homesec/test_logging_setup.py`
- **Set the `recording_id` value injected into log records.** (1 connections) — `src/homesec/logging_setup.py`
- **Tests for logging setup module.** (1 connections) — `tests/homesec/test_logging_setup.py`
- **Tests for DB handler filtering via configure_logging.** (1 connections) — `tests/homesec/test_logging_setup.py`
- **DB handler allows event logs and blocks low-level non-events.** (1 connections) — `tests/homesec/test_logging_setup.py`
- **Reset global logging state before each test.** (1 connections) — `tests/homesec/test_logging_setup.py`
- **Restore root logger handlers/levels after each test.** (1 connections) — `tests/homesec/test_logging_setup.py`
- **Tests for camera/recording injection via configure_logging.** (1 connections) — `tests/homesec/test_logging_setup.py`
- **Recording ID is injected into log records.** (1 connections) — `tests/homesec/test_logging_setup.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/logging_setup.py`
- `tests/homesec/test_logging_setup.py`

## Audit Trail

- EXTRACTED: 32 (82%)
- INFERRED: 7 (18%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*