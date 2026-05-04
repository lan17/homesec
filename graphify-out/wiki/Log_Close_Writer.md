# Log Close Writer

> 20 nodes · cohesion 0.12

## Key Concepts

- **PostgresConfig** (12 connections) — `src/homesec/telemetry/postgres_settings.py`
- **_FakeThread** (7 connections) — `tests/homesec/test_db_log_handler.py`
- **test_db_log_handler.py** (6 connections) — `tests/homesec/test_db_log_handler.py`
- **test_db_log_handler_close_joins_writer_thread_when_started()** (5 connections) — `tests/homesec/test_db_log_handler.py`
- **test_db_log_handler_close_skips_join_on_writer_thread_itself()** (4 connections) — `tests/homesec/test_db_log_handler.py`
- **postgres_settings.py** (4 connections) — `src/homesec/telemetry/postgres_settings.py`
- **test_record_to_payload_excludes_formatter_created_standard_fields()** (3 connections) — `tests/homesec/test_db_log_handler.py`
- **test_record_to_payload_treats_event_type_as_canonical_event_kind()** (3 connections) — `tests/homesec/test_db_log_handler.py`
- **BaseSettings** (1 connections)
- **.__init__()** (1 connections) — `tests/homesec/test_db_log_handler.py`
- **.is_alive()** (1 connections) — `tests/homesec/test_db_log_handler.py`
- **.join()** (1 connections) — `tests/homesec/test_db_log_handler.py`
- **Tests for DB log handler lifecycle behavior.** (1 connections) — `tests/homesec/test_db_log_handler.py`
- **Close should not attempt to join when called from writer thread context.** (1 connections) — `tests/homesec/test_db_log_handler.py`
- **DB payload fields should only contain caller custom extras.** (1 connections) — `tests/homesec/test_db_log_handler.py`
- **DB payload should classify event_type records as event telemetry.** (1 connections) — `tests/homesec/test_db_log_handler.py`
- **Close should wait briefly for writer thread to flush before exit.** (1 connections) — `tests/homesec/test_db_log_handler.py`
- **enabled()** (1 connections) — `src/homesec/telemetry/postgres_settings.py`
- **_normalize_level()** (1 connections) — `src/homesec/telemetry/postgres_settings.py`
- **sync_dsn()** (1 connections) — `src/homesec/telemetry/postgres_settings.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/telemetry/postgres_settings.py`
- `tests/homesec/test_db_log_handler.py`

## Audit Trail

- EXTRACTED: 38 (68%)
- INFERRED: 18 (32%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*