# Postgresconfig Asyncpostgresjsonloghandler Paylo

> 19 nodes · cohesion 0.12

## Key Concepts

- **PostgresConfig** (5 connections) — `src/homesec/telemetry/postgres_settings.py`
- **_DbRow** (3 connections) — `src/homesec/telemetry/db_log_handler.py`
- **AsyncPostgresJsonLogHandler.emit** (3 connections) — `src/homesec/telemetry/db_log_handler.py`
- **_record_to_payload** (3 connections) — `src/homesec/telemetry/db_log_handler.py`
- **AsyncPostgresJsonLogHandler._run_worker** (3 connections) — `src/homesec/telemetry/db_log_handler.py`
- **AsyncPostgresJsonLogHandler** (2 connections) — `src/homesec/telemetry/db_log_handler.py`
- **db_metadata** (2 connections) — `src/homesec/telemetry/db_log_handler.py`
- **logs** (2 connections) — `src/homesec/telemetry/db_log_handler.py`
- **PostgresConfig** (2 connections) — `src/homesec/telemetry/db_log_handler.py`
- **AsyncPostgresJsonLogHandler.start** (2 connections) — `src/homesec/telemetry/db_log_handler.py`
- **logs** (2 connections) — `src/homesec/telemetry/db/log_table.py`
- **metadata** (2 connections) — `src/homesec/telemetry/db/log_table.py`
- **PostgresConfig.enabled** (2 connections) — `src/homesec/telemetry/postgres_settings.py`
- **PostgresConfig.sync_dsn** (2 connections) — `src/homesec/telemetry/postgres_settings.py`
- **AsyncPostgresJsonLogHandler._drain_batch** (1 connections) — `src/homesec/telemetry/db_log_handler.py`
- **_PROMOTED_PAYLOAD_ATTRS** (1 connections) — `src/homesec/telemetry/db_log_handler.py`
- **_utc_iso** (1 connections) — `src/homesec/telemetry/db_log_handler.py`
- **PostgresConfig._normalize_level** (1 connections) — `src/homesec/telemetry/postgres_settings.py`
- **_REPO_DOTENV** (1 connections) — `src/homesec/telemetry/postgres_settings.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/telemetry/db/log_table.py`
- `src/homesec/telemetry/db_log_handler.py`
- `src/homesec/telemetry/postgres_settings.py`

## Audit Trail

- EXTRACTED: 34 (85%)
- INFERRED: 2 (5%)
- AMBIGUOUS: 4 (10%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*