# Worker Log Asyncpostgresjsonloghandler

> 14 nodes · cohesion 0.24

## Key Concepts

- **AsyncPostgresJsonLogHandler** (23 connections) — `src/homesec/telemetry/db_log_handler.py`
- **_record_to_payload()** (7 connections) — `src/homesec/telemetry/db_log_handler.py`
- **._run_worker()** (5 connections) — `src/homesec/telemetry/db_log_handler.py`
- **db_log_handler.py** (4 connections) — `src/homesec/telemetry/db_log_handler.py`
- **.emit()** (4 connections) — `src/homesec/telemetry/db_log_handler.py`
- **._ensure_schema()** (3 connections) — `src/homesec/telemetry/db_log_handler.py`
- **._flush()** (3 connections) — `src/homesec/telemetry/db_log_handler.py`
- **_DbRow** (3 connections) — `src/homesec/telemetry/db_log_handler.py`
- **.close()** (2 connections) — `src/homesec/telemetry/db_log_handler.py`
- **._drain_batch()** (2 connections) — `src/homesec/telemetry/db_log_handler.py`
- **.start()** (2 connections) — `src/homesec/telemetry/db_log_handler.py`
- **_utc_iso()** (2 connections) — `src/homesec/telemetry/db_log_handler.py`
- **.__init__()** (1 connections) — `src/homesec/telemetry/db_log_handler.py`
- **Best-effort DB log handler using async SQLAlchemy in a worker thread.      - `em** (1 connections) — `src/homesec/telemetry/db_log_handler.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/telemetry/db_log_handler.py`

## Audit Trail

- EXTRACTED: 44 (71%)
- INFERRED: 18 (29%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*