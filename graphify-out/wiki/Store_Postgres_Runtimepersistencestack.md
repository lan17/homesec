# Store Postgres Runtimepersistencestack

> 15 nodes · cohesion 0.14

## Key Concepts

- **build_runtime_persistence_stack** (8 connections) — `src/homesec/runtime/bootstrap.py`
- **_create_postgres_event_store** (3 connections) — `src/homesec/runtime/bootstrap.py`
- **RuntimePersistenceStack** (3 connections) — `src/homesec/runtime/bootstrap.py`
- **ClipRepository** (2 connections) — `src/homesec/runtime/bootstrap.py`
- **_create_event_store** (2 connections) — `src/homesec/runtime/bootstrap.py`
- **_create_repository** (2 connections) — `src/homesec/runtime/bootstrap.py`
- **_create_state_store** (2 connections) — `src/homesec/runtime/bootstrap.py`
- **_create_storage_backend** (2 connections) — `src/homesec/runtime/bootstrap.py`
- **RuntimePersistenceStack** (2 connections) — `src/homesec/runtime/worker.py`
- **create_event_store_for_postgres_state_store** (1 connections) — `src/homesec/runtime/bootstrap.py`
- **_InitializableStateStore** (1 connections) — `src/homesec/runtime/bootstrap.py`
- **load_storage_plugin** (1 connections) — `src/homesec/runtime/bootstrap.py`
- **NoopEventStore** (1 connections) — `src/homesec/runtime/bootstrap.py`
- **PostgresStateStore** (1 connections) — `src/homesec/runtime/bootstrap.py`
- **_safe_shutdown** (1 connections) — `src/homesec/runtime/bootstrap.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/runtime/bootstrap.py`
- `src/homesec/runtime/worker.py`

## Audit Trail

- EXTRACTED: 30 (94%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 2 (6%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*