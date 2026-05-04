# Async Postgres Normalize

> 14 nodes · cohesion 0.20

## Key Concepts

- **postgres_support.py** (10 connections) — `src/homesec/postgres_support.py`
- **drop_schema_cascade()** (7 connections) — `src/homesec/postgres_support.py`
- **normalize_async_dsn()** (7 connections) — `src/homesec/postgres_support.py`
- **create_schema_if_missing()** (6 connections) — `src/homesec/postgres_support.py`
- **schema_ddl_identifier()** (6 connections) — `src/homesec/postgres_support.py`
- **create_scoped_async_engine()** (5 connections) — `src/homesec/postgres_support.py`
- **test_normalize_async_dsn()** (3 connections) — `tests/homesec/test_state_store.py`
- **Shared Postgres helpers for async engines and test schema isolation.  The separa** (1 connections) — `src/homesec/postgres_support.py`
- **Drop an isolated test schema and everything inside it.** (1 connections) — `src/homesec/postgres_support.py`
- **Normalize Postgres DSNs to the asyncpg SQLAlchemy dialect.** (1 connections) — `src/homesec/postgres_support.py`
- **Create an async engine scoped to the explicit schema when present.** (1 connections) — `src/homesec/postgres_support.py`
- **Return a quoted schema identifier for DDL statements.** (1 connections) — `src/homesec/postgres_support.py`
- **Create a schema for an isolated test run if it does not exist.** (1 connections) — `src/homesec/postgres_support.py`
- **Test DSN normalization adds asyncpg driver.** (1 connections) — `tests/homesec/test_state_store.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/postgres_support.py`
- `tests/homesec/test_state_store.py`

## Audit Trail

- EXTRACTED: 41 (80%)
- INFERRED: 10 (20%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*