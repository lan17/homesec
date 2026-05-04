# Build Engine Kwargs

> 16 nodes · cohesion 0.16

## Key Concepts

- **test_postgres_support.py** (10 connections) — `tests/homesec/test_postgres_support.py`
- **build_async_engine_kwargs()** (8 connections) — `src/homesec/postgres_support.py`
- **validate_schema_name()** (7 connections) — `src/homesec/postgres_support.py`
- **test_build_async_engine_kwargs_ignores_test_env_without_explicit_schema()** (3 connections) — `tests/homesec/test_postgres_support.py`
- **test_build_async_engine_kwargs_rejects_conflicting_search_path()** (3 connections) — `tests/homesec/test_postgres_support.py`
- **test_build_async_engine_kwargs_sets_search_path_for_schema()** (3 connections) — `tests/homesec/test_postgres_support.py`
- **test_validate_schema_name_accepts_lowercase_max_length()** (3 connections) — `tests/homesec/test_postgres_support.py`
- **test_validate_schema_name_rejects_invalid_values()** (3 connections) — `tests/homesec/test_postgres_support.py`
- **Validate a Postgres schema identifier used for test isolation.** (1 connections) — `src/homesec/postgres_support.py`
- **Build SQLAlchemy engine kwargs with an optional explicit schema search path.** (1 connections) — `src/homesec/postgres_support.py`
- **Tests for Postgres helper utilities.** (1 connections) — `tests/homesec/test_postgres_support.py`
- **build_async_engine_kwargs should set search_path for the requested schema.** (1 connections) — `tests/homesec/test_postgres_support.py`
- **build_async_engine_kwargs should reject conflicting caller-provided search_path** (1 connections) — `tests/homesec/test_postgres_support.py`
- **validate_schema_name should accept valid lowercase identifiers up to 63 chars.** (1 connections) — `tests/homesec/test_postgres_support.py`
- **validate_schema_name should reject invalid test schema identifiers.** (1 connections) — `tests/homesec/test_postgres_support.py`
- **build_async_engine_kwargs should ignore the test env unless schema is passed exp** (1 connections) — `tests/homesec/test_postgres_support.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/postgres_support.py`
- `tests/homesec/test_postgres_support.py`

## Audit Trail

- EXTRACTED: 37 (77%)
- INFERRED: 11 (23%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*