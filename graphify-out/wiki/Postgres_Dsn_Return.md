# Postgres Dsn Return

> 9 nodes · cohesion 0.22

## Key Concepts

- **isolated_postgres_schema()** (7 connections) — `tests/homesec/conftest.py`
- **default_test_dsn()** (5 connections) — `tests/homesec/postgres_test_support.py`
- **postgres_dsn()** (3 connections) — `tests/homesec/conftest.py`
- **postgres_test_support.py** (3 connections) — `tests/homesec/postgres_test_support.py`
- **_generate_test_schema_name()** (2 connections) — `tests/homesec/conftest.py`
- **Return test Postgres DSN (requires local DB running).** (1 connections) — `tests/homesec/conftest.py`
- **Provision a per-run schema so parallel test runs can share one Postgres instance** (1 connections) — `tests/homesec/conftest.py`
- **Shared Postgres helpers for tests.** (1 connections) — `tests/homesec/postgres_test_support.py`
- **Return the Docker-backed Postgres DSN used by server-facing tests.** (1 connections) — `tests/homesec/postgres_test_support.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/conftest.py`
- `tests/homesec/postgres_test_support.py`

## Audit Trail

- EXTRACTED: 16 (67%)
- INFERRED: 8 (33%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*