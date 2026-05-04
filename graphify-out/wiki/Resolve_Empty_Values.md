# Resolve Empty Values

> 6 nodes · cohesion 0.33

## Key Concepts

- **resolve_test_db_schema()** (7 connections) — `src/homesec/postgres_support.py`
- **test_resolve_test_db_schema_rejects_invalid_values()** (3 connections) — `tests/homesec/test_postgres_support.py`
- **test_resolve_test_db_schema_returns_none_for_empty_string()** (3 connections) — `tests/homesec/test_postgres_support.py`
- **Return the configured test schema, if any.** (1 connections) — `src/homesec/postgres_support.py`
- **resolve_test_db_schema should treat an empty env var as unset.** (1 connections) — `tests/homesec/test_postgres_support.py`
- **resolve_test_db_schema should validate non-empty env values.** (1 connections) — `tests/homesec/test_postgres_support.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/postgres_support.py`
- `tests/homesec/test_postgres_support.py`

## Audit Trail

- EXTRACTED: 10 (62%)
- INFERRED: 6 (38%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*