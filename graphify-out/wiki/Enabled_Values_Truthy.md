# Enabled Values Truthy

> 6 nodes · cohesion 0.33

## Key Concepts

- **is_test_db_schema_enabled()** (5 connections) — `src/homesec/postgres_support.py`
- **test_test_db_schema_enabled_accepts_truthy_values()** (3 connections) — `tests/homesec/test_postgres_support.py`
- **test_test_db_schema_enabled_rejects_falsy_values()** (3 connections) — `tests/homesec/test_postgres_support.py`
- **Return whether test-only schema scoping is explicitly enabled.** (1 connections) — `src/homesec/postgres_support.py`
- **is_test_db_schema_enabled should stay off for missing or falsy values.** (1 connections) — `tests/homesec/test_postgres_support.py`
- **is_test_db_schema_enabled should accept the documented truthy enable values.** (1 connections) — `tests/homesec/test_postgres_support.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/postgres_support.py`
- `tests/homesec/test_postgres_support.py`

## Audit Trail

- EXTRACTED: 9 (64%)
- INFERRED: 5 (36%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*