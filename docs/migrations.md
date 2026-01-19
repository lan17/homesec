HomeSec Dual-Backend Migrations (PostgreSQL + SQLite)
=====================================================

This document explains how to generate, edit, and validate Alembic migrations
that must work for both PostgreSQL and SQLite.


Goals
-----
- Keep a single Alembic history that is valid on both backends.
- Autogenerate using PostgreSQL as the canonical source of truth.
- Manually adjust the migration for SQLite compatibility where needed.
- Validate with the migration smoke test and normal test suite.


Prerequisites
-------------
- PostgreSQL running locally (for autogenerate and validation).
- `DB_DSN` set for the backend you are targeting.
- `alembic/env.py` already normalizes DSNs and includes `src/` in `sys.path`.

Examples:

```
export DB_DSN=postgresql://homesec:homesec@localhost:5432/homesec
export DB_DSN=sqlite+aiosqlite:///tmp/homesec.db
```


Generate a New Migration (PostgreSQL First)
-------------------------------------------
1) Point Alembic at PostgreSQL (canonical schema).

```
export DB_DSN=postgresql://homesec:homesec@localhost:5432/homesec
```

2) Generate the migration.

```
make db-migration m="add foo columns"
```

3) Open the new file in `alembic/versions/` and review it carefully.


Edit the Migration for SQLite Compatibility
-------------------------------------------
### 1) Use database-agnostic types
- Prefer `JSONType` for JSON columns.
- Avoid `postgresql.JSONB` in migrations.

Example:

```
from homesec.db import JSONType

sa.Column("data", JSONType(), nullable=False)
```

### 2) Use portable defaults
- Prefer `sa.func.now()` over `sa.text("now()")`.

Example:

```
sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now())
```

### 3) Use dialect-aware DDL for indexes and PG-only features
Some indexes or expressions are PostgreSQL-only. Use a dialect branch:

```
from alembic import context, op

def _dialect_name() -> str:
    return context.get_context().dialect.name

def _is_postgres() -> bool:
    return _dialect_name() == "postgresql"
```

Example: JSON path indexes

```
if _is_postgres():
    op.create_index(
        "idx_clip_states_status",
        "clip_states",
        [sa.literal_column("jsonb_extract_path_text(data, 'status')")],
        unique=False,
    )
else:
    op.create_index(
        "idx_clip_states_status",
        "clip_states",
        [sa.literal_column("json_extract(data, '$.status')")],
        unique=False,
    )
```

SQLite relies on the JSON1 extension (enabled in standard Python builds).
If a custom SQLite build lacks JSON1, these indexes will fail.

### 4) Autoincrement and identity columns
SQLite only auto-increments for `INTEGER PRIMARY KEY`. If a model uses
`Integer` + `Identity()` for cross-backend compatibility, make sure the
migration matches that.

### 5) Use batch mode for SQLite ALTER TABLE
`alembic/env.py` enables `render_as_batch=True`, but for table alters
you should still use `op.batch_alter_table()` in migrations.


Run Migrations on Both Backends
-------------------------------
Run the migration on PostgreSQL:

```
export DB_DSN=postgresql://homesec:homesec@localhost:5432/homesec
make db-migrate
```

Run the migration on SQLite (file-based is recommended, not `:memory:`):

```
export DB_DSN=sqlite+aiosqlite:///tmp/homesec.db
make db-migrate
```


Validate Against Models and Basic Queries
-----------------------------------------
The migration smoke test runs on both backends and verifies:
- Alembic upgrade works.
- Schema matches SQLAlchemy metadata.
- A basic state/event roundtrip succeeds.

Run it via the normal test suite:

```
make test
```

If PostgreSQL is unavailable locally:

```
SKIP_POSTGRES_TESTS=1 make test-sqlite
```


Checklist for Every Migration
-----------------------------
- [ ] Generate with PostgreSQL (`make db-migration`).
- [ ] Replace PG-only types with `JSONType` and portable defaults.
- [ ] Add dialect branches for PG-only indexes/functions.
- [ ] Use `Integer` + `Identity()` when models require it.
- [ ] Run `make db-migrate` with PostgreSQL and SQLite DSNs.
- [ ] Run `make test` (or at least `make test-sqlite`).
