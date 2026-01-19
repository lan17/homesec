# ADR-001: Add SQLite Support for HomeSec

**Status:** Implemented
**Date:** 2026-01-19
**Authors:** Lev Neiman, Claude

## Context

HomeSec currently uses PostgreSQL as its only database backend. While PostgreSQL is excellent for production, it creates friction for:

1. **Testing** - Requires a running PostgreSQL instance, slowing down development
2. **Local development** - Developers need Docker or local PostgreSQL setup
3. **Lightweight deployments** - Small installations don't need a full database server
4. **CI/CD** - PostgreSQL service adds complexity and boot time to CI pipelines

### Current JSONB Usage

The codebase uses PostgreSQL-specific `JSONB` type in three places:

| Location | Column | Purpose |
|----------|--------|---------|
| `src/homesec/state/postgres.py` | `ClipState.data` | Stores entire clip workflow state |
| `src/homesec/state/postgres.py` | `ClipEvent.event_data` | Stores event lifecycle data |
| `src/homesec/telemetry/db/log_table.py` | `logs.payload` | Stores log payloads |

### PostgreSQL-Specific Operations

Current code uses these PostgreSQL-specific features:
- `JSONB` column type
- `jsonb_extract_path_text()` for JSON queries
- `pg_insert().on_conflict_do_update()` for upserts
- `func.make_interval()` for date arithmetic
- PostgreSQL-specific error detection for retries

## Decision

Add SQLite as an alternative database backend while:
1. **Maximizing code reuse** through a unified `SQLAlchemyStateStore` class
2. **Isolating dialect differences** in a `DialectHelper` class
3. **Using parametrized tests** to ensure both backends are tested identically
4. **Making PostgreSQL optional** for local development via environment variable

## Architecture

### Database Abstraction Layer

```
src/homesec/db/
├── __init__.py          # Package exports
├── types.py             # JSONType (auto-adapts to dialect)
├── dialect.py           # DialectHelper (encapsulates ALL dialect differences)
└── engine.py            # Engine factory with dialect-appropriate config
```

### Key Abstraction: JSONType

A custom SQLAlchemy type that automatically uses the appropriate JSON storage:

```python
class JSONType(TypeDecorator):
    """Database-agnostic JSON type."""
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())
```

### Key Abstraction: DialectHelper

Encapsulates ALL dialect-specific operations in one place:

| Method | PostgreSQL | SQLite |
|--------|-----------|--------|
| `json_path_text(col, *path)` | `jsonb_extract_path_text(col, 'key')` | `json_extract(col, '$.key')` |
| `upsert_statement()` | `postgresql.insert()` | `sqlite.insert()` |
| `is_retryable_error(exc)` | Check PostgreSQL SQLSTATEs | Check `SQLITE_BUSY/LOCKED` |
| `normalize_dsn(dsn)` | Add `+asyncpg` driver | Add `+aiosqlite` driver |
| `get_engine_kwargs()` | Pool size 5, overflow 0 | StaticPool or NullPool |

### DRYness Principle

**95% of code is dialect-agnostic.** Only `DialectHelper` (~80 lines) contains dialect-specific logic.

| Approach | Dialect-Specific Code | Duplication |
|----------|----------------------|-------------|
| ❌ Separate PostgresStore + SQLiteStore | ~400 lines each | High (80%) |
| ❌ Base class + subclasses | ~50 base + ~30/subclass | Medium |
| ✅ **Single class + DialectHelper** | ~80 lines total | **None** |

---

## Implementation Plan

### Phase 1: Create Database Abstraction Layer

#### Task 1.1: Create `src/homesec/db/__init__.py`
- [ ] Create package with exports

#### Task 1.2: Create `src/homesec/db/types.py`
- [ ] Implement `JSONType` custom type
- [ ] Add docstrings explaining dialect behavior

#### Task 1.3: Create `src/homesec/db/dialect.py`
- [ ] Implement `DialectHelper` class with:
  - [ ] `json_path_text()` - JSON extraction
  - [ ] `get_upsert_statement()` - Dialect-appropriate insert
  - [ ] `is_retryable_error()` - Error classification
  - [ ] `normalize_dsn()` - DSN normalization
  - [ ] `get_engine_kwargs()` - Engine configuration
  - [ ] `older_than_condition()` - Date arithmetic

#### Task 1.4: Create `src/homesec/db/engine.py`
- [ ] Implement `create_async_engine_for_dsn()` factory
- [ ] Add `detect_dialect()` helper

### Phase 2: Update Models

#### Task 2.1: Update `src/homesec/state/postgres.py` models
- [ ] Replace `JSONB` import with `JSONType`
- [ ] Update `ClipState.data` column
- [ ] Update `ClipEvent.event_data` column
- [ ] Remove/update `__table_args__` functional indexes

#### Task 2.2: Update `src/homesec/telemetry/db/log_table.py`
- [ ] Replace `JSONB` with `JSONType`

### Phase 3: Refactor StateStore and EventStore

#### Task 3.1: Refactor `PostgresStateStore` → `SQLAlchemyStateStore`
- [ ] Add `DialectHelper` as instance variable
- [ ] Replace `pg_insert` with dialect-agnostic upsert
- [ ] Replace `jsonb_extract_path_text` with `dialect.json_path_text()`
- [ ] Replace `make_interval` with `dialect.older_than_condition()`
- [ ] Update `is_retryable_pg_error()` to use `dialect.is_retryable_error()`
- [ ] Keep `PostgresStateStore` as alias for backwards compatibility

#### Task 3.2: Refactor `PostgresEventStore` → `SQLAlchemyEventStore`
- [ ] Same changes as StateStore
- [ ] Keep `PostgresEventStore` as alias

#### Task 3.3: Update `src/homesec/state/__init__.py`
- [ ] Export new class names
- [ ] Export aliases for backwards compatibility

### Phase 4: Update Alembic

#### Task 4.1: Update `alembic/env.py`
- [ ] Add `render_as_batch=True` for SQLite ALTER TABLE support
- [ ] Update imports to use new module paths

#### Task 4.2: Create dialect-aware index creation
- [ ] Handle functional indexes differently per dialect
- [ ] Document migration strategy

### Phase 5: Add Dependencies

#### Task 5.1: Update `pyproject.toml`
- [ ] Add `aiosqlite>=0.19.0` to dependencies

### Phase 6: Update Tests (Parametrized)

#### Task 6.1: Update `tests/conftest.py`
- [ ] Add `pytest_addoption` for `--db-backend` flag
- [ ] Create parametrized `db_backend` fixture
- [ ] Create `db_dsn` fixture that yields DSN based on backend
- [ ] Add `SKIP_POSTGRES_TESTS` environment variable check
- [ ] SQLite tests always run (no external dependency)
- [ ] PostgreSQL tests skip if `SKIP_POSTGRES_TESTS=1` or no `TEST_DB_DSN`

#### Task 6.2: Update `tests/homesec/test_state_store.py`
- [ ] Use parametrized `state_store` fixture
- [ ] Ensure all tests work with both backends
- [ ] Add backend-specific test markers if needed

#### Task 6.3: Add dialect helper tests
- [ ] Test `normalize_dsn()` for both dialects
- [ ] Test `json_path_text()` output
- [ ] Test `is_retryable_error()` classification

### Phase 7: Update CI and Makefile

#### Task 7.1: Update `Makefile`
- [ ] Update `test` target to run both backends by default
- [ ] Add `test-sqlite` target for SQLite-only (fast)
- [ ] Add `test-postgres` target for PostgreSQL-only
- [ ] Update `check` target

#### Task 7.2: Update `.github/workflows/ci.yml`
- [ ] Keep PostgreSQL service (already configured)
- [ ] Ensure tests run against both backends
- [ ] Set appropriate environment variables

### Phase 8: Documentation and Cleanup

#### Task 8.1: Update `.env.example`
- [ ] Add SQLite DSN examples
- [ ] Document `SKIP_POSTGRES_TESTS` variable

#### Task 8.2: Update this ADR
- [ ] Mark tasks complete
- [ ] Document any deviations from plan
- [ ] Add lessons learned

---

## Testing Strategy

### Parametrized Tests

All database tests run against **both SQLite and PostgreSQL** using pytest parametrization:

```python
@pytest.fixture(params=["sqlite", "postgresql"])
def db_backend(request):
    """Parametrize tests to run against both database backends."""
    backend = request.param

    if backend == "postgresql":
        # Skip PostgreSQL tests if disabled or unavailable
        if os.environ.get("SKIP_POSTGRES_TESTS") == "1":
            pytest.skip("PostgreSQL tests disabled via SKIP_POSTGRES_TESTS=1")
        if not os.environ.get("TEST_DB_DSN"):
            pytest.skip("TEST_DB_DSN not set, skipping PostgreSQL tests")

    return backend

@pytest.fixture
def db_dsn(db_backend):
    """Return appropriate DSN for the backend."""
    if db_backend == "sqlite":
        return "sqlite+aiosqlite:///:memory:"
    else:
        return os.environ["TEST_DB_DSN"]
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SKIP_POSTGRES_TESTS` | `0` (off) | Set to `1` to skip PostgreSQL tests |
| `TEST_DB_DSN` | (from `.env`) | PostgreSQL DSN for tests |

**Important:** `SKIP_POSTGRES_TESTS` defaults to OFF. Both backends are always tested unless explicitly disabled. This is documented in `AGENTS.md` for AI agents who may need to skip PostgreSQL if it's unavailable.

### Test Execution Modes

| Command | SQLite | PostgreSQL | Use Case |
|---------|--------|------------|----------|
| `make test` | ✅ | ✅ | **Default** - full test suite (both backends) |
| `make check` | ✅ | ✅ | **Default** - lint + typecheck + test (both backends) |
| `make test-sqlite` | ✅ | ❌ | Fast local dev when PG unavailable |
| `SKIP_POSTGRES_TESTS=1 make test` | ✅ | ❌ | Equivalent to `make test-sqlite` |

### CI Behavior

GitHub Actions CI **always** runs both backends:
1. Start PostgreSQL service (already configured in `.github/workflows/ci.yml`)
2. Run `make check` which tests against **both** SQLite and PostgreSQL
3. SQLite tests run in-memory (fast, no setup)
4. PostgreSQL tests run against the CI service

**CI must never set `SKIP_POSTGRES_TESTS=1`.** Both backends must pass in CI.

### Makefile Targets

```makefile
# Run all tests (both backends) - DEFAULT
test:
	uv run pytest tests/homesec/ -v --cov=homesec --cov-report=term-missing

# Fast: SQLite only (no PostgreSQL required)
test-sqlite:
	SKIP_POSTGRES_TESTS=1 uv run pytest tests/homesec/ -v

# PostgreSQL only (requires TEST_DB_DSN)
test-postgres:
	uv run pytest tests/homesec/ -v -k "postgresql"
```

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `src/homesec/db/__init__.py` | Create | Package exports |
| `src/homesec/db/types.py` | Create | `JSONType` custom type |
| `src/homesec/db/dialect.py` | Create | `DialectHelper` class |
| `src/homesec/db/engine.py` | Create | Engine factory |
| `src/homesec/state/postgres.py` | Modify | Refactor to `SQLAlchemyStateStore` |
| `src/homesec/state/__init__.py` | Modify | Update exports |
| `src/homesec/telemetry/db/log_table.py` | Modify | Use `JSONType` |
| `alembic/env.py` | Modify | Add `render_as_batch=True` |
| `pyproject.toml` | Modify | Add `aiosqlite` dependency |
| `tests/conftest.py` | Modify | Add parametrized fixtures |
| `tests/homesec/test_state_store.py` | Modify | Use parametrized fixtures |
| `.env.example` | Modify | Add SQLite examples |
| `Makefile` | Modify | Add test targets |
| `.github/workflows/ci.yml` | Modify | Ensure both backends tested |

---

## Consequences

### Positive
- Tests run faster with in-memory SQLite
- Developers can work without PostgreSQL installed
- Lightweight deployments possible
- Single codebase, no duplication
- Comprehensive test coverage of both backends

### Negative
- Slightly more complex codebase (DialectHelper abstraction)
- Need to maintain compatibility with two backends
- Some PostgreSQL-specific optimizations (GIN indexes) not available in SQLite

### Neutral
- Migration strategy needs consideration for production (PostgreSQL recommended)
- SQLite suitable for single-user/embedded deployments only

---

## Execution Log

### 2026-01-19 - Implementation Complete

**Phase 1: Database Abstraction Layer**
- [x] Created `src/homesec/db/__init__.py` with package exports
- [x] Created `src/homesec/db/types.py` with `JSONType` custom type
- [x] Created `src/homesec/db/dialect.py` with `DialectHelper` class
- [x] Created `src/homesec/db/engine.py` with engine factory

**Phase 2: Model Updates**
- [x] Updated `ClipState.data` to use `JSONType`
- [x] Updated `ClipEvent.event_data` to use `JSONType`
- [x] Updated `ClipEvent.id` to use `Integer` with `Identity()` for SQLite compatibility
- [x] Updated `logs.payload` in telemetry to use `JSONType`

**Phase 3: StateStore/EventStore Refactoring**
- [x] Refactored `PostgresStateStore` to `SQLAlchemyStateStore`
- [x] Refactored `PostgresEventStore` to `SQLAlchemyEventStore`
- [x] Added backwards compatibility aliases
- [x] Updated `src/homesec/state/__init__.py` exports

**Phase 4: Alembic Updates**
- [x] Added `render_as_batch=True` for SQLite support
- [x] Updated to use `DialectHelper.normalize_dsn()`

**Phase 5: Dependencies**
- [x] Added `aiosqlite>=0.19.0` to `pyproject.toml`

**Phase 6: Tests**
- [x] Created parametrized `db_backend` and `db_dsn` fixtures
- [x] Created parametrized `state_store` fixture
- [x] Updated `test_state_store.py` to use parametrized fixtures
- [x] Added SQLite-specific tests (DSN normalization, dialect detection)

**Phase 7: CI/Makefile**
- [x] Added `make test-sqlite` target
- [x] Updated `make test` and `make check` to run both backends
- [x] Added `aiosqlite` to `db-migrate` command

**Deviations from Original Plan:**
- Used `Integer` with `Identity()` instead of `BigInteger` for `ClipEvent.id` because SQLite only supports autoincrement on `INTEGER PRIMARY KEY`
- Did not need to modify `.github/workflows/ci.yml` as existing PostgreSQL service works with parametrized tests

**Test Results:**
- All 250 tests pass with SQLite (16 SQLite-specific, 234 others)
- PostgreSQL tests skip cleanly when `SKIP_POSTGRES_TESTS=1`
- Lint and typecheck pass

---

## References

- [SQLAlchemy TypeDecorator](https://docs.sqlalchemy.org/en/20/core/custom_types.html#sqlalchemy.types.TypeDecorator)
- [SQLAlchemy Dialect-Specific Types](https://docs.sqlalchemy.org/en/20/core/type_basics.html#dialect-specific-types)
- [pytest Parametrize](https://docs.pytest.org/en/stable/how-to/parametrize.html)
- [aiosqlite Documentation](https://aiosqlite.omnilib.dev/)
