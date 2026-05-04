# Runtime Build Persistence

> 30 nodes · cohesion 0.11

## Key Concepts

- **_RecordingStorage** (16 connections) — `tests/homesec/test_runtime_bootstrap.py`
- **build_runtime_persistence_stack()** (15 connections) — `src/homesec/runtime/bootstrap.py`
- **_make_config()** (13 connections) — `tests/homesec/test_runtime_bootstrap.py`
- **bootstrap.py** (10 connections) — `src/homesec/runtime/bootstrap.py`
- **test_runtime_bootstrap.py** (9 connections) — `tests/homesec/test_runtime_bootstrap.py`
- **._build_runtime_persistence_stack()** (4 connections) — `src/homesec/app.py`
- **test_build_runtime_persistence_stack_shuts_down_partial_state_when_event_store_fails()** (4 connections) — `tests/homesec/test_runtime_bootstrap.py`
- **test_build_runtime_persistence_stack_shuts_down_storage_when_state_init_fails()** (4 connections) — `tests/homesec/test_runtime_bootstrap.py`
- **test_build_runtime_persistence_stack_uses_noop_events_when_state_init_returns_false()** (4 connections) — `tests/homesec/test_runtime_bootstrap.py`
- **_create_repository()** (4 connections) — `src/homesec/runtime/bootstrap.py`
- **_create_state_store()** (4 connections) — `src/homesec/runtime/bootstrap.py`
- **test_build_runtime_persistence_stack_preserves_primary_failure_when_cleanup_raises()** (3 connections) — `tests/homesec/test_runtime_bootstrap.py`
- **test_build_runtime_persistence_stack_requires_event_factory_with_custom_state()** (3 connections) — `tests/homesec/test_runtime_bootstrap.py`
- **_create_event_store()** (3 connections) — `src/homesec/runtime/bootstrap.py`
- **_create_storage_backend()** (3 connections) — `src/homesec/runtime/bootstrap.py`
- **.initialize()** (3 connections) — `src/homesec/runtime/bootstrap.py`
- **_safe_shutdown()** (3 connections) — `src/homesec/runtime/bootstrap.py`
- **Build shared storage and persistence components for the app runtime.** (1 connections) — `src/homesec/app.py`
- **_noop_event_store_factory()** (1 connections) — `tests/homesec/test_runtime_bootstrap.py`
- **Tests for shared runtime bootstrap helpers.** (1 connections) — `tests/homesec/test_runtime_bootstrap.py`
- **.__init__()** (1 connections) — `tests/homesec/test_runtime_bootstrap.py`
- **.shutdown()** (1 connections) — `tests/homesec/test_runtime_bootstrap.py`
- **Shared runtime bootstrap helpers for storage and persistence wiring.** (1 connections) — `src/homesec/runtime/bootstrap.py`
- **Create repository over the configured state and event stores.** (1 connections) — `src/homesec/runtime/bootstrap.py`
- **Build shared storage and persistence components for a runtime.** (1 connections) — `src/homesec/runtime/bootstrap.py`
- *... and 5 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/app.py`
- `src/homesec/runtime/bootstrap.py`
- `tests/homesec/test_runtime_bootstrap.py`

## Audit Trail

- EXTRACTED: 87 (74%)
- INFERRED: 31 (26%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*