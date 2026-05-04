# Clips Mark Deleted

> 33 nodes · cohesion 0.07

## Key Concepts

- **postgres.py** (13 connections) — `src/homesec/state/postgres.py`
- **.select()** (11 connections) — `tests/homesec/test_plugin_utils.py`
- **._hydrate_clip_state()** (8 connections) — `src/homesec/state/postgres.py`
- **.get_events()** (6 connections) — `src/homesec/state/postgres.py`
- **.get_many_with_created_at()** (5 connections) — `src/homesec/state/postgres.py`
- **.list_candidate_clips_for_cleanup()** (5 connections) — `src/homesec/state/postgres.py`
- **test_mark_clip_deleted_preserves_existing_jsonb_fields()** (4 connections) — `tests/homesec/test_state_store.py`
- **.append()** (4 connections) — `src/homesec/state/postgres.py`
- **_parse_jsonb_payload()** (4 connections) — `src/homesec/state/postgres.py`
- **.get_clip()** (4 connections) — `src/homesec/state/postgres.py`
- **is_retryable_pg_error()** (3 connections) — `src/homesec/state/postgres.py`
- **_parse_state_data()** (3 connections) — `src/homesec/state/postgres.py`
- **.count_alerts_since()** (3 connections) — `src/homesec/state/postgres.py`
- **.count_clips_since()** (3 connections) — `src/homesec/state/postgres.py`
- **.initialize()** (3 connections) — `src/homesec/state/postgres.py`
- **.mark_clip_deleted()** (3 connections) — `src/homesec/state/postgres.py`
- **.ping()** (3 connections) — `src/homesec/state/postgres.py`
- **_extract_sqlstate()** (2 connections) — `src/homesec/state/postgres.py`
- **.get()** (2 connections) — `src/homesec/state/postgres.py`
- **mark_clip_deleted should mutate status without clobbering unrelated JSONB fields** (1 connections) — `tests/homesec/test_state_store.py`
- **Postgres implementation of StateStore and EventStore.** (1 connections) — `src/homesec/state/postgres.py`
- **Initialize connection pool.          Note: Tables are created via alembic migrat** (1 connections) — `src/homesec/state/postgres.py`
- **Get clip state by ID.** (1 connections) — `src/homesec/state/postgres.py`
- **Retrieve state and created_at for a set of clip ids.** (1 connections) — `src/homesec/state/postgres.py`
- **Mark a clip as deleted.** (1 connections) — `src/homesec/state/postgres.py`
- *... and 8 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/state/postgres.py`
- `tests/homesec/test_plugin_utils.py`
- `tests/homesec/test_state_store.py`

## Audit Trail

- EXTRACTED: 82 (80%)
- INFERRED: 21 (20%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*