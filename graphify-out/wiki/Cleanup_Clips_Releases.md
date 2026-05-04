# Cleanup Clips Releases

> 48 nodes · cohesion 0.09

## Key Concepts

- **CleanupOptions** (23 connections) — `src/homesec/maintenance/cleanup_clips.py`
- **run_cleanup()** (23 connections) — `src/homesec/maintenance/cleanup_clips.py`
- **LocalStorageConfig** (20 connections) — `src/homesec/plugins/storage/local.py`
- **_TestFilter** (19 connections) — `tests/homesec/test_cleanup_clips.py`
- **_CleanupStateStore** (16 connections) — `tests/homesec/test_cleanup_clips.py`
- **_CleanupEventStore** (15 connections) — `tests/homesec/test_cleanup_clips.py`
- **_CleanupStorage** (15 connections) — `tests/homesec/test_cleanup_clips.py`
- **test_cleanup_deletes_empty_clips()** (15 connections) — `tests/homesec/test_cleanup_clips.py`
- **test_cleanup_marks_false_negatives()** (14 connections) — `tests/homesec/test_cleanup_clips.py`
- **_Counts** (13 connections) — `src/homesec/maintenance/cleanup_clips.py`
- **cleanup_clips.py** (11 connections) — `src/homesec/maintenance/cleanup_clips.py`
- **test_cleanup_clips.py** (11 connections) — `tests/homesec/test_cleanup_clips.py`
- **test_cleanup_attempts_all_shutdowns_and_preserves_primary_error()** (10 connections) — `tests/homesec/test_cleanup_clips.py`
- **test_cleanup_releases_persistence_when_filter_composition_fails()** (8 connections) — `tests/homesec/test_cleanup_clips.py`
- **test_cleanup_releases_storage_and_state_when_postgres_init_fails()** (7 connections) — `tests/homesec/test_cleanup_clips.py`
- **_process_candidate()** (7 connections) — `src/homesec/maintenance/cleanup_clips.py`
- **_write_cleanup_config()** (6 connections) — `tests/homesec/test_cleanup_clips.py`
- **.shutdown()** (3 connections) — `tests/homesec/test_cleanup_clips.py`
- **.initialize()** (3 connections) — `tests/homesec/test_cleanup_clips.py`
- **_base_payload()** (3 connections) — `src/homesec/maintenance/cleanup_clips.py`
- **_log_json()** (3 connections) — `src/homesec/maintenance/cleanup_clips.py`
- **_recheck_settings()** (3 connections) — `src/homesec/maintenance/cleanup_clips.py`
- **.detect()** (2 connections) — `tests/homesec/test_cleanup_clips.py`
- **.ping()** (2 connections) — `tests/homesec/test_cleanup_clips.py`
- **_build_recheck_filter_config()** (2 connections) — `src/homesec/maintenance/cleanup_clips.py`
- *... and 23 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/maintenance/cleanup_clips.py`
- `src/homesec/plugins/storage/local.py`
- `tests/homesec/test_cleanup_clips.py`

## Audit Trail

- EXTRACTED: 157 (56%)
- INFERRED: 123 (44%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*