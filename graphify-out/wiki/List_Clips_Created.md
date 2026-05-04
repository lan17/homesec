# List Clips Created

> 57 nodes · cohesion 0.04

## Key Concepts

- **test_state_store.py** (28 connections) — `tests/homesec/test_state_store.py`
- **.set()** (24 connections) — `tests/homesec/test_yolo_filter.py`
- **sample_state()** (10 connections) — `tests/homesec/test_state_store.py`
- **reset_store_tables()** (5 connections) — `tests/homesec/postgres_test_support.py`
- **test_list_clips_applies_risk_and_activity_filters()** (5 connections) — `tests/homesec/test_state_store.py`
- **clean_test_db()** (4 connections) — `tests/homesec/conftest.py`
- **state_store()** (4 connections) — `tests/homesec/test_state_store.py`
- **test_get_many_with_created_at_chunks_large_id_sets()** (4 connections) — `tests/homesec/test_state_store.py`
- **test_get_many_with_created_at_roundtrip()** (4 connections) — `tests/homesec/test_state_store.py`
- **test_graceful_degradation_uninitialized()** (4 connections) — `tests/homesec/test_state_store.py`
- **test_list_candidate_clips_for_cleanup_skips_deleted_and_filters_camera()** (4 connections) — `tests/homesec/test_state_store.py`
- **test_list_clips_alerted_false_includes_false_and_missing()** (4 connections) — `tests/homesec/test_state_store.py`
- **test_list_clips_applies_filters_and_includes_clip_ids()** (4 connections) — `tests/homesec/test_state_store.py`
- **test_list_clips_detected_filter_distinguishes_present_and_missing()** (4 connections) — `tests/homesec/test_state_store.py`
- **test_list_clips_keyset_cursor_uses_created_at_and_clip_id()** (4 connections) — `tests/homesec/test_state_store.py`
- **.get_many_with_created_at()** (4 connections) — `tests/homesec/mocks/state_store.py`
- **test_count_alerts_since_returns_zero_when_uninitialized()** (3 connections) — `tests/homesec/test_state_store.py`
- **test_get_clip_returns_clip_id_and_created_at()** (3 connections) — `tests/homesec/test_state_store.py`
- **test_initialize_returns_false_on_bad_dsn()** (3 connections) — `tests/homesec/test_state_store.py`
- **test_list_clips_excludes_deleted_by_default()** (3 connections) — `tests/homesec/test_state_store.py`
- **test_mark_clip_deleted_updates_persisted_status()** (3 connections) — `tests/homesec/test_state_store.py`
- **test_parse_state_data_accepts_bytes()** (3 connections) — `tests/homesec/test_state_store.py`
- **test_parse_state_data_accepts_dict_and_str()** (3 connections) — `tests/homesec/test_state_store.py`
- **test_upsert_and_get_roundtrip()** (3 connections) — `tests/homesec/test_state_store.py`
- **test_upsert_updates_existing()** (3 connections) — `tests/homesec/test_state_store.py`
- *... and 32 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/conftest.py`
- `tests/homesec/mocks/state_store.py`
- `tests/homesec/postgres_test_support.py`
- `tests/homesec/rtsp/test_runtime.py`
- `tests/homesec/test_runtime_manager.py`
- `tests/homesec/test_state_store.py`
- `tests/homesec/test_yolo_filter.py`

## Audit Trail

- EXTRACTED: 125 (68%)
- INFERRED: 58 (32%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*