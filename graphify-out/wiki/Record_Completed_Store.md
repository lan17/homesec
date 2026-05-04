# Record Completed Store

> 95 nodes · cohesion 0.04

## Key Concepts

- **ClipRepository** (104 connections) — `src/homesec/repository/clip_repository.py`
- **Clip** (68 connections) — `src/homesec/models/clip.py`
- **MockStateStore** (32 connections) — `tests/homesec/mocks/state_store.py`
- **MockEventStore** (29 connections) — `tests/homesec/mocks/event_store.py`
- **create_event_store_for_postgres_state_store()** (26 connections) — `src/homesec/state/postgres.py`
- **test_clip_repository.py** (19 connections) — `tests/homesec/test_clip_repository.py`
- **LocalRetentionPruner** (18 connections) — `src/homesec/retention/pruner.py`
- **._run_with_retries()** (10 connections) — `src/homesec/repository/clip_repository.py`
- **_build_state()** (8 connections) — `tests/homesec/test_clip_repository.py`
- **_seed_state()** (8 connections) — `tests/homesec/test_retention_pruner.py`
- **test_retention_pruner.py** (8 connections) — `tests/homesec/test_retention_pruner.py`
- **test_record_vlm_completed()** (7 connections) — `tests/homesec/test_clip_repository.py`
- **test_record_vlm_completed_applies_explicit_status_transition_rules()** (7 connections) — `tests/homesec/test_clip_repository.py`
- **test_get_clip_states_with_created_at_uses_batch_lookup()** (6 connections) — `tests/homesec/test_clip_repository.py`
- **test_initialize_clip_records_timezone_aware_timestamp()** (6 connections) — `tests/homesec/test_clip_repository.py`
- **test_record_alert_decision_records_state_and_event()** (6 connections) — `tests/homesec/test_clip_repository.py`
- **test_record_clip_rechecked_updates_state_and_event()** (6 connections) — `tests/homesec/test_clip_repository.py`
- **test_record_filter_completed()** (6 connections) — `tests/homesec/test_clip_repository.py`
- **test_prune_deletes_oldest_until_under_limit()** (6 connections) — `tests/homesec/test_retention_pruner.py`
- **test_prune_discovers_local_dir_from_clip_arrival_path()** (6 connections) — `tests/homesec/test_retention_pruner.py`
- **test_prune_reports_incomplete_measurement_on_stat_error()** (6 connections) — `tests/homesec/test_retention_pruner.py`
- **test_prune_respects_upload_and_done_gates()** (6 connections) — `tests/homesec/test_retention_pruner.py`
- **test_prune_tie_breaks_by_clip_id_for_equal_created_at()** (6 connections) — `tests/homesec/test_retention_pruner.py`
- **test_prune_uses_local_files_as_source_of_candidates()** (6 connections) — `tests/homesec/test_retention_pruner.py`
- **test_build_local_retention_pruner_starts_empty_and_uses_configured_cap()** (6 connections) — `tests/homesec/test_retention_wiring.py`
- *... and 70 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/models/clip.py`
- `src/homesec/repository/clip_repository.py`
- `src/homesec/retention/pruner.py`
- `src/homesec/retention/wiring.py`
- `src/homesec/runtime/bootstrap.py`
- `src/homesec/state/postgres.py`
- `tests/homesec/mocks/event_store.py`
- `tests/homesec/mocks/state_store.py`
- `tests/homesec/test_clip_repository.py`
- `tests/homesec/test_pipeline.py`
- `tests/homesec/test_retention_pruner.py`
- `tests/homesec/test_retention_wiring.py`

## Audit Trail

- EXTRACTED: 247 (44%)
- INFERRED: 319 (56%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*