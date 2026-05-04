# Vlm When Failure

> 34 nodes · cohesion 0.18

## Key Concepts

- **test_pipeline.py** (41 connections) — `tests/homesec/test_pipeline.py`
- **MockRetentionPruner** (36 connections) — `tests/homesec/mocks/retention_pruner.py`
- **make_alert_policy()** (28 connections) — `tests/homesec/test_pipeline.py`
- **make_repository()** (25 connections) — `tests/homesec/test_pipeline.py`
- **test_vlm_overlaps_upload()** (10 connections) — `tests/homesec/test_pipeline.py`
- **test_multiple_clips_processed_concurrently()** (9 connections) — `tests/homesec/test_pipeline.py`
- **test_no_notification_when_below_risk_threshold()** (8 connections) — `tests/homesec/test_pipeline.py`
- **test_notify_on_motion_override_alerts_when_vlm_skipped()** (8 connections) — `tests/homesec/test_pipeline.py`
- **test_run_mode_always_runs_vlm_regardless()** (8 connections) — `tests/homesec/test_pipeline.py`
- **test_stage_retries_succeed()** (8 connections) — `tests/homesec/test_pipeline.py`
- **test_notification_sent_when_above_risk_threshold()** (7 connections) — `tests/homesec/test_pipeline.py`
- **test_run_mode_never_skips_vlm_with_explicit_reason()** (7 connections) — `tests/homesec/test_pipeline.py`
- **test_shutdown_waits_for_in_flight_clips()** (7 connections) — `tests/homesec/test_pipeline.py`
- **test_vlm_failure_continues_to_notify()** (7 connections) — `tests/homesec/test_pipeline.py`
- **test_vlm_failure_falls_back_to_filter_trigger()** (7 connections) — `tests/homesec/test_pipeline.py`
- **test_vlm_skipped_when_no_trigger_classes()** (7 connections) — `tests/homesec/test_pipeline.py`
- **test_filter_failure_aborts_processing()** (6 connections) — `tests/homesec/test_pipeline.py`
- **test_global_concurrency_limit()** (6 connections) — `tests/homesec/test_pipeline.py`
- **test_no_notification_when_notifier_entries_explicitly_empty()** (6 connections) — `tests/homesec/test_pipeline.py`
- **test_notify_failure_still_marks_done()** (6 connections) — `tests/homesec/test_pipeline.py`
- **test_stage_concurrency_limit()** (6 connections) — `tests/homesec/test_pipeline.py`
- **test_full_pipeline_with_person_detection()** (5 connections) — `tests/homesec/test_pipeline.py`
- **test_retention_failures_are_logged_and_non_fatal()** (5 connections) — `tests/homesec/test_pipeline.py`
- **test_retention_trigger_fires_on_clip_arrival()** (5 connections) — `tests/homesec/test_pipeline.py`
- **test_retention_triggered_on_clip_arrival_even_when_upload_fails()** (5 connections) — `tests/homesec/test_pipeline.py`
- *... and 9 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/mocks/retention_pruner.py`
- `tests/homesec/test_pipeline.py`

## Audit Trail

- EXTRACTED: 172 (59%)
- INFERRED: 121 (41%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*