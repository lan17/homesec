# ClipRepository

> God node · 104 connections · `src/homesec/repository/clip_repository.py`

## Connections by Relation

### calls
- [[make_repository()]] `INFERRED`
- [[run_cleanup()]] `INFERRED`
- [[test_clip_flows_through_pipeline()]] `INFERRED`
- [[test_multiple_clips_processed_concurrently()]] `INFERRED`
- [[test_pipeline_emits_success_events()]] `INFERRED`
- [[test_pipeline_emits_notification_events_per_notifier()]] `INFERRED`
- [[test_pipeline_records_alert_decision_without_notification_events_when_no_notifiers()]] `INFERRED`
- [[test_pipeline_emits_vlm_skipped_event()]] `INFERRED`
- [[test_pipeline_emits_vlm_skipped_event_for_run_mode_never()]] `INFERRED`
- [[test_pipeline_emits_upload_failed_event()]] `INFERRED`
- [[test_vlm_overlaps_upload()]] `INFERRED`
- [[test_record_vlm_completed()]] `INFERRED`
- [[test_record_vlm_completed_applies_explicit_status_transition_rules()]] `INFERRED`
- [[test_build_local_retention_pruner_starts_empty_and_uses_configured_cap()]] `INFERRED`
- [[test_prune_discovers_local_dir_from_clip_arrival_path()]] `INFERRED`
- [[test_prune_deletes_oldest_until_under_limit()]] `INFERRED`
- [[test_prune_respects_upload_and_done_gates()]] `INFERRED`
- [[test_prune_uses_local_files_as_source_of_candidates()]] `INFERRED`
- [[test_prune_tie_breaks_by_clip_id_for_equal_created_at()]] `INFERRED`
- [[test_prune_reports_incomplete_measurement_on_stat_error()]] `INFERRED`

### contains
- [[clip_repository.py]] `EXTRACTED`

### method
- [[._safe_append()]] `EXTRACTED`
- [[._safe_upsert()]] `EXTRACTED`
- [[._load_state()]] `EXTRACTED`
- [[._run_with_retries()]] `EXTRACTED`
- [[.record_upload_completed()]] `EXTRACTED`
- [[.record_filter_failed()]] `EXTRACTED`
- [[.record_vlm_completed()]] `EXTRACTED`
- [[.record_notification_sent()]] `EXTRACTED`
- [[.record_clip_deleted()]] `EXTRACTED`
- [[.initialize_clip()]] `EXTRACTED`
- [[.record_filter_completed()]] `EXTRACTED`
- [[.record_alert_decision()]] `EXTRACTED`
- [[.record_clip_rechecked()]] `EXTRACTED`
- [[.record_upload_started()]] `EXTRACTED`
- [[.record_upload_failed()]] `EXTRACTED`
- [[.record_filter_started()]] `EXTRACTED`
- [[.record_vlm_started()]] `EXTRACTED`
- [[.record_vlm_failed()]] `EXTRACTED`
- [[.record_vlm_skipped()]] `EXTRACTED`
- [[.record_notification_failed()]] `EXTRACTED`

### rationale_for
- [[Coordinates state + event writes with best-effort retries.      State is the aut]] `EXTRACTED`

### uses
- [[FilterResult]] `INFERRED`
- [[ClipStateData]] `INFERRED`
- [[AnalysisResult]] `INFERRED`
- [[Clip]] `INFERRED`
- [[ClipStatus]] `INFERRED`
- [[EventStore]] `INFERRED`
- [[AlertDecision]] `INFERRED`
- [[ClipListCursor]] `INFERRED`
- [[StateStore]] `INFERRED`
- [[ClipListPage]] `INFERRED`
- [[VLMSkipReason]] `INFERRED`
- [[SetupTestConnectionRequestError]] `INFERRED`
- [[SetupFinalizeValidationError]] `INFERRED`
- [[_PingablePlugin]] `INFERRED`
- [[CleanupOptions]] `INFERRED`
- [[RetryConfig]] `INFERRED`
- [[AlertDecisionMadeEvent]] `INFERRED`
- [[ClipRecordedEvent]] `INFERRED`
- [[UploadCompletedEvent]] `INFERRED`
- [[ClipDeletedEvent]] `INFERRED`

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*