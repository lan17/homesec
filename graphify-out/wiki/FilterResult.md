# FilterResult

> God node · 129 connections · `src/homesec/models/filter.py`

## Connections by Relation

### calls
- [[test_cleanup_deletes_empty_clips()]] `INFERRED`
- [[test_clip_flows_through_pipeline()]] `INFERRED`
- [[test_multiple_clips_processed_concurrently()]] `INFERRED`
- [[test_cleanup_marks_false_negatives()]] `INFERRED`
- [[test_pipeline_emits_success_events()]] `INFERRED`
- [[test_pipeline_emits_notification_events_per_notifier()]] `INFERRED`
- [[test_pipeline_records_alert_decision_without_notification_events_when_no_notifiers()]] `INFERRED`
- [[test_pipeline_emits_vlm_skipped_event()]] `INFERRED`
- [[test_pipeline_emits_vlm_skipped_event_for_run_mode_never()]] `INFERRED`
- [[test_pipeline_emits_upload_failed_event()]] `INFERRED`
- [[test_get_clip_includes_analysis_and_alert_details()]] `INFERRED`
- [[_make_filter_result()]] `INFERRED`
- [[test_vlm_overlaps_upload()]] `INFERRED`
- [[mocks()]] `INFERRED`
- [[test_multiple_clips_processed_concurrently()]] `INFERRED`
- [[test_runtime_assembly_skips_analyzer_load_when_run_mode_never()]] `INFERRED`
- [[test_run_mode_always_runs_vlm_regardless()]] `INFERRED`
- [[test_stage_retries_succeed()]] `INFERRED`
- [[test_notify_on_motion_override_alerts_when_vlm_skipped()]] `INFERRED`
- [[test_vlm_skipped_when_no_trigger_classes()]] `INFERRED`

### contains
- [[filter.py]] `EXTRACTED`

### inherits
- [[BaseModel]] `EXTRACTED`

### rationale_for
- [[Result from object detection filter on a video clip.]] `EXTRACTED`

### uses
- [[ClipRepository]] `INFERRED`
- [[_StubApp]] `INFERRED`
- [[_StubRepository]] `INFERRED`
- [[_StubStorage]] `INFERRED`
- [[ClipStateData]] `INFERRED`
- [[ClipPipeline]] `INFERRED`
- [[Clip]] `INFERRED`
- [[OpenAIConfig]] `INFERRED`
- [[YoloFilterConfig]] `INFERRED`
- [[_FakeController]] `INFERRED`
- [[Notifier]] `INFERRED`
- [[EventStore]] `INFERRED`
- [[AlertPolicy]] `INFERRED`
- [[ClipSource]] `INFERRED`
- [[StorageBackend]] `INFERRED`
- [[ObjectFilter]] `INFERRED`
- [[StateStore]] `INFERRED`
- [[VLMAnalyzer]] `INFERRED`
- [[ClipListCursor]] `INFERRED`
- [[ClipListPage]] `INFERRED`

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*