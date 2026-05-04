# ClipStateData

> God node · 95 connections · `src/homesec/models/clip.py`

## Connections by Relation

### calls
- [[test_cleanup_deletes_empty_clips()]] `INFERRED`
- [[test_cleanup_marks_false_negatives()]] `INFERRED`
- [[test_get_clip_includes_analysis_and_alert_details()]] `INFERRED`
- [[test_list_clips_cursor_pagination()]] `INFERRED`
- [[sample_state()]] `INFERRED`
- [[test_get_clip_media_accepts_api_key_when_auth_enabled()]] `INFERRED`
- [[test_get_clip_media_rejects_invalid_token_when_auth_enabled()]] `INFERRED`
- [[test_get_clip_media_rejects_missing_token_when_auth_enabled()]] `INFERRED`
- [[test_get_clip_media_accepts_valid_media_token_when_auth_enabled()]] `INFERRED`
- [[test_get_clip_media_missing_storage_uri_returns_409()]] `INFERRED`
- [[test_get_clip_media_proxy_success_returns_inline_video()]] `INFERRED`
- [[test_get_clip_media_defaults_filename_suffix_when_storage_uri_has_no_extension()]] `INFERRED`
- [[test_get_clip_media_storage_failure_returns_502()]] `INFERRED`
- [[test_get_clip_media_success_cleans_temp_directory()]] `INFERRED`
- [[test_create_clip_media_token_auth_disabled_returns_direct_media_url()]] `INFERRED`
- [[test_create_clip_media_token_rejects_token_only_auth()]] `INFERRED`
- [[test_delete_clip_storage_failure_returns_500()]] `INFERRED`
- [[_seed_state()]] `INFERRED`
- [[test_create_clip_media_token_missing_storage_uri_returns_409()]] `INFERRED`
- [[test_delete_clip_success_removes_storage()]] `INFERRED`

### contains
- [[clip.py]] `EXTRACTED`

### inherits
- [[BaseModel]] `EXTRACTED`

### rationale_for
- [[Lightweight snapshot of current clip state (stored in clip_states.data JSONB).]] `EXTRACTED`

### uses
- [[FilterResult]] `INFERRED`
- [[ClipRepository]] `INFERRED`
- [[_StubApp]] `INFERRED`
- [[_StubRepository]] `INFERRED`
- [[_StubStorage]] `INFERRED`
- [[PostgresStateStore]] `INFERRED`
- [[AnalysisResult]] `INFERRED`
- [[Notifier]] `INFERRED`
- [[NoopEventStore]] `INFERRED`
- [[ClipStatus]] `INFERRED`
- [[EventStore]] `INFERRED`
- [[AlertPolicy]] `INFERRED`
- [[ClipSource]] `INFERRED`
- [[StorageBackend]] `INFERRED`
- [[ObjectFilter]] `INFERRED`
- [[AlertDecision]] `INFERRED`
- [[NoopStateStore]] `INFERRED`
- [[StateStore]] `INFERRED`
- [[VLMAnalyzer]] `INFERRED`
- [[PostgresEventStore]] `INFERRED`

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*