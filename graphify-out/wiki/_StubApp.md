# _StubApp

> God node · 99 connections · `tests/homesec/test_api_routes.py`

## Connections by Relation

### calls
- [[test_get_clip_includes_analysis_and_alert_details()]] `EXTRACTED`
- [[test_list_clips_cursor_pagination()]] `EXTRACTED`
- [[test_ui_serving_serves_spa_shell_without_shadowing_api()]] `EXTRACTED`
- [[test_get_clip_media_accepts_api_key_when_auth_enabled()]] `EXTRACTED`
- [[test_get_clip_media_rejects_invalid_token_when_auth_enabled()]] `EXTRACTED`
- [[test_get_clip_media_rejects_missing_token_when_auth_enabled()]] `EXTRACTED`
- [[test_get_clip_media_accepts_valid_media_token_when_auth_enabled()]] `EXTRACTED`
- [[test_diagnostics_ignores_disabled_unhealthy_camera_for_global_status()]] `EXTRACTED`
- [[test_diagnostics_degrades_when_enabled_camera_has_no_source()]] `EXTRACTED`
- [[test_list_cameras_includes_health_fields()]] `EXTRACTED`
- [[test_list_cameras_serializes_model_config()]] `EXTRACTED`
- [[test_delete_camera_apply_changes_triggers_runtime_reload()]] `EXTRACTED`
- [[test_ui_serving_catch_all_rejects_path_traversal_input()]] `EXTRACTED`
- [[test_get_clip_media_missing_storage_uri_returns_409()]] `EXTRACTED`
- [[test_get_clip_media_proxy_success_returns_inline_video()]] `EXTRACTED`
- [[test_get_clip_media_defaults_filename_suffix_when_storage_uri_has_no_extension()]] `EXTRACTED`
- [[test_get_clip_media_storage_failure_returns_502()]] `EXTRACTED`
- [[test_get_clip_media_success_cleans_temp_directory()]] `EXTRACTED`
- [[test_create_clip_media_token_auth_disabled_returns_direct_media_url()]] `EXTRACTED`
- [[test_create_clip_media_token_rejects_token_only_auth()]] `EXTRACTED`

### contains
- [[test_api_routes.py]] `EXTRACTED`

### method
- [[.__init__()]] `EXTRACTED`
- [[.get_source()]] `EXTRACTED`
- [[.request_runtime_reload()]] `EXTRACTED`
- [[.request_restart()]] `EXTRACTED`
- [[.activate_setup_config()]] `EXTRACTED`

### uses
- [[FilterResult]] `INFERRED`
- [[ClipStateData]] `INFERRED`
- [[FastAPIServerConfig]] `INFERRED`
- [[AnalysisResult]] `INFERRED`
- [[CameraConfig]] `INFERRED`
- [[CameraSourceConfig]] `INFERRED`
- [[ClipStatus]] `INFERRED`
- [[ConfigManager]] `INFERRED`
- [[AlertDecision]] `INFERRED`
- [[ClipListCursor]] `INFERRED`
- [[ClipListPage]] `INFERRED`
- [[RuntimeReloadRequest]] `INFERRED`
- [[RiskLevel]] `INFERRED`
- [[RuntimeReloadConfigError]] `INFERRED`

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*