# Application

> God node · 105 connections · `src/homesec/app.py`

## Connections by Relation

### calls
- [[test_get_runtime_status_preserves_reloading_when_heartbeat_is_stale()]] `INFERRED`
- [[test_camera_health_degrades_when_runtime_heartbeat_is_stale()]] `INFERRED`
- [[test_get_runtime_status_uses_worker_exit_code_for_stale_runtime()]] `INFERRED`
- [[test_application_shuts_down_postgres_backup_manager_before_storage()]] `INFERRED`
- [[test_build_runtime_persistence_stack_prefers_env_dsn()]] `INFERRED`
- [[test_build_runtime_persistence_stack_raises_when_env_resolution_returns_empty()]] `INFERRED`
- [[test_application_wires_runtime_and_api()]] `INFERRED`
- [[test_application_cleans_up_started_backup_manager_when_api_creation_fails()]] `INFERRED`
- [[test_application_does_not_create_api_server_when_disabled()]] `INFERRED`
- [[.run()]] `INFERRED`
- [[test_application_run_starts_api_and_performs_graceful_shutdown()]] `INFERRED`
- [[test_pipeline_running_returns_false_without_runtime_or_after_shutdown()]] `INFERRED`
- [[test_application_shutdown_stops_api_before_runtime_manager()]] `INFERRED`
- [[test_request_runtime_reload_maps_semantic_config_error_to_422()]] `INFERRED`
- [[test_request_runtime_reload_maps_source_config_error_to_400()]] `INFERRED`
- [[test_application_preview_methods_delegate_to_runtime_manager()]] `INFERRED`
- [[test_application_run_enters_bootstrap_mode_when_config_missing()]] `INFERRED`
- [[test_repository_and_storage_accessors_require_initialization()]] `INFERRED`

### contains
- [[app.py]] `EXTRACTED`

### method
- [[._create_components()]] `EXTRACTED`
- [[._run_bootstrap()]] `EXTRACTED`
- [[._require_runtime_manager()]] `EXTRACTED`
- [[.run()]] `EXTRACTED`
- [[.request_runtime_reload()]] `EXTRACTED`
- [[._validate_config()]] `EXTRACTED`
- [[.__init__()]] `EXTRACTED`
- [[._camera_statuses()]] `EXTRACTED`
- [[.shutdown()]] `EXTRACTED`
- [[._build_runtime_persistence_stack()]] `EXTRACTED`
- [[._active_subprocess_runtime()]] `EXTRACTED`
- [[.get_runtime_status()]] `EXTRACTED`
- [[._setup_signal_handlers()]] `EXTRACTED`
- [[.activate_setup_config()]] `EXTRACTED`
- [[._create_runtime_controller()]] `EXTRACTED`
- [[.wait_for_runtime_reload()]] `EXTRACTED`
- [[.get_camera_preview_status()]] `EXTRACTED`
- [[.ensure_camera_preview_active()]] `EXTRACTED`
- [[.force_stop_camera_preview()]] `EXTRACTED`
- [[.note_camera_preview_viewer_activity()]] `EXTRACTED`

### rationale_for
- [[Main application that orchestrates all components.      Handles component creati]] `EXTRACTED`

### uses
- [[Config]] `INFERRED`
- [[FastAPIServerConfig]] `INFERRED`
- [[CameraPreviewStatus]] `INFERRED`
- [[SubprocessRuntimeController]] `INFERRED`
- [[PostgresBackupManager]] `INFERRED`
- [[RuntimeManager]] `INFERRED`
- [[CameraPreviewStartRefusal]] `INFERRED`
- [[CameraPreviewStopResult]] `INFERRED`
- [[EventStore]] `INFERRED`
- [[ConfigManager]] `INFERRED`
- [[_StubStorage]] `INFERRED`
- [[_StubRuntimeController]] `INFERRED`
- [[StorageBackend]] `INFERRED`
- [[_StubRuntimeManagerStatus]] `INFERRED`
- [[PluginType]] `INFERRED`
- [[_StubStateStore]] `INFERRED`
- [[_FakeProcess]] `INFERRED`
- [[SubprocessRuntimeHandle]] `INFERRED`
- [[StateStore]] `INFERRED`
- [[_RecordingEventStore]] `INFERRED`

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*