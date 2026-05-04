# Setup Finalize When

> 53 nodes · cohesion 0.07

## Key Concepts

- **ConfigManager** (43 connections) — `src/homesec/config/manager.py`
- **test_setup_service.py** (25 connections) — `tests/homesec/test_setup_service.py`
- **_StubApp** (24 connections) — `tests/homesec/test_setup_service.py`
- **_server_config()** (19 connections) — `src/homesec/services/setup.py`
- **_StubRepository** (11 connections) — `tests/homesec/test_setup_service.py`
- **test_finalize_setup_rejects_empty_camera_set_on_fresh_bootstrap()** (6 connections) — `tests/homesec/test_setup_service.py`
- **test_finalize_setup_rejects_missing_state_store_env_when_defaults_apply()** (6 connections) — `tests/homesec/test_setup_service.py`
- **test_finalize_setup_returns_validation_error_without_writing_config()** (6 connections) — `tests/homesec/test_setup_service.py`
- **test_finalize_setup_reuses_existing_sections_when_payload_omits_them()** (6 connections) — `tests/homesec/test_setup_service.py`
- **test_finalize_setup_validate_only_returns_success_without_writing_or_restart()** (6 connections) — `tests/homesec/test_setup_service.py`
- **test_finalize_setup_writes_config_with_defaults_and_activates_runtime()** (6 connections) — `tests/homesec/test_setup_service.py`
- **test_get_setup_status_returns_complete_when_pipeline_running()** (5 connections) — `tests/homesec/test_setup_service.py`
- **test_get_setup_status_returns_partial_without_pipeline()** (5 connections) — `tests/homesec/test_setup_service.py`
- **test_run_preflight_checks_pg_dump_from_existing_config_in_bootstrap_mode()** (5 connections) — `tests/homesec/test_setup_service.py`
- **test_run_preflight_marks_check_failed_when_timeout_expires()** (5 connections) — `tests/homesec/test_setup_service.py`
- **test_run_preflight_returns_all_passed_when_checks_succeed()** (5 connections) — `tests/homesec/test_setup_service.py`
- **test_disk_probe_path_walks_up_to_existing_parent()** (4 connections) — `tests/homesec/test_setup_service.py`
- **test_get_setup_status_marks_auth_unconfigured_when_key_missing()** (4 connections) — `tests/homesec/test_setup_service.py`
- **test_get_setup_status_returns_fresh_in_bootstrap_mode()** (4 connections) — `tests/homesec/test_setup_service.py`
- **test_run_preflight_probes_postgres_via_dsn_in_bootstrap_mode()** (4 connections) — `tests/homesec/test_setup_service.py`
- **test_run_preflight_reports_missing_state_store_env()** (4 connections) — `tests/homesec/test_setup_service.py`
- **_write_existing_config()** (4 connections) — `tests/homesec/test_setup_service.py`
- **server_config()** (2 connections) — `tests/homesec/test_setup_service.py`
- **.__init__()** (2 connections) — `tests/homesec/test_setup_service.py`
- **test_network_probe_returns_config_error_for_invalid_port_env()** (2 connections) — `tests/homesec/test_setup_service.py`
- *... and 28 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/config/manager.py`
- `src/homesec/services/setup.py`
- `tests/homesec/test_setup_service.py`

## Audit Trail

- EXTRACTED: 156 (64%)
- INFERRED: 86 (36%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*