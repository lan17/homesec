# Runtime Active Make

> 80 nodes · cohesion 0.05

## Key Concepts

- **RuntimeManager** (50 connections) — `src/homesec/runtime/manager.py`
- **_FakeController** (47 connections) — `tests/homesec/test_runtime_manager.py`
- **LocalFolderSourceConfig** (39 connections) — `src/homesec/sources/local_folder.py`
- **_StubFilter** (31 connections) — `tests/homesec/test_runtime_manager.py`
- **_StubNotifier** (31 connections) — `tests/homesec/test_runtime_manager.py`
- **_StubAlertPolicy** (29 connections) — `tests/homesec/test_runtime_manager.py`
- **_StubPipeline** (29 connections) — `tests/homesec/test_runtime_manager.py`
- **_make_config()** (23 connections) — `tests/homesec/test_runtime_manager.py`
- **test_runtime_manager.py** (20 connections) — `tests/homesec/test_runtime_manager.py`
- **test_runtime_manager_delegates_preview_methods_to_active_runtime()** (11 connections) — `tests/homesec/test_runtime_manager.py`
- **_make_runtime_bundle()** (10 connections) — `tests/homesec/test_runtime_manager.py`
- **test_runtime_manager_preview_methods_require_active_runtime()** (8 connections) — `tests/homesec/test_runtime_manager.py`
- **test_runtime_manager_shutdown_cancels_stuck_reload_task()** (7 connections) — `tests/homesec/test_runtime_manager.py`
- **._run_reload()** (7 connections) — `src/homesec/runtime/manager.py`
- **._require_active_runtime()** (6 connections) — `src/homesec/runtime/manager.py`
- **.start_initial_runtime()** (6 connections) — `src/homesec/runtime/manager.py`
- **test_runtime_manager_fails_initial_start_when_activation_callback_fails()** (5 connections) — `tests/homesec/test_runtime_manager.py`
- **test_runtime_manager_happy_path_swaps_runtime()** (5 connections) — `tests/homesec/test_runtime_manager.py`
- **test_runtime_manager_rejects_concurrent_reload_requests()** (5 connections) — `tests/homesec/test_runtime_manager.py`
- **test_runtime_manager_rollback_when_activation_callback_fails()** (5 connections) — `tests/homesec/test_runtime_manager.py`
- **test_runtime_manager_rolls_back_when_candidate_start_fails()** (5 connections) — `tests/homesec/test_runtime_manager.py`
- **test_runtime_manager_start_initial_runtime_logs_warning_on_double_call()** (5 connections) — `tests/homesec/test_runtime_manager.py`
- **test_runtime_manager_survives_candidate_cleanup_errors()** (5 connections) — `tests/homesec/test_runtime_manager.py`
- **.shutdown()** (5 connections) — `src/homesec/runtime/manager.py`
- **manager.py** (5 connections) — `src/homesec/runtime/manager.py`
- *... and 55 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/runtime/manager.py`
- `src/homesec/sources/local_folder.py`
- `tests/homesec/test_runtime_manager.py`

## Audit Trail

- EXTRACTED: 256 (52%)
- INFERRED: 239 (48%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*