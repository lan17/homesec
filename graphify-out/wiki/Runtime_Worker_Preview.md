# Runtime Worker Preview

> 29 nodes · cohesion 0.13

## Key Concepts

- **_make_config()** (21 connections) — `tests/homesec/test_runtime_worker.py`
- **test_runtime_worker.py** (19 connections) — `tests/homesec/test_runtime_worker.py`
- **_make_service()** (15 connections) — `tests/homesec/test_runtime_worker.py`
- **test_runtime_worker_create_sources_wires_rtsp_preview_publisher()** (7 connections) — `tests/homesec/test_runtime_worker.py`
- **test_runtime_worker_handle_command_connection_offloads_preview_work()** (6 connections) — `tests/homesec/test_runtime_worker.py`
- **test_runtime_worker_handle_command_connection_swallows_client_disconnects()** (5 connections) — `tests/homesec/test_runtime_worker.py`
- **test_runtime_worker_preview_ensure_command_degrades_when_source_raises()** (5 connections) — `tests/homesec/test_runtime_worker.py`
- **test_runtime_worker_preview_ensure_command_returns_machine_readable_refusal()** (5 connections) — `tests/homesec/test_runtime_worker.py`
- **test_runtime_worker_preview_force_stop_command_rejects_when_source_raises()** (5 connections) — `tests/homesec/test_runtime_worker.py`
- **test_runtime_worker_preview_force_stop_command_returns_stopping_ack()** (5 connections) — `tests/homesec/test_runtime_worker.py`
- **test_runtime_worker_preview_status_command_reports_runtime_preview_state()** (5 connections) — `tests/homesec/test_runtime_worker.py`
- **test_runtime_worker_run_runtime_skips_analyzer_load_when_run_mode_never()** (5 connections) — `tests/homesec/test_runtime_worker.py`
- **test_runtime_worker_create_notifier_skips_disabled_entries()** (4 connections) — `tests/homesec/test_runtime_worker.py`
- **test_runtime_worker_event_payload_stays_under_unix_datagram_limit()** (4 connections) — `tests/homesec/test_runtime_worker.py`
- **test_runtime_worker_preview_status_collection_degrades_when_source_raises()** (4 connections) — `tests/homesec/test_runtime_worker.py`
- **test_runtime_worker_create_notifier_returns_noop_when_notifier_list_empty()** (3 connections) — `tests/homesec/test_runtime_worker.py`
- **test_worker_main_uses_shared_logging_configuration()** (2 connections) — `tests/homesec/test_runtime_worker.py`
- **Tests for runtime worker entrypoint behavior.** (1 connections) — `tests/homesec/test_runtime_worker.py`
- **Worker main should configure shared logging before running service.** (1 connections) — `tests/homesec/test_runtime_worker.py`
- **run_mode=never should avoid analyzer plugin loading in worker startup.** (1 connections) — `tests/homesec/test_runtime_worker.py`
- **Preview status command should map source preview status into runtime fields.** (1 connections) — `tests/homesec/test_runtime_worker.py`
- **RTSP source creation should inject the configured live preview publisher.** (1 connections) — `tests/homesec/test_runtime_worker.py`
- **Preview status collection should fail closed when a source raises.** (1 connections) — `tests/homesec/test_runtime_worker.py`
- **Heartbeat events should stay within the conservative Unix datagram size budget.** (1 connections) — `tests/homesec/test_runtime_worker.py`
- **Preview ensure command should preserve refusal semantics as structured data.** (1 connections) — `tests/homesec/test_runtime_worker.py`
- *... and 4 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/test_runtime_worker.py`

## Audit Trail

- EXTRACTED: 113 (86%)
- INFERRED: 19 (14%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*