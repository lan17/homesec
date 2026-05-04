# Pipeline Emits Events

> 41 nodes · cohesion 0.08

## Key Concepts

- **MockFilter** (26 connections) — `tests/homesec/mocks/filter.py`
- **MockVLM** (24 connections) — `tests/homesec/mocks/vlm.py`
- **MockNotifier** (21 connections) — `tests/homesec/mocks/notifier.py`
- **test_pipeline_emits_notification_events_per_notifier()** (14 connections) — `tests/homesec/test_pipeline_events.py`
- **test_pipeline_emits_success_events()** (14 connections) — `tests/homesec/test_pipeline_events.py`
- **test_pipeline_emits_upload_failed_event()** (13 connections) — `tests/homesec/test_pipeline_events.py`
- **test_pipeline_emits_vlm_skipped_event()** (13 connections) — `tests/homesec/test_pipeline_events.py`
- **test_pipeline_emits_vlm_skipped_event_for_run_mode_never()** (13 connections) — `tests/homesec/test_pipeline_events.py`
- **test_pipeline_records_alert_decision_without_notification_events_when_no_notifiers()** (13 connections) — `tests/homesec/test_pipeline_events.py`
- **test_pipeline_events.py** (10 connections) — `tests/homesec/test_pipeline_events.py`
- **make_alert_policy()** (9 connections) — `tests/homesec/test_pipeline_events.py`
- **make_clip()** (9 connections) — `tests/homesec/test_pipeline_events.py`
- **mocks()** (9 connections) — `tests/homesec/test_pipeline.py`
- **.__init__()** (3 connections) — `tests/homesec/mocks/filter.py`
- **.send()** (3 connections) — `tests/homesec/mocks/notifier.py`
- **.__init__()** (3 connections) — `tests/homesec/mocks/vlm.py`
- **.shutdown()** (2 connections) — `tests/homesec/mocks/filter.py`
- **.__init__()** (2 connections) — `tests/homesec/mocks/notifier.py`
- **.ping()** (2 connections) — `tests/homesec/mocks/notifier.py`
- **.shutdown()** (2 connections) — `tests/homesec/mocks/notifier.py`
- **.shutdown()** (2 connections) — `tests/homesec/mocks/vlm.py`
- **filter.py** (2 connections) — `tests/homesec/mocks/filter.py`
- **notifier.py** (2 connections) — `tests/homesec/mocks/notifier.py`
- **vlm.py** (2 connections) — `tests/homesec/mocks/vlm.py`
- **Integration tests for pipeline event emission.** (1 connections) — `tests/homesec/test_pipeline_events.py`
- *... and 16 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/mocks/filter.py`
- `tests/homesec/mocks/notifier.py`
- `tests/homesec/mocks/vlm.py`
- `tests/homesec/test_pipeline.py`
- `tests/homesec/test_pipeline_events.py`

## Audit Trail

- EXTRACTED: 107 (47%)
- INFERRED: 123 (53%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*