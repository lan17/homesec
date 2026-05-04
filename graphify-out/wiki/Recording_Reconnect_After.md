# Recording Reconnect After

> 127 nodes · cohesion 0.04

## Key Concepts

- **RTSPSource** (150 connections) — `src/homesec/sources/rtsp/core.py`
- **test_runtime.py** (58 connections) — `tests/homesec/rtsp/test_runtime.py`
- **_make_config()** (48 connections) — `tests/homesec/rtsp/test_runtime.py`
- **FakeClock** (38 connections) — `tests/homesec/rtsp/test_runtime.py`
- **FakeRecorder** (36 connections) — `tests/homesec/rtsp/test_runtime.py`
- **FakeFramePipeline** (33 connections) — `tests/homesec/rtsp/test_runtime.py`
- **DummyProc** (23 connections) — `tests/homesec/rtsp/test_runtime.py`
- **test_recording_survives_short_stall()** (10 connections) — `tests/homesec/rtsp/test_runtime.py`
- **test_recording_rotates_after_max_duration()** (9 connections) — `tests/homesec/rtsp/test_runtime.py`
- **test_rtsp_start_degrades_when_startup_preflight_fails()** (9 connections) — `tests/homesec/rtsp/test_runtime.py`
- **test_stall_grace_applies_while_recording()** (9 connections) — `tests/homesec/rtsp/test_runtime.py`
- **test_detect_stream_recovers_after_probe()** (8 connections) — `tests/homesec/rtsp/test_runtime.py`
- **test_reconnect_defers_detect_fallback_while_recording()** (8 connections) — `tests/homesec/rtsp/test_runtime.py`
- **test_reconnect_no_frames_exhausts()** (8 connections) — `tests/homesec/rtsp/test_runtime.py`
- **test_recording_restarts_when_dead()** (8 connections) — `tests/homesec/rtsp/test_runtime.py`
- **test_detect_fallback_after_attempts()** (7 connections) — `tests/homesec/rtsp/test_runtime.py`
- **test_reconnect_exhausted_returns_true()** (7 connections) — `tests/homesec/rtsp/test_runtime.py`
- **test_reconnect_respects_max_attempts_exact()** (7 connections) — `tests/homesec/rtsp/test_runtime.py`
- **test_reconnect_retries_detect_stream_when_fallback_fails()** (7 connections) — `tests/homesec/rtsp/test_runtime.py`
- **test_reconnect_retries_until_success()** (7 connections) — `tests/homesec/rtsp/test_runtime.py`
- **test_recording_lifecycle_tolerates_live_publisher_sync_failures()** (7 connections) — `tests/homesec/rtsp/test_runtime.py`
- **test_recording_start_backoff_throttles_retries()** (7 connections) — `tests/homesec/rtsp/test_runtime.py`
- **test_recording_state_changes_are_forwarded_to_live_publisher()** (7 connections) — `tests/homesec/rtsp/test_runtime.py`
- **test_stop_delay_resets_on_motion()** (7 connections) — `tests/homesec/rtsp/test_runtime.py`
- **test_preview_methods_delegate_to_live_publisher()** (6 connections) — `tests/homesec/rtsp/test_runtime.py`
- *... and 102 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/sources/rtsp/core.py`
- `tests/homesec/rtsp/test_runtime.py`

## Audit Trail

- EXTRACTED: 553 (74%)
- INFERRED: 190 (26%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*