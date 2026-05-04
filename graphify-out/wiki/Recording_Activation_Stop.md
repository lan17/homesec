# Recording Activation Stop

> 27 nodes · cohesion 0.12

## Key Concepts

- **SlowDiscovery** (20 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **_probe_stream()** (19 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **.wait()** (16 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **test_recording_priority_cancels_waiting_activation_after_recording_window()** (8 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **test_request_stop_cancels_activation_that_queued_after_stop()** (8 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **test_request_stop_cancels_waiting_activation_queue()** (8 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **test_request_stop_keeps_cancelled_queue_from_observing_later_restart()** (8 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **test_hardware_transcode_does_not_retry_software_on_session_budget_exhaustion()** (7 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **test_recording_priority_cancels_stale_probe_when_recording_ends_before_release()** (7 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **test_recording_priority_preempts_slow_probe_before_spawn()** (7 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **test_request_stop_preempts_slow_probe_before_spawn()** (7 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **test_waiting_activation_inherits_failed_start_result()** (7 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **test_waiting_activation_keeps_failed_result_after_later_restart()** (7 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **test_recording_priority_preempts_slow_startup()** (5 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **.probe()** (4 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **Recording activation should preempt a slow preview startup without blocking.** (1 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **Explicit stop should cancel startup before ffmpeg spawn when probing is slow.** (1 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **Explicit stop should prevent queued activation callers from immediately restarti** (1 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **Explicit stop should fence callers that queue behind a start already being cance** (1 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **Explicit stop should keep stale queued callers from inheriting a later restart.** (1 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **Queued activation callers should share the failed start they were waiting on.** (1 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **Queued activation callers should not inherit a later restart after their start a** (1 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **Recording activation should interrupt startup before ffmpeg spawn when probing i** (1 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **Recording preemption should fence a slow probe even if recording ends before it** (1 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **Recording preemption should fence queued activations after recording has ended.** (1 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- *... and 2 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/rtsp/test_live_publisher.py`

## Audit Trail

- EXTRACTED: 130 (87%)
- INFERRED: 20 (13%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*