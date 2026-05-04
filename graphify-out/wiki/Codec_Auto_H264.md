# Codec Auto H264

> 27 nodes · cohesion 0.09

## Key Concepts

- **LivePublisherStatus** (81 connections) — `src/homesec/sources/rtsp/live_publisher.py`
- **FakeDiscovery** (19 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **test_hardware_transcode_falls_back_to_software_when_encoder_startup_fails()** (8 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **test_transcode_prefers_matching_hardware_encoder_when_available()** (8 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **test_auto_codec_prefers_copy_for_h264_and_aac()** (7 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **test_auto_codec_transcodes_mp4_safe_but_hls_unsafe_audio()** (7 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **test_auto_codec_transcodes_non_browser_safe_source()** (7 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **test_h264_codec_copies_already_h264_video()** (7 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **test_request_stop_preempts_slow_startup_without_error()** (5 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **test_intervening_temporary_start_failure_resets_recording_failure_counter()** (4 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **test_successful_start_resets_recording_failure_counter()** (4 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **.preview_status()** (3 connections) — `src/homesec/sources/rtsp/core.py`
- **.__init__()** (2 connections) — `src/homesec/sources/rtsp/live_publisher.py`
- **.request_stop()** (2 connections) — `src/homesec/sources/rtsp/live_publisher.py`
- **.shutdown()** (2 connections) — `src/homesec/sources/rtsp/live_publisher.py`
- **.__init__()** (2 connections) — `tests/homesec/rtsp/test_runtime.py`
- **Return the current preview publisher status for this camera.** (1 connections) — `src/homesec/sources/rtsp/core.py`
- **.__init__()** (1 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **A successful concurrent preview start should break a failure streak.** (1 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **A non-session start failure should break a session-budget failure streak.** (1 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **Explicit stop should cancel a slow startup instead of surfacing preview failure.** (1 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **auto codec mode should copy browser-safe source codecs.** (1 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **auto codec mode should transcode non-browser HLS audio even if recording can cop** (1 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **h264 video mode should avoid transcoding sources that already satisfy the output** (1 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- **auto codec mode should transcode unsupported source codecs to browser-safe outpu** (1 connections) — `tests/homesec/rtsp/test_live_publisher.py`
- *... and 2 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/sources/rtsp/core.py`
- `src/homesec/sources/rtsp/live_publisher.py`
- `tests/homesec/rtsp/test_live_publisher.py`
- `tests/homesec/rtsp/test_runtime.py`

## Audit Trail

- EXTRACTED: 88 (49%)
- INFERRED: 91 (51%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*