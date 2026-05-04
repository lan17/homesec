# RTSPSource

> God node · 150 connections · `src/homesec/sources/rtsp/core.py`

## Connections by Relation

### calls
- [[test_rtsp_start_degrades_when_startup_preflight_fails()]] `INFERRED`
- [[test_startup_preflight_downgrade_reaches_live_publisher()]] `INFERRED`
- [[test_startup_preflight_downgrade_failure_does_not_abort_source()]] `INFERRED`
- [[test_recording_rotates_after_max_duration()]] `INFERRED`
- [[test_stall_grace_applies_while_recording()]] `INFERRED`
- [[test_reconnect_no_frames_exhausts()]] `INFERRED`
- [[test_reconnect_defers_detect_fallback_while_recording()]] `INFERRED`
- [[test_recording_restarts_when_dead()]] `INFERRED`
- [[test_detect_stream_recovers_after_probe()]] `INFERRED`
- [[test_recording_state_changes_are_forwarded_to_live_publisher()]] `INFERRED`
- [[test_recording_lifecycle_tolerates_live_publisher_sync_failures()]] `INFERRED`
- [[test_reconnect_retries_until_success()]] `INFERRED`
- [[test_reconnect_respects_max_attempts_exact()]] `INFERRED`
- [[test_reconnect_exhausted_returns_true()]] `INFERRED`
- [[test_detect_fallback_after_attempts()]] `INFERRED`
- [[test_reconnect_retries_detect_stream_when_fallback_fails()]] `INFERRED`
- [[test_stop_delay_resets_on_motion()]] `INFERRED`
- [[test_recording_start_backoff_throttles_retries()]] `INFERRED`
- [[test_preview_methods_delegate_to_live_publisher()]] `INFERRED`
- [[test_shutdown_shuts_down_live_publisher_even_before_start()]] `INFERRED`

### contains
- [[core.py]] `EXTRACTED`

### inherits
- [[ThreadedClipSource]] `EXTRACTED`

### method
- [[.__init__()]] `EXTRACTED`
- [[._event_extra()]] `EXTRACTED`
- [[._reconnect_frame_pipeline()]] `EXTRACTED`
- [[._run()]] `EXTRACTED`
- [[.start_recording()]] `EXTRACTED`
- [[._rotate_recording_if_needed()]] `EXTRACTED`
- [[.stop_recording()]] `EXTRACTED`
- [[._handle_reconnect_needed()]] `EXTRACTED`
- [[._run_startup_preflight_until_ready()]] `EXTRACTED`
- [[.check_recording_health()]] `EXTRACTED`
- [[._maybe_recover_detect_stream()]] `EXTRACTED`
- [[._process_frame()]] `EXTRACTED`
- [[.stop()]] `EXTRACTED`
- [[._build_fallback_preflight_outcome()]] `EXTRACTED`
- [[._set_run_state()]] `EXTRACTED`
- [[._clear_recording_state()]] `EXTRACTED`
- [[._redact_rtsp_url()]] `EXTRACTED`
- [[._stop_recording_process()]] `EXTRACTED`
- [[._start_frame_pipeline()]] `EXTRACTED`
- [[._ensure_recording()]] `EXTRACTED`

### rationale_for
- [[RTSP clip source with motion detection.      Uses ffmpeg for frame extraction an]] `EXTRACTED`

### uses
- [[LivePublisherStatus]] `INFERRED`
- [[HLSLivePublisher]] `INFERRED`
- [[Clip]] `INFERRED`
- [[RTSPStartupPreflight]] `INFERRED`
- [[RTSPTimeoutCapabilities]] `INFERRED`
- [[HardwareAccelConfig]] `INFERRED`
- [[MotionProfile]] `INFERRED`
- [[PreviewConfig]] `INFERRED`
- [[FakeClock]] `INFERRED`
- [[LivePublisherStartRefusal]] `INFERRED`
- [[FakeRecorder]] `INFERRED`
- [[ThreadedClipSource]] `INFERRED`
- [[FfmpegFramePipeline]] `INFERRED`
- [[LivePublisherRefusalReason]] `INFERRED`
- [[LivePublisherState]] `INFERRED`
- [[FakeFramePipeline]] `INFERRED`
- [[NoopLivePublisher]] `INFERRED`
- [[FakeLivePublisher]] `INFERRED`
- [[LivePublisher]] `INFERRED`
- [[CameraPreflightDiagnostics]] `INFERRED`

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*