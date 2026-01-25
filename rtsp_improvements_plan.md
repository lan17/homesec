# RTSP Improvements Plan (Reconnect + Architecture)

Goal: make RTSP source resilient to disconnects, avoid premature clip stops during motion, and improve code structure/testability while keeping configuration simple.

## Summary of Intent
- Retry RTSP reconnection forever by default.
- Make reconnect behavior consistent with startup.
- Keep recording alive during short stalls and restart recording immediately after failures.
- Preserve simple motion tuning: 2x more sensitive while recording (configurable with one factor).
- Add hermetic tests via injected fakes (no real ffmpeg/RTSP).

## Current Issues (Observed in Code)
- The main loop breaks on frame pipeline death or reconnect failure, which stops the source and kills recording.
- Reconnect behavior differs from startup because startup probes detect stream and can fall back to main; reconnect does not.
- Recording stop-delay relies on a fixed keepalive rule with no explicit config, making behavior opaque.
- `rtsp_io_timeout_s` is defined but unused by ffmpeg commands.
- No tests cover reconnect/stop-delay logic.

## Decisions
- Default reconnect to "retry forever" (`max_reconnect_attempts = 0`).
- Keep motion sensitivity simple: while recording, keepalive threshold is `min_changed_pct / recording_sensitivity_factor` (default factor = 2.0).
- Recording should restart immediately if it dies during motion (even without a new motion edge).
- Detect-stream fallback: switch motion detection to main stream after repeated failures (default attempts = 3, configurable), then probe and return to detect stream when it recovers. Skip fallback logic when no detect stream is configured.
- Detect-stream recovery probing: balanced default (probe every 20-30s with short timeouts, exponential backoff up to ~60s on repeated failures).

## Architecture Improvements

### 1) Interfaces (for testability and clarity)
- `FramePipeline` (RTSP decode to frames)
  - `start() -> None`
  - `read_frame(timeout_s: float) -> bytes | None`
  - `stop() -> None`
  - `probe_info(url: str) -> StreamInfo | None` (optional helper)
- `Recorder` (ffmpeg recording process)
  - `start(output_path, stderr_path) -> RecordingHandle | None`
  - `is_alive(handle) -> bool`
  - `stop(handle) -> None`
- `Clock`
  - `now() -> float`
  - `sleep(seconds: float) -> None`

Default implementations will wrap current subprocess/ffmpeg logic.

### 2) State Machine
States:
- `Idle` (no recording, waiting for motion)
- `Recording` (recording active, motion keepalive active)
- `Stalled` (no frames; keep recording for grace period)
- `Reconnecting` (attempting to restore frame pipeline)

Transitions:
- `Idle -> Recording` on motion
- `Recording -> Stalled` on frame timeout
- `Stalled -> Reconnecting` if pipeline down or no frames after grace
- `Reconnecting -> Idle/Recording` on success (resume state if recording ongoing)
- `Recording -> Idle` on stop-delay expiry

This removes "break" exits and makes reconnect behavior uniform.

### 3) Logging/Telemetry (DB logs)
Emit structured events via `extra`:
- `rtsp_reconnect_attempt`, `rtsp_reconnect_success`, `rtsp_reconnect_failure`
- `rtsp_detect_fallback_enabled`, `rtsp_detect_fallback_recovered`
- `rtsp_recording_restart`, `rtsp_recording_failed`
Include fields: `attempt`, `mode`, `detect_stream_source`, `detect_is_fallback`, `reconnect_backoff_s`.

## Behavior Changes
- Reconnect attempts never stop the source by default.
- If detect stream fails repeatedly, motion detection switches to main stream.
- Recording survives short stalls; if recorder dies, restart immediately.
- Motion keepalive sensitivity explicitly derived by one factor (default 2.0).

## Implementation Steps

1) Refactor internals for dependency injection
- Add `FramePipeline`, `Recorder`, `Clock` to `RTSPSource` with defaults.
- Wire `Clock` into sleep/time, and `FramePipeline` into frame reading.

2) State machine refactor
- Move main `_run` loop logic into state transitions.
- Replace loop `break`s with state transitions and cleanups.

3) Reconnect robustness
- Route all frame failures through `_handle_frame_timeout` and reconnection loop.
- Remove heartbeat "break" on pipeline death; attempt reconnect.
- Add detect-stream fallback after N failed reconnects.
- Periodically probe detect stream to return from fallback.
- Ensure recording restarts if it dies and recent motion exists.

4) ffmpeg timeouts
- Use `rtsp_io_timeout_s` in both detection and recording (e.g., `-rw_timeout`).
- Preserve existing `-stimeout` connect timeout behavior.

5) Motion keepalive
- Add `recording_sensitivity_factor` to config (default 2.0).
- Keepalive threshold = `min_changed_pct / factor` (clamped to >= 0).
- Log effective thresholds at startup.

6) Tests (hermetic)
- Fake FramePipeline: returns frames, stalls, or dies deterministically.
- Fake Recorder: simulates process death and restart.
- Fake Clock: control time without sleeps.
- Tests (Given/When/Then):
  - Reconnect attempts continue indefinitely when `max_reconnect_attempts=0`.
  - Detect fallback triggers after N failures and recovers after probe.
  - Recording does not stop during short stalls (stop-delay grace).
  - Recording restarts immediately after recorder death while motion is active.
  - Motion keepalive resets stop-delay at 2x sensitivity while recording.

7) Config + docs
- Update `RTSPSourceConfig` with `recording_sensitivity_factor`.
- Update `config/example.yaml` with defaults and comments.
- Note behavior changes in `CHANGELOG.md` if required by repo practice.

## Open Questions
- None (balanced detect probing chosen: ~25s interval with backoff up to ~60s).

## Estimated Impact
- More reliable reconnect behavior without manual restarts.
- Fewer premature clip stops during ongoing motion.
- Easier test coverage and future maintenance.
