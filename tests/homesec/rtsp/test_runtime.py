"""Hermetic tests for RTSPSource reconnect and recording behavior."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from homesec.sources.rtsp.core import RTSPRunState, RTSPSource, RTSPSourceConfig
from homesec.sources.rtsp.frame_pipeline import FfmpegFramePipeline
from homesec.sources.rtsp.hardware import HardwareAccelConfig
from homesec.sources.rtsp.recorder import FfmpegRecorder


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    def now(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self._now += float(seconds)


class FakeFramePipeline:
    def __init__(
        self,
        *,
        frames: list[bytes] | None = None,
        fail_start_times: int = 0,
        fail_urls: set[str] | None = None,
        on_empty: Callable[[], None] | None = None,
    ) -> None:
        self.frame_width = 2
        self.frame_height = 2
        self._frames = list(frames or [])
        self._fail_start_times = int(fail_start_times)
        self._fail_urls = set(fail_urls or set())
        self._on_empty = on_empty
        self.start_calls: list[str] = []
        self._running = False
        self._exit_code: int | None = None
        self.frames_read = 0

    def start(self, rtsp_url: str) -> None:
        self.start_calls.append(rtsp_url)
        if self._fail_start_times > 0 or rtsp_url in self._fail_urls:
            if self._fail_start_times > 0:
                self._fail_start_times -= 1
            self._running = False
            self._exit_code = 1
            raise RuntimeError("start failed")
        self._running = True
        self._exit_code = None

    def stop(self) -> None:
        self._running = False
        self._exit_code = 0

    def read_frame(self, timeout_s: float) -> bytes | None:
        if not self._running:
            return None
        if self._frames:
            self.frames_read += 1
            return self._frames.pop(0)
        if self._on_empty:
            self._on_empty()
        return None

    def is_running(self) -> bool:
        return self._running

    def exit_code(self) -> int | None:
        return self._exit_code


@dataclass(frozen=True)
class DummyProc:
    pid: int
    returncode: int | None = None


class FakeRecorder:
    def __init__(self) -> None:
        self.started: list[Path] = []
        self.stopped: list[DummyProc] = []
        self.dead: set[DummyProc] = set()
        self.fail_start = False
        self.start_calls = 0
        self._pid = 1000

    def start(self, output_file: Path, stderr_log: Path) -> DummyProc | None:
        _ = stderr_log
        self.start_calls += 1
        if self.fail_start:
            return None
        proc = DummyProc(pid=self._pid)
        self._pid += 1
        self.started.append(output_file)
        return proc

    def stop(self, proc: DummyProc, output_file: Path | None) -> None:
        _ = output_file
        self.stopped.append(proc)

    def is_alive(self, proc: DummyProc) -> bool:
        return proc not in self.dead


def _make_config(tmp_path: Path, **overrides: object) -> RTSPSourceConfig:
    data: dict[str, object] = {
        "rtsp_url": "rtsp://host/stream",
        "output_dir": str(tmp_path),
        "stream": {"disable_hwaccel": True},
    }
    for key in ("motion", "recording", "stream", "reconnect", "runtime"):
        if key in overrides:
            data.setdefault(key, {})
            data[key].update(overrides.pop(key))
    data.update(overrides)
    return RTSPSourceConfig.model_validate(data)


def test_rtsp_url_env_overrides_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RTSP URL env var should override explicit config."""
    # Given: a config RTSP URL and a matching env var
    monkeypatch.setenv("RTSP_URL", "rtsp://env/stream")
    config = _make_config(
        tmp_path,
        rtsp_url="rtsp://config/stream",
        rtsp_url_env="RTSP_URL",
    )

    # When: initializing the source
    source = RTSPSource(config, camera_name="cam")

    # Then: the env var value wins
    assert source.rtsp_url == "rtsp://env/stream"


def test_rtsp_url_env_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RTSP source should raise when env var is missing and no URL is provided."""
    # Given: only an env var name with no value set
    monkeypatch.delenv("MISSING_RTSP_URL", raising=False)
    config = _make_config(tmp_path, rtsp_url=None, rtsp_url_env="MISSING_RTSP_URL")

    # When/Then: initialization fails due to missing URL
    with pytest.raises(ValueError, match="rtsp_url_env or rtsp_url required"):
        RTSPSource(config, camera_name="cam")


def test_detect_rtsp_url_env_overrides_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Detect RTSP env var should override explicit detect URL."""
    # Given: a detect RTSP URL and a matching env var
    monkeypatch.setenv("DETECT_URL", "rtsp://env/detect")
    config = _make_config(
        tmp_path,
        rtsp_url="rtsp://host/stream",
        detect_rtsp_url="rtsp://config/detect",
        detect_rtsp_url_env="DETECT_URL",
    )

    # When: initializing the source
    source = RTSPSource(config, camera_name="cam")

    # Then: detect URL comes from the env var
    assert source.detect_rtsp_url == "rtsp://env/detect"
    assert source._detect_rtsp_url_source == "explicit"


def test_detect_rtsp_url_env_missing_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing detect env var should fall back to explicit detect URL."""
    # Given: a detect RTSP URL and a missing env var
    monkeypatch.delenv("DETECT_URL", raising=False)
    config = _make_config(
        tmp_path,
        rtsp_url="rtsp://host/stream?subtype=0",
        detect_rtsp_url="rtsp://config/detect",
        detect_rtsp_url_env="DETECT_URL",
    )

    # When: initializing the source
    source = RTSPSource(config, camera_name="cam")

    # Then: explicit detect URL is used
    assert source.detect_rtsp_url == "rtsp://config/detect"
    assert source._detect_rtsp_url_source == "explicit"


def test_detect_stream_explicit_overrides_derived(tmp_path: Path) -> None:
    """Explicit detect stream should be used even when subtype=0 is present."""
    # Given: an explicit detect stream URL alongside subtype=0
    config = _make_config(
        tmp_path,
        rtsp_url="rtsp://host/stream?subtype=0",
        detect_rtsp_url="rtsp://host/stream?subtype=2",
    )

    # When: initializing the source
    source = RTSPSource(config, camera_name="cam")

    # Then: explicit detect stream is chosen
    assert source.detect_rtsp_url == "rtsp://host/stream?subtype=2"
    assert source._detect_rtsp_url_source == "explicit"
    assert source._detect_stream_available


def test_disable_hwaccel_skips_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabling hwaccel should skip detection and force software decode."""

    # Given: hardware accel disabled
    def _boom(_: str) -> HardwareAccelConfig:
        raise AssertionError("HardwareAccelDetector.detect should not run")

    monkeypatch.setattr("homesec.sources.rtsp.core.HardwareAccelDetector.detect", _boom)
    config = _make_config(tmp_path, stream={"disable_hwaccel": True})

    # When: initializing the source
    source = RTSPSource(config, camera_name="cam")

    # Then: software decode is forced
    assert source.hwaccel_config.hwaccel is None
    assert not source.hwaccel_config.is_available


def test_hwaccel_detection_used_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hardware accel detection should run when not disabled."""

    # Given: a detector that returns a CUDA config
    def _detect(_: str) -> HardwareAccelConfig:
        return HardwareAccelConfig(hwaccel="cuda")

    monkeypatch.setattr("homesec.sources.rtsp.core.HardwareAccelDetector.detect", _detect)
    config = _make_config(tmp_path, stream={"disable_hwaccel": False})

    # When: initializing the source
    source = RTSPSource(config, camera_name="cam")

    # Then: hwaccel config is applied
    assert source.hwaccel_config.hwaccel == "cuda"
    assert source.hwaccel_config.is_available


def test_detect_stream_derived_from_subtype(tmp_path: Path) -> None:
    """Subtype=0 URLs should derive a subtype=1 detect stream."""
    # Given: a subtype=0 main stream with no explicit detect stream
    config = _make_config(tmp_path, rtsp_url="rtsp://host/stream?subtype=0")

    # When: initializing the source
    source = RTSPSource(config, camera_name="cam")

    # Then: detect stream is derived
    assert source.detect_rtsp_url == "rtsp://host/stream?subtype=1"
    assert source._detect_rtsp_url_source == "derived_subtype=1"
    assert source._detect_stream_available


def test_detect_stream_derived_from_stream1(tmp_path: Path) -> None:
    """Trailing /stream1 URLs should derive /stream2 detect streams."""
    # Given: a /stream1 main stream with no explicit detect stream
    config = _make_config(tmp_path, rtsp_url="rtsp://host/camera/stream1")

    # When: initializing the source
    source = RTSPSource(config, camera_name="cam")

    # Then: detect stream is derived from stream sibling
    assert source.detect_rtsp_url == "rtsp://host/camera/stream2"
    assert source._detect_rtsp_url_source == "derived_stream2"
    assert source._detect_stream_available


def test_detect_stream_defaults_to_main(tmp_path: Path) -> None:
    """Detect stream should default to main stream when no subtype is available."""
    # Given: a main stream without subtype and no detect stream
    config = _make_config(tmp_path, rtsp_url="rtsp://host/stream")

    # When: initializing the source
    source = RTSPSource(config, camera_name="cam")

    # Then: detect stream equals main stream
    assert source.detect_rtsp_url == source.rtsp_url
    assert source._detect_rtsp_url_source == "same_as_rtsp_url"
    assert not source._detect_stream_available


def test_reconnect_retries_until_success(tmp_path: Path) -> None:
    """Reconnect should retry until a pipeline starts when attempts are infinite."""
    # Given: a pipeline that fails twice then succeeds
    pipeline = FakeFramePipeline(frames=[b"0000"], fail_start_times=2)
    recorder = FakeRecorder()
    clock = FakeClock()
    config = _make_config(tmp_path, reconnect={"max_attempts": 0})
    source = RTSPSource(
        config,
        camera_name="cam",
        frame_pipeline=pipeline,
        recorder=recorder,
        clock=clock,
    )

    # When: reconnecting aggressively
    ok = source._reconnect_frame_pipeline(aggressive=True)

    # Then: reconnect succeeds after multiple attempts
    assert ok
    assert len(pipeline.start_calls) == 3


def test_reconnect_respects_max_attempts_exact(tmp_path: Path) -> None:
    """Reconnect should attempt exactly max_attempts times."""
    # Given: a pipeline that always fails to start
    pipeline = FakeFramePipeline(fail_start_times=5)
    recorder = FakeRecorder()
    clock = FakeClock()
    config = _make_config(tmp_path, reconnect={"max_attempts": 2})
    source = RTSPSource(
        config,
        camera_name="cam",
        frame_pipeline=pipeline,
        recorder=recorder,
        clock=clock,
    )

    # When: reconnecting aggressively with a finite max
    ok = source._reconnect_frame_pipeline(aggressive=True)

    # Then: reconnect fails after exactly max attempts
    assert not ok
    assert len(pipeline.start_calls) == 2


def test_reconnect_no_frames_exhausts(tmp_path: Path) -> None:
    """Reconnect should stop when no frames arrive and attempts are exhausted."""
    # Given: a frame pipeline that starts but yields no frames
    pipeline = FakeFramePipeline(frames=[])
    recorder = FakeRecorder()
    clock = FakeClock()
    config = _make_config(tmp_path, reconnect={"max_attempts": 1})
    source = RTSPSource(
        config,
        camera_name="cam",
        frame_pipeline=pipeline,
        recorder=recorder,
        clock=clock,
    )

    # When: reconnecting aggressively with no frames available
    ok = source._reconnect_frame_pipeline(aggressive=True)

    # Then: reconnect stops after the single attempt
    assert not ok
    assert pipeline.start_calls == [source.detect_rtsp_url]
    assert not pipeline.is_running()


def test_reconnect_exhausted_returns_true(tmp_path: Path) -> None:
    """Reconnect exhaustion should return true once max attempts are reached."""
    # Given: a source with a finite reconnect limit
    pipeline = FakeFramePipeline()
    recorder = FakeRecorder()
    clock = FakeClock()
    config = _make_config(tmp_path, reconnect={"max_attempts": 2})
    source = RTSPSource(
        config,
        camera_name="cam",
        frame_pipeline=pipeline,
        recorder=recorder,
        clock=clock,
    )

    # When: below the max attempt threshold
    source.reconnect_count = 1
    exhausted_before = source._reconnect_exhausted(aggressive=True)

    # Then: reconnect is not exhausted yet
    assert not exhausted_before

    # When: attempts reach the max
    source.reconnect_count = 2
    exhausted_after = source._reconnect_exhausted(aggressive=True)

    # Then: reconnect is exhausted
    assert exhausted_after


def test_detect_fallback_after_attempts(tmp_path: Path) -> None:
    """Detect fallback should switch to main stream after failures."""
    # Given: a derived detect stream that fails and a main stream that works
    config = _make_config(
        tmp_path,
        rtsp_url="rtsp://host/stream?subtype=0",
        reconnect={"detect_fallback_attempts": 2, "max_attempts": 5},
    )
    detect_url = RTSPSource(config, camera_name="cam").detect_rtsp_url
    pipeline = FakeFramePipeline(frames=[b"0000"], fail_urls={detect_url})
    recorder = FakeRecorder()
    clock = FakeClock()
    source = RTSPSource(
        config,
        camera_name="cam",
        frame_pipeline=pipeline,
        recorder=recorder,
        clock=clock,
    )

    # When: reconnecting with repeated detect failures
    ok = source._reconnect_frame_pipeline(aggressive=True)

    # Then: reconnect succeeds using the main stream fallback
    assert ok
    assert pipeline.start_calls[:2] == [detect_url, detect_url]
    assert pipeline.start_calls[-1] == source.rtsp_url
    assert source._detect_fallback_active


def test_reconnect_defers_detect_fallback_while_recording(tmp_path: Path) -> None:
    """Reconnect should defer detect fallback when recording is active."""
    # Given: detect stream always fails while recording is active
    config = _make_config(
        tmp_path,
        rtsp_url="rtsp://host/stream?subtype=0",
        reconnect={"detect_fallback_attempts": 1, "max_attempts": 1},
    )
    detect_url = RTSPSource(config, camera_name="cam").detect_rtsp_url
    pipeline = FakeFramePipeline(fail_urls={detect_url})
    recorder = FakeRecorder()
    clock = FakeClock()
    source = RTSPSource(
        config,
        camera_name="cam",
        frame_pipeline=pipeline,
        recorder=recorder,
        clock=clock,
    )
    source.recording_process = DummyProc(pid=1, returncode=None)

    # When: reconnecting hits detect failures
    ok = source._reconnect_frame_pipeline(aggressive=True)

    # Then: reconnect fails, fallback is deferred, and detect URL is retained
    assert not ok
    assert source._detect_fallback_deferred
    assert not source._detect_fallback_active
    assert source._motion_rtsp_url == detect_url


def test_detect_fallback_deferred_while_recording(tmp_path: Path) -> None:
    """Detect fallback should defer while recording is active."""
    # Given: detect stream failures during an active recording
    config = _make_config(
        tmp_path,
        rtsp_url="rtsp://host/stream?subtype=0",
        reconnect={"detect_fallback_attempts": 1},
    )
    source = RTSPSource(config, camera_name="cam")
    source.recording_process = DummyProc(pid=7, returncode=None)

    # When: a detect failure is noted
    triggered = source._note_detect_failure(0.0)

    # Then: fallback is deferred
    assert not triggered
    assert not source._detect_fallback_active
    assert source._detect_failure_count == source.detect_fallback_attempts


def test_detect_fallback_activates_after_recording_stops(tmp_path: Path) -> None:
    """Detect fallback should activate once recording stops."""
    # Given: a deferred detect fallback while recording
    config = _make_config(
        tmp_path,
        rtsp_url="rtsp://host/stream?subtype=0",
        reconnect={"detect_fallback_attempts": 1},
    )
    source = RTSPSource(config, camera_name="cam")
    source.recording_process = DummyProc(pid=9, returncode=None)
    _ = source._note_detect_failure(0.0)
    source.recording_process = None

    # When: another detect failure occurs after recording stops
    triggered = source._note_detect_failure(1.0)

    # Then: fallback activates to the main stream
    assert triggered
    assert source._detect_fallback_active
    assert source._motion_rtsp_url == source.rtsp_url


def test_frame_pipeline_retries_without_timeouts_when_unsupported(tmp_path: Path) -> None:
    """Frame pipeline should retry without timeout options when unsupported."""
    # Given: a frame pipeline with timeouts enabled
    clock = FakeClock()
    pipeline = FfmpegFramePipeline(
        output_dir=tmp_path,
        frame_queue_size=5,
        rtsp_connect_timeout_s=2.0,
        rtsp_io_timeout_s=2.0,
        ffmpeg_flags=[],
        motion_profile=None,
        hwaccel_config=HardwareAccelConfig(hwaccel=None),
        hwaccel_failed=True,
        on_frame=lambda: None,
        clock=clock,
    )

    attempts: list[list[str]] = []

    class FakeProc:
        def __init__(self, returncode: int | None) -> None:
            self.returncode = returncode

        def poll(self) -> int | None:
            return self.returncode

    def fake_popen(cmd: list[str], stdout=None, stderr=None, bufsize=None):  # type: ignore[no-untyped-def]
        attempts.append(cmd)
        if len(attempts) == 1:
            if stderr is not None:
                stderr.write("Option rw_timeout not found.\n")
                stderr.flush()
            return FakeProc(returncode=1)
        return FakeProc(returncode=None)

    # When: starting the frame pipeline
    with patch("homesec.sources.rtsp.frame_pipeline.subprocess.Popen", side_effect=fake_popen):
        proc, stderr_handle, _width, _height = pipeline._get_frame_pipe("rtsp://host/stream")

    # Then: it retries without timeout options and succeeds
    assert proc.poll() is None
    assert len(attempts) == 2
    assert "-rw_timeout" in attempts[0]
    assert "-rw_timeout" not in attempts[1]
    stderr_handle.close()


def test_recording_threshold_is_more_sensitive(tmp_path: Path) -> None:
    """Recording threshold should be lower than idle threshold."""
    # Given: a source with a 2x recording sensitivity factor
    config = _make_config(
        tmp_path,
        motion={
            "min_changed_pct": 30.0,
            "recording_sensitivity_factor": 2.0,
            "pixel_threshold": 1,
            "blur_kernel": 0,
        },
    )
    idle_source = RTSPSource(config, camera_name="cam")
    recording_source = RTSPSource(config, camera_name="cam")
    frame_a = np.zeros((2, 2), dtype=np.uint8)
    frame_b = frame_a.copy()
    frame_b[0, 0] = 255

    # When: detecting motion while idle (threshold 30%)
    _ = idle_source.detect_motion(frame_a, threshold=idle_source.min_changed_pct)
    idle_motion = idle_source.detect_motion(frame_b, threshold=idle_source.min_changed_pct)

    # Then: idle detection does not trigger (25% < 30%)
    assert not idle_motion

    # When: detecting motion while recording (threshold 15%)
    recording_threshold = recording_source._recording_threshold()
    _ = recording_source.detect_motion(frame_a, threshold=recording_threshold)
    recording_motion = recording_source.detect_motion(frame_b, threshold=recording_threshold)

    # Then: recording detection triggers (25% >= 15%)
    assert recording_motion


def test_stop_delay_resets_on_motion(tmp_path: Path) -> None:
    """Motion should reset the stop-delay countdown while recording."""
    # Given: an active recording with a stop_delay of 10s
    recorder = FakeRecorder()
    clock = FakeClock()
    config = _make_config(tmp_path, recording={"stop_delay": 10.0})
    source = RTSPSource(config, camera_name="cam", recorder=recorder, clock=clock)
    proc = DummyProc(pid=1, returncode=None)
    source.recording_process = proc
    source.last_motion_time = 0.0

    # When: motion is detected again at t=9s
    source._handle_motion_detected(9.0, frame_count=1)

    # Then: the stop-delay window is reset
    assert source.last_motion_time == 9.0

    # When: updating state shortly after (t=10.5s)
    source._update_recording_state(10.5)

    # Then: recording is still active
    assert source.recording_process is proc

    # When: updating state after the new stop-delay expires (t=19.1s)
    source._update_recording_state(19.1)

    # Then: recording stops
    assert source.recording_process is None
    assert source.last_motion_time is None


def test_recording_rotates_after_max_duration(tmp_path: Path) -> None:
    """Recording should rotate when max duration is exceeded during motion."""
    # Given: a recording that has exceeded max_recording_s with recent motion
    recorder = FakeRecorder()
    clock = FakeClock()
    config = _make_config(tmp_path, recording={"max_recording_s": 5.0, "stop_delay": 10.0})
    source = RTSPSource(config, camera_name="cam", recorder=recorder, clock=clock)
    old_proc = DummyProc(pid=42, returncode=None)
    old_output = tmp_path / "old_motion.mp4"
    old_log = tmp_path / "old_recording.log"
    source._set_recording_state(
        proc=old_proc,
        output_file=old_output,
        stderr_log=old_log,
        start_mono=0.0,
        start_wall=datetime.now(),
    )
    source.last_motion_time = 4.0
    clock.sleep(6.0)

    # When: checking for rotation at t=6s
    source._rotate_recording_if_needed()

    # Then: a new recording is started and the old one is stopped
    assert recorder.started
    assert recorder.stopped == [old_proc]
    assert source.recording_process is not old_proc
    assert source.recording_start_time == clock.now()
    assert source.last_motion_time == clock.now()


def test_stall_grace_applies_while_recording(tmp_path: Path) -> None:
    """Recording should enter stall grace window when frames are missing."""
    # Given: an active recording with recent motion
    pipeline = FakeFramePipeline(frames=[])
    recorder = FakeRecorder()
    clock = FakeClock()
    config = _make_config(tmp_path, recording={"stop_delay": 10.0})
    source = RTSPSource(
        config,
        camera_name="cam",
        frame_pipeline=pipeline,
        recorder=recorder,
        clock=clock,
    )
    source.recording_process = DummyProc(pid=1, returncode=None)
    start_now = clock.now()
    source.last_motion_time = start_now

    # When: a missing frame triggers reconnect handling
    with patch.object(source, "_handle_frame_timeout", return_value=False):
        ok = source._handle_missing_frame(start_now)

    # Then: stall grace is set and we enter STALLED state
    assert ok
    assert source._stall_grace_until == start_now + source.motion_stop_delay
    assert source._run_state == RTSPRunState.STALLED


def test_recording_restarts_when_dead(tmp_path: Path) -> None:
    """Recording should restart immediately if it dies during recent motion."""
    # Given: a recording process marked dead with recent motion
    recorder = FakeRecorder()
    pipeline = FakeFramePipeline(frames=[b"0000"])
    clock = FakeClock()
    config = _make_config(tmp_path)
    source = RTSPSource(
        config,
        camera_name="cam",
        frame_pipeline=pipeline,
        recorder=recorder,
        clock=clock,
    )
    dead_proc = DummyProc(pid=42, returncode=1)
    recorder.dead.add(dead_proc)
    source.recording_process = dead_proc
    source.last_motion_time = 0.0

    # When: ensuring recording at t=5s
    source._ensure_recording(5.0)

    # Then: a new recording is started
    assert source.recording_process is not dead_proc
    assert len(recorder.started) == 1


def test_recording_start_backoff_throttles_retries(tmp_path: Path) -> None:
    """Recording start should respect backoff when failures occur."""
    # Given: a recorder that fails to start
    recorder = FakeRecorder()
    recorder.fail_start = True
    clock = FakeClock()
    config = _make_config(tmp_path)
    source = RTSPSource(config, camera_name="cam", recorder=recorder, clock=clock)

    # When: start is called twice without advancing time
    source.start_recording()
    source.start_recording()

    # Then: only one attempt is made
    assert recorder.start_calls == 1

    # When: time advances past the backoff window
    clock.sleep(0.6)
    source.start_recording()

    # Then: another attempt is made (still failing)
    assert recorder.start_calls == 2


def test_recording_retries_without_timeouts_when_unsupported(tmp_path: Path) -> None:
    """Recording should retry without timeout flags when ffmpeg rejects them."""
    # Given: a recorder and a fake ffmpeg that rejects timeout flags
    clock = FakeClock()
    recorder = FfmpegRecorder(
        rtsp_url="rtsp://host/stream",
        ffmpeg_flags=[],
        rtsp_connect_timeout_s=1.0,
        rtsp_io_timeout_s=1.0,
        clock=clock,
    )
    output_file = tmp_path / "clip.mp4"
    stderr_log = tmp_path / "clip.log"
    calls: list[list[str]] = []

    class DummyPopen:
        def __init__(self, returncode: int | None) -> None:
            self._returncode = returncode
            self.returncode = returncode
            self.pid = 1234

        def poll(self) -> int | None:
            return self._returncode

    def fake_popen(cmd: list[str], **kwargs: object) -> DummyPopen:
        calls.append(list(cmd))
        stderr = kwargs.get("stderr")
        if "-rw_timeout" in cmd or "-stimeout" in cmd:
            if hasattr(stderr, "write"):
                stderr.write("Option rw_timeout not found.\n")
                stderr.flush()
            return DummyPopen(returncode=1)
        return DummyPopen(returncode=None)

    # When: starting a recording
    with patch("homesec.sources.rtsp.recorder.subprocess.Popen", side_effect=fake_popen):
        proc = recorder.start(output_file, stderr_log)

    # Then: fallback removes timeout flags and succeeds
    assert proc is not None
    assert len(calls) == 2
    assert "-rw_timeout" in calls[0] or "-stimeout" in calls[0]
    assert "-rw_timeout" not in calls[1]


def test_probe_retries_without_timeouts_when_unsupported(tmp_path: Path) -> None:
    """Stream probe should retry without timeouts when ffprobe rejects them."""
    # Given: an RTSP source and a fake ffprobe error on timeout options
    config = _make_config(tmp_path, rtsp_url="rtsp://host/stream")
    source = RTSPSource(config, camera_name="cam")
    calls: list[list[str]] = []

    class FakeResult:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(cmd: list[str], **kwargs: object) -> FakeResult:
        _ = kwargs
        calls.append(list(cmd))
        if "-rw_timeout" in cmd or "-stimeout" in cmd:
            return FakeResult(returncode=1, stderr="Option rw_timeout not found")
        return FakeResult(
            returncode=0,
            stdout='{"streams":[{"codec_name":"h264","width":640,"height":480,"avg_frame_rate":"30/1"}]}',
        )

    # When: probing the stream info
    with patch("homesec.sources.rtsp.core.subprocess.run", side_effect=fake_run):
        info = source._probe_stream_info(source.rtsp_url)

    # Then: probe retries without timeouts and succeeds
    assert info is not None
    assert info["width"] == 640
    assert len(calls) == 2
    assert "-rw_timeout" in calls[0] or "-stimeout" in calls[0]
    assert "-rw_timeout" not in calls[1]


def test_recording_survives_short_stall(tmp_path: Path) -> None:
    """Recording should remain active during brief frame stalls."""

    class TestRTSPSource(RTSPSource):
        def _log_startup_info(self) -> None:
            return

        def cleanup(self) -> None:
            return

    # Given: a pipeline that detects motion then stalls briefly
    frame_a = np.zeros((2, 2), dtype=np.uint8)
    frame_b = frame_a.copy()
    frame_b[0, 0] = 255
    pipeline = FakeFramePipeline(frames=[frame_a.tobytes(), frame_b.tobytes()])
    recorder = FakeRecorder()
    clock = FakeClock()
    config = _make_config(
        tmp_path,
        motion={"blur_kernel": 0, "min_changed_pct": 1.0},
        recording={"stop_delay": 10.0},
        runtime={"frame_timeout_s": 0.1},
        reconnect={"max_attempts": 1},
    )
    source = TestRTSPSource(
        config,
        camera_name="cam",
        frame_pipeline=pipeline,
        recorder=recorder,
        clock=clock,
    )
    pipeline._on_empty = source._stop_event.set
    source._stop_event.clear()

    # When: running until the first stall
    source._run()

    # Then: recording is still active and never stopped
    assert recorder.started
    assert not recorder.stopped
    assert source.recording_process is not None
    assert pipeline.frames_read == 2


def test_detect_stream_recovers_after_probe(tmp_path: Path) -> None:
    """Detect fallback should restore the detect stream after a successful probe."""
    # Given: fallback active with a successful detect probe
    config = _make_config(
        tmp_path,
        rtsp_url="rtsp://host/stream?subtype=0",
        reconnect={"detect_fallback_attempts": 1},
    )
    pipeline = FakeFramePipeline(frames=[b"0000"])
    recorder = FakeRecorder()
    clock = FakeClock()
    source = RTSPSource(
        config,
        camera_name="cam",
        frame_pipeline=pipeline,
        recorder=recorder,
        clock=clock,
    )
    now = clock.now()
    source._activate_detect_fallback(now)
    source._detect_next_probe_at = now

    def _probe(rtsp_url: str, *, timeout_s: float = 10.0) -> dict[str, object] | None:
        _ = rtsp_url
        _ = timeout_s
        return {"width": 640, "height": 480, "avg_frame_rate": "30/1"}

    source._probe_stream_info = _probe  # type: ignore[assignment]

    # When: attempting to recover the detect stream
    ok = source._maybe_recover_detect_stream(now)

    # Then: detect stream is restored
    assert ok
    assert not source._detect_fallback_active
    assert source._motion_rtsp_url == source.detect_rtsp_url
    assert pipeline.start_calls[-1] == source.detect_rtsp_url


def test_detect_probe_schedules_backoff(tmp_path: Path) -> None:
    """Failed detect probe should schedule a backoff before retry."""
    # Given: detect fallback active and a failing probe
    config = _make_config(tmp_path, rtsp_url="rtsp://host/stream?subtype=0")
    source = RTSPSource(config, camera_name="cam")
    now = 0.0
    source._activate_detect_fallback(now)
    source._detect_next_probe_at = 0.0

    def _probe(_: str, *, timeout_s: float = 3.0) -> dict[str, object] | None:
        _ = timeout_s
        return None

    source._probe_stream_info = _probe  # type: ignore[assignment]

    # When: attempting to recover the detect stream
    ok = source._maybe_recover_detect_stream(now)

    # Then: probe is scheduled with backoff and recovery fails
    assert not ok
    assert source._detect_next_probe_at == now + 25.0
    assert source._detect_probe_backoff_s == 40.0


def test_detect_switch_failure_falls_back(tmp_path: Path) -> None:
    """Detect switch failure should fall back to main stream and reschedule probes."""
    # Given: detect fallback active and a successful probe result
    config = _make_config(tmp_path, rtsp_url="rtsp://host/stream?subtype=0")
    source = RTSPSource(config, camera_name="cam")
    now = 0.0
    source._activate_detect_fallback(now)
    source._detect_next_probe_at = now

    def _probe(_: str, *, timeout_s: float = 3.0) -> dict[str, object] | None:
        _ = timeout_s
        return {"width": 640, "height": 480}

    def _fail_start() -> None:
        raise RuntimeError("switch failed")

    source._probe_stream_info = _probe  # type: ignore[assignment]
    source._start_frame_pipeline = _fail_start  # type: ignore[assignment]

    # When: attempting to recover the detect stream
    ok = source._maybe_recover_detect_stream(now)

    # Then: fallback remains active and a new probe is scheduled
    assert not ok
    assert source._detect_fallback_active
    assert source._motion_rtsp_url == source.rtsp_url
    assert source._detect_next_probe_at == now + 25.0
    assert source._detect_probe_backoff_s == 40.0


def test_probe_stream_info_returns_none_on_error(tmp_path: Path) -> None:
    """ffprobe errors should return None and stop retries."""
    # Given: a source and an ffprobe failure without timeout option errors
    config = _make_config(tmp_path)
    source = RTSPSource(config, camera_name="cam")

    def _fake_run(*_args: object, **_kwargs: object) -> object:
        class Result:
            returncode = 1
            stdout = ""
            stderr = "Some other ffprobe error"

        return Result()

    # When: probing stream info
    with patch("homesec.sources.rtsp.core.subprocess.run", side_effect=_fake_run):
        info = source._probe_stream_info(source.rtsp_url)

    # Then: probe fails and returns None
    assert info is None


def test_probe_stream_info_handles_exception(tmp_path: Path) -> None:
    """ffprobe exceptions should return None."""
    # Given: a source and an ffprobe exception
    config = _make_config(tmp_path)
    source = RTSPSource(config, camera_name="cam")

    # When: probing stream info raises
    with patch(
        "homesec.sources.rtsp.core.subprocess.run",
        side_effect=RuntimeError("ffprobe crashed"),
    ):
        info = source._probe_stream_info(source.rtsp_url)

    # Then: probe fails and returns None
    assert info is None
