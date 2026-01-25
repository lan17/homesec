"""Hermetic tests for RTSPSource reconnect and recording behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from homesec.models.source import RTSPSourceConfig
from homesec.sources.rtsp import RTSPSource


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
    ) -> None:
        self.frame_width = 2
        self.frame_height = 2
        self._frames = list(frames or [])
        self._fail_start_times = int(fail_start_times)
        self._fail_urls = set(fail_urls or set())
        self.start_calls: list[str] = []
        self._running = False
        self._exit_code: int | None = None

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
            return self._frames.pop(0)
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
        self._pid = 1000

    def start(self, output_file: Path, stderr_log: Path) -> DummyProc:
        _ = stderr_log
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
        "disable_hwaccel": True,
    }
    data.update(overrides)
    return RTSPSourceConfig.model_validate(data)


def test_reconnect_retries_until_success(tmp_path: Path) -> None:
    """Reconnect should retry until a pipeline starts when attempts are infinite."""
    # Given: a pipeline that fails twice then succeeds
    pipeline = FakeFramePipeline(frames=[b"0000"], fail_start_times=2)
    recorder = FakeRecorder()
    clock = FakeClock()
    config = _make_config(tmp_path, max_reconnect_attempts=0)
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


def test_detect_fallback_after_attempts(tmp_path: Path) -> None:
    """Detect fallback should switch to main stream after failures."""
    # Given: a derived detect stream that fails and a main stream that works
    config = _make_config(
        tmp_path,
        rtsp_url="rtsp://host/stream?subtype=0",
        detect_fallback_attempts=2,
        max_reconnect_attempts=5,
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


def test_recording_threshold_is_more_sensitive(tmp_path: Path) -> None:
    """Recording threshold should be lower than idle threshold."""
    # Given: a source with a 2x recording sensitivity factor
    config = _make_config(
        tmp_path,
        min_changed_pct=30.0,
        recording_sensitivity_factor=2.0,
        pixel_threshold=1,
        blur_kernel=0,
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
    recording_threshold = recording_source._keepalive_threshold()
    _ = recording_source.detect_motion(frame_a, threshold=recording_threshold)
    recording_motion = recording_source.detect_motion(frame_b, threshold=recording_threshold)

    # Then: recording detection triggers (25% >= 15%)
    assert recording_motion


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
