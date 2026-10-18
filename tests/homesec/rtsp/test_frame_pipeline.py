"""Tests for RTSP frame pipeline behavior."""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path
from threading import Thread

import pytest

from homesec.sources.rtsp.frame_pipeline import FfmpegFramePipeline
from homesec.sources.rtsp.hardware import HardwareAccelConfig
from homesec.sources.rtsp.recording_profile import MotionProfile


class FakeClock:
    def __init__(self) -> None:
        self._now = 0.0

    def now(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self._now += float(seconds)


class FakeStdout:
    def __init__(self, frames: list[bytes]) -> None:
        self._frames = list(frames)

    def read(self, _size: int) -> bytes:
        if self._frames:
            return self._frames.pop(0)
        return b""


class ErrorStdout:
    def read(self, _size: int) -> bytes:
        raise RuntimeError("read failed")


class FakeProcess:
    def __init__(self, stdout: object) -> None:
        self.stdout = stdout
        self.pid = 1234
        self._returncode: int | None = None

    def poll(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self._returncode = 0

    def wait(self, timeout: float | None = None) -> int | None:
        _ = timeout
        return self._returncode

    def kill(self) -> None:
        self._returncode = -9


def _make_pipeline(tmp_path: Path, *, frame_queue_size: int) -> FfmpegFramePipeline:
    return FfmpegFramePipeline(
        output_dir=tmp_path,
        frame_queue_size=frame_queue_size,
        rtsp_connect_timeout_s=1.0,
        rtsp_io_timeout_s=1.0,
        ffmpeg_flags=[],
        motion_profile=MotionProfile(input_url="rtsp://host/stream"),
        hwaccel_config=HardwareAccelConfig(hwaccel=None),
        hwaccel_failed=True,
        on_frame=lambda: None,
        clock=FakeClock(),
    )


def test_frame_pipeline_drops_oldest_when_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frame pipeline should drop oldest frame when queue is full."""
    # Given: a pipeline with a single-slot queue and two frames
    pipeline = _make_pipeline(tmp_path, frame_queue_size=1)
    frame_a = b"A" * 4
    frame_b = b"B" * 4
    fake_process = FakeProcess(FakeStdout([frame_a, frame_b, b""]))

    def _fake_get_frame_pipe(_rtsp_url: str) -> tuple[FakeProcess, io.StringIO, int, int]:
        return fake_process, io.StringIO(), 2, 2

    monkeypatch.setattr(pipeline, "_get_frame_pipe", _fake_get_frame_pipe)

    # When: starting the pipeline and reading frames
    pipeline.start("rtsp://host/stream")
    if pipeline._reader_thread is not None:
        pipeline._reader_thread.join(timeout=1)

    # Then: only the newest frame remains
    assert pipeline.read_frame(timeout_s=0.1) == frame_b
    assert pipeline.read_frame(timeout_s=0.1) is None
    pipeline.stop()


def test_frame_pipeline_reader_error_stops_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frame pipeline should stop reading when stdout errors."""
    # Given: a pipeline whose stdout read raises an exception
    pipeline = _make_pipeline(tmp_path, frame_queue_size=1)
    fake_process = FakeProcess(ErrorStdout())

    def _fake_get_frame_pipe(_rtsp_url: str) -> tuple[FakeProcess, io.StringIO, int, int]:
        return fake_process, io.StringIO(), 2, 2

    monkeypatch.setattr(pipeline, "_get_frame_pipe", _fake_get_frame_pipe)

    # When: starting the pipeline
    pipeline.start("rtsp://host/stream")
    if pipeline._reader_thread is not None:
        pipeline._reader_thread.join(timeout=1)

    # Then: no frames are available
    assert pipeline.read_frame(timeout_s=0.1) is None
    pipeline.stop()


@pytest.mark.parametrize("hardware_mode", ["reject", "stall"])
def test_compatible_hardware_failure_retries_software_once_and_remembers(
    hardware_mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: Native-selected hardware fails in the separate compatible executor,
    # with real child processes at the subprocess boundary and usable software frames.
    launches: list[tuple[bool, subprocess.Popen[bytes]]] = []
    failures: list[str] = []
    real_popen = subprocess.Popen

    def launch(command: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
        hardware = "-hwaccel" in command
        script = "import sys,time; "
        if hardware and hardware_mode == "reject":
            script += "sys.exit(1)"
        elif hardware:
            script += "time.sleep(10)"
        else:
            script += (
                "sys.stdout.buffer.write(b'F'*(320*240)); sys.stdout.buffer.flush(); time.sleep(10)"
            )
        process = real_popen([sys.executable, "-c", script], **kwargs)
        launches.append((hardware, process))
        return process

    monkeypatch.setattr("homesec.sources.rtsp.frame_pipeline.subprocess.Popen", launch)
    pipeline = FfmpegFramePipeline(
        output_dir=tmp_path,
        frame_queue_size=2,
        rtsp_connect_timeout_s=1,
        rtsp_io_timeout_s=1,
        ffmpeg_flags=[],
        motion_profile=MotionProfile(input_url="rtsp://host/stream"),
        hwaccel_config=HardwareAccelConfig(hwaccel="videotoolbox"),
        hwaccel_failed=False,
        on_frame=lambda: None,
        clock=FakeClock(),
        startup_timeout_s=0.1,
        on_hwaccel_failure=lambda: failures.append("hardware_failed"),
    )
    try:
        # When: Starting the input, then reconnecting in the same camera worker.
        pipeline.start("rtsp://host/stream")
        first_frame = pipeline.read_frame(1)
        assert launches[0][1].poll() is not None
        pipeline.stop()
        pipeline.start("rtsp://host/stream")
        next_frame = pipeline.read_frame(1)

        # Then: Both attempts get software media, hardware is tried just once,
        # and the previous input is reaped before opening its replacement.
        assert first_frame == next_frame == b"F" * (320 * 240)
        assert [hardware for hardware, _ in launches] == [True, False, False]
        assert failures == ["hardware_failed"]
    finally:
        pipeline.stop()
    assert all(process.poll() is not None for _, process in launches)


def test_compatible_hardware_and_software_failure_do_not_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: Both native-selected hardware and compatible software produce no frames.
    launches: list[tuple[bool, subprocess.Popen[bytes]]] = []
    real_popen = subprocess.Popen

    def launch(command: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
        process = real_popen([sys.executable, "-c", "import time; time.sleep(10)"], **kwargs)
        launches.append(("-hwaccel" in command, process))
        return process

    monkeypatch.setattr("homesec.sources.rtsp.frame_pipeline.subprocess.Popen", launch)
    pipeline = FfmpegFramePipeline(
        output_dir=tmp_path,
        frame_queue_size=2,
        rtsp_connect_timeout_s=1,
        rtsp_io_timeout_s=1,
        ffmpeg_flags=[],
        motion_profile=MotionProfile(input_url="rtsp://host/stream"),
        hwaccel_config=HardwareAccelConfig(hwaccel="videotoolbox"),
        hwaccel_failed=False,
        on_frame=lambda: None,
        clock=FakeClock(),
        startup_timeout_s=0.05,
    )
    try:
        # When: No usable frame arrives on either input, including a later retry.
        pipeline.start("rtsp://host/stream")
        assert pipeline.read_frame(0.05) is None
        pipeline.stop()
        pipeline.start("rtsp://host/stream")
        assert pipeline.read_frame(0.05) is None

        # Then: Existing source retry/exhaustion receives no frame; hardware never cycles back.
        assert [hardware for hardware, _ in launches] == [True, False, False]
    finally:
        pipeline.stop()
    assert all(process.poll() is not None for _, process in launches)


def test_compatible_hardware_readiness_preserves_burst_frame_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: A real compatible executor supplies a complete burst before startup
    # returns, with enough queue space to retain every frame in its original order.
    launches: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen

    class BurstReaderThread(Thread):
        def start(self) -> None:
            super().start()
            self.join(timeout=2)
            assert not self.is_alive(), "reader did not drain the finite burst"

    def launch(_command: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
        process = real_popen(
            [
                sys.executable,
                "-c",
                "import os,sys,time; "
                "sys.stdout.buffer.write(b''.join(bytes([value])*(320*240) for value in range(65,73))); "
                "sys.stdout.buffer.flush(); os.close(1); time.sleep(10)",
            ],
            **kwargs,
        )
        launches.append(process)
        return process

    monkeypatch.setattr("homesec.sources.rtsp.frame_pipeline.subprocess.Popen", launch)
    monkeypatch.setattr("homesec.sources.rtsp.frame_pipeline.Thread", BurstReaderThread)
    pipeline = FfmpegFramePipeline(
        output_dir=tmp_path,
        frame_queue_size=8,
        rtsp_connect_timeout_s=1,
        rtsp_io_timeout_s=1,
        ffmpeg_flags=[],
        motion_profile=MotionProfile(input_url="rtsp://host/stream"),
        hwaccel_config=HardwareAccelConfig(hwaccel="videotoolbox"),
        hwaccel_failed=False,
        on_frame=lambda: None,
        clock=FakeClock(),
        startup_timeout_s=0.1,
    )
    try:
        # When: Hardware readiness observes complete frames and the caller consumes them.
        pipeline.start("rtsp://host/stream")
        frames = [pipeline.read_frame(0.1) for _ in range(8)]
        # Then: Readiness neither consumes nor requeues a frame behind its successors.
        assert frames == [bytes([value]) * (320 * 240) for value in range(65, 73)]
        assert len(launches) == 1
    finally:
        pipeline.stop()
    assert launches[0].poll() is not None


def test_unreaped_hardware_input_blocks_replacement_until_cleanup_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: A subprocess boundary whose first hardware input cannot initially be reaped.
    class UnreapableProcess(FakeProcess):
        reapable = False

        def terminate(self) -> None:
            if self.reapable:
                super().terminate()

        def kill(self) -> None:
            if self.reapable:
                super().kill()

        def wait(self, timeout: float | None = None) -> int | None:
            if not self.reapable:
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            return super().wait(timeout)

    stuck = UnreapableProcess(FakeStdout([]))
    launches: list[bool] = []
    failures: list[str] = []

    def launch(command: list[str], **_kwargs: object) -> FakeProcess:
        launches.append("-hwaccel" in command)
        if len(launches) == 1:
            return stuck
        return FakeProcess(FakeStdout([b"F" * (320 * 240)]))

    monkeypatch.setattr("homesec.sources.rtsp.frame_pipeline.subprocess.Popen", launch)
    monkeypatch.setattr(
        "homesec.sources.rtsp.frame_pipeline._signal_process_group", lambda *_args: False
    )
    pipeline = FfmpegFramePipeline(
        output_dir=tmp_path,
        frame_queue_size=2,
        rtsp_connect_timeout_s=1,
        rtsp_io_timeout_s=1,
        ffmpeg_flags=[],
        motion_profile=MotionProfile(input_url="rtsp://host/stream"),
        hwaccel_config=HardwareAccelConfig(hwaccel="videotoolbox"),
        hwaccel_failed=False,
        on_frame=lambda: None,
        clock=FakeClock(),
        startup_timeout_s=0.01,
        on_hwaccel_failure=lambda: failures.append("hardware_failed"),
    )

    # When: The failed hardware fallback cannot stop its input and source retry starts again.
    with pytest.raises(RuntimeError, match="Failed to stop"):
        pipeline.start("rtsp://host/stream")
    with pytest.raises(RuntimeError, match="still owned"):
        pipeline.start("rtsp://host/stream")
    with pytest.raises(RuntimeError, match="Failed to stop"):
        pipeline.stop()

    # Then: No replacement opens while the live process is owned; a later successful
    # stop can still reap that same process before a software retry becomes possible.
    assert launches == [True]
    assert failures == ["hardware_failed"]
    assert pipeline.is_running()
    stuck.reapable = True
    pipeline.stop()
    assert stuck.poll() is not None
    assert not pipeline.is_running()
    try:
        pipeline.start("rtsp://host/stream")
        assert pipeline.read_frame(1) == b"F" * (320 * 240)
        assert launches == [True, False]
    finally:
        pipeline.stop()
