"""Tests for RTSP frame pipeline behavior."""

from __future__ import annotations

import io
from pathlib import Path

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
