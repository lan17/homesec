"""Motion-input behavior through the existing RTSP source and clip callback."""

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import pytest

from homesec.models.clip import Clip
from homesec.sources.rtsp.core import RTSPSource, RTSPSourceConfig
from homesec.sources.rtsp.frame_pipeline import FfmpegFramePipeline
from homesec.sources.rtsp.motion_input import MotionInput, MotionObservation
from homesec.sources.rtsp.preflight import (
    CameraPreflightDiagnostics,
    CameraPreflightOutcome,
    RTSPStartupPreflight,
)
from homesec.sources.rtsp.recording_profile import MotionProfile, build_default_recording_profile


class ControlledClock:
    def __init__(self) -> None:
        self.time = 1.0

    def now(self) -> float:
        return self.time

    def sleep(self, seconds: float) -> None:
        self.time += seconds


@dataclass(frozen=True)
class FrameStep:
    pixels: bytes
    advance_s: float = 0.1


class ScriptedFrames:
    frame_width: int | None = 2
    frame_height: int | None = 2

    def __init__(self, clock: ControlledClock, sessions: list[list[FrameStep] | None]) -> None:
        self.clock = clock
        self.sessions = deque(sessions)
        self.frames: deque[FrameStep] = deque()
        self.start_calls: list[str] = []
        self.running = False
        self.exhausted = False
        self.finished = Event()

    def start(self, rtsp_url: str) -> None:
        self.start_calls.append(rtsp_url)
        if not self.sessions:
            raise RuntimeError("synthetic stream exhausted")
        session = self.sessions.popleft()
        if session is None:
            raise RuntimeError("synthetic camera unavailable")
        self.frames = deque(session)
        self.running = True

    def stop(self) -> None:
        self.running = False
        if self.exhausted and not self.sessions:
            self.finished.set()

    def read_frame(self, timeout_s: float) -> bytes | None:
        if self.frames:
            frame = self.frames.popleft()
            self.clock.sleep(frame.advance_s)
            return frame.pixels
        self.clock.sleep(max(timeout_s, 100.0))
        self.exhausted = True
        return None

    def is_running(self) -> bool:
        return self.running

    def exit_code(self) -> int | None:
        return None if self.running else 0


@dataclass(frozen=True)
class RecordingProcess:
    pid: int = 1


class RecordingBoundary:
    def __init__(self, clock: ControlledClock, *, refuse_start: bool = False) -> None:
        self.clock = clock
        self.refuse_start = refuse_start
        self.calls: list[tuple[str, float]] = []

    def start(self, output_file: Path, stderr_log: Path) -> RecordingProcess | None:
        self.calls.append(("start", self.clock.now()))
        if self.refuse_start:
            return None
        output_file.write_bytes(b"synthetic completed recording")
        return RecordingProcess()

    def stop(self, process: RecordingProcess, output_file: Path | None) -> None:
        self.calls.append(("stop", self.clock.now()))

    def is_alive(self, process: RecordingProcess) -> bool:
        return True


class ScriptedObservations:
    """Native-input boundary that delivers observations without any pixel interface."""

    def __init__(self, steps: list[tuple[MotionObservation, float]]) -> None:
        self.clock = ControlledClock()
        self.steps = deque(steps)
        self.thresholds: list[float] = []
        self.calls: list[tuple[str, str | None]] = []
        self.on_frame: Callable[[], None] = lambda: None
        self.running = False
        self.started = False
        self.exhausted = False
        self.finished = Event()

    def start(self, rtsp_url: str) -> None:
        self.calls.append(("start", rtsp_url))
        if self.started:
            raise RuntimeError("synthetic observation stream exhausted")
        self.started = True
        self.running = True

    def stop(self) -> None:
        self.calls.append(("stop", None))
        self.running = False
        if self.exhausted:
            self.finished.set()

    def is_running(self) -> bool:
        return self.running

    def exit_code(self) -> int | None:
        return None if self.running else 0

    def read_motion(self, timeout_s: float, threshold: float) -> MotionObservation | None:
        self.thresholds.append(threshold)
        if self.steps:
            observation, advance_s = self.steps.popleft()
            self.clock.sleep(advance_s)
            self.on_frame()
            return observation
        self.clock.sleep(max(timeout_s, 100.0))
        self.exhausted = True
        return None

    def discard_frame(self, timeout_s: float) -> bool:
        self.calls.append(("discard", None))
        return False

    def set_motion_profile(self, profile: MotionProfile) -> None:
        self.calls.append(("profile", profile.input_url))


class CallerConfiguredFrames(ScriptedFrames, FfmpegFramePipeline):
    """An injected FFmpeg provider whose profile belongs to its caller."""

    def __init__(self, clock: ControlledClock, sessions: list[list[FrameStep] | None]) -> None:
        super().__init__(clock, sessions)
        self.profile_calls: list[MotionProfile] = []

    def set_motion_profile(self, profile: MotionProfile) -> None:
        self.profile_calls.append(profile)


@pytest.fixture
def camera_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the external-camera preflight boundary with a fixed stream profile."""

    def run(
        self: RTSPStartupPreflight,
        *,
        camera_name: str,
        primary_rtsp_url: str,
        detect_rtsp_url: str,
        preview_rtsp_url: str | None = None,
        preview_probe_rtsp_url: str | None = None,
        preview_audio_enabled: bool = False,
    ) -> CameraPreflightOutcome:
        return CameraPreflightOutcome(
            camera_key=camera_name,
            motion_profile=MotionProfile(input_url=detect_rtsp_url),
            recording_profile=build_default_recording_profile(primary_rtsp_url),
            diagnostics=CameraPreflightDiagnostics(
                attempted_urls=[detect_rtsp_url],
                probes=[],
                selected_motion_url=detect_rtsp_url,
                selected_recording_url=primary_rtsp_url,
                selected_recording_profile="mp4:v=copy:a=none",
                session_mode="dual_stream",
            ),
        )

    monkeypatch.setattr(RTSPStartupPreflight, "run", run)


async def run_source(
    tmp_path: Path,
    sessions: list[list[FrameStep] | None],
    *,
    refuse_start: bool = False,
    min_changed_pct: float = 30.0,
    debug_motion: bool = False,
    owned_input: ScriptedObservations | None = None,
) -> tuple[ScriptedFrames, RecordingBoundary, list[Clip]]:
    clock = ControlledClock()
    if owned_input is not None:
        owned_input.clock = clock
    frames = ScriptedFrames(clock, sessions)
    recorder = RecordingBoundary(clock, refuse_start=refuse_start)
    config = RTSPSourceConfig.model_validate(
        {
            "rtsp_url": "rtsp://camera/main",
            "detect_rtsp_url": "rtsp://camera/detect",
            "output_dir": str(tmp_path),
            "stream": {"disable_hwaccel": True},
            "motion": {
                "pixel_threshold": 1,
                "min_changed_pct": min_changed_pct,
                "recording_sensitivity_factor": 2.0,
                "blur_kernel": 0,
            },
            "recording": {"stop_delay": 5.0},
            "reconnect": {"max_attempts": 1},
            "runtime": {"debug_motion": debug_motion},
        }
    )
    source = RTSPSource(
        config,
        camera_name="cam",
        frame_pipeline=frames if owned_input is None else None,
        recorder=recorder,
        clock=clock,
    )
    clips: list[Clip] = []
    source.register_callback(clips.append)
    await source.start()
    try:
        finished = frames.finished if owned_input is None else owned_input.finished
        assert await asyncio.to_thread(finished.wait, 2.0), "source did not finish"
    finally:
        source.stop()
    return frames, recorder, clips


@pytest.mark.asyncio
async def test_initial_start_uses_first_frame_as_baseline_and_emits_existing_clip(
    tmp_path: Path, camera_preflight: None
) -> None:
    # Given: Normal startup with black followed by a fully changed white frame.
    session = [FrameStep(bytes(4)), FrameStep(bytes([255] * 4))]

    # When: Running the source through completion using its public callback.
    frames, recorder, clips = await run_source(tmp_path, [session])

    # Then: White starts recording without an initial readiness discard, and cleanup emits once.
    assert len(clips) == 1
    assert [call[0] for call in recorder.calls] == ["start", "stop"]
    assert recorder.calls[0][1] == pytest.approx(1.2)
    assert frames.start_calls[0] == "rtsp://camera/detect"
    assert clips[0].camera_name == "cam"
    assert clips[0].source_backend == "rtsp"
    assert clips[0].clip_id == clips[0].local_path.stem
    assert clips[0].local_path.read_bytes() == b"synthetic completed recording"


@pytest.mark.asyncio
async def test_reconnect_discards_readiness_before_new_baseline(
    tmp_path: Path, camera_preflight: None
) -> None:
    # Given: An initial camera failure, then black readiness and two identical white frames.
    recovered = [FrameStep(bytes(4)), FrameStep(bytes([255] * 4)), FrameStep(bytes([255] * 4))]

    # When: Startup recovers through the source's existing reconnect path.
    frames, recorder, clips = await run_source(tmp_path, [None, recovered])

    # Then: Discarded black never participates in detection or starts recording.
    assert frames.start_calls[:2] == ["rtsp://camera/detect", "rtsp://camera/detect"]
    assert recorder.calls == []
    assert clips == []


@pytest.mark.asyncio
async def test_source_uses_recording_sensitivity_to_extend_stop_delay(
    tmp_path: Path, camera_preflight: None
) -> None:
    # Given: Motion starts recording; a later 25-percent change is below idle sensitivity.
    session = [
        FrameStep(bytes(4)),
        FrameStep(bytes([255, 255, 0, 0])),
        FrameStep(bytes([255, 0, 0, 0]), advance_s=4.0),
        FrameStep(bytes([255, 0, 0, 0]), advance_s=2.0),
        FrameStep(bytes([255, 0, 0, 0]), advance_s=3.1),
    ]

    # When: Consuming observations before and after the refreshed five-second stop window.
    _, recorder, clips = await run_source(tmp_path, [session])

    # Then: Recording stops after the later motion's window, not the initial window.
    assert [call[0] for call in recorder.calls] == ["start", "stop"]
    assert recorder.calls[1][1] == pytest.approx(10.3)
    assert len(clips) == 1


@pytest.mark.asyncio
async def test_failed_start_keeps_idle_motion_sensitivity(
    tmp_path: Path, camera_preflight: None
) -> None:
    # Given: A rejected recording start and a later 25-percent change under idle threshold.
    session = [
        FrameStep(bytes(4)),
        FrameStep(bytes([255, 255, 0, 0])),
        FrameStep(bytes([255, 0, 0, 0]), advance_s=5.1),
    ]

    # When: Reading the later change after the original stop-delay window expires.
    _, recorder, clips = await run_source(tmp_path, [session], refuse_start=True)

    # Then: No new motion window or extra start attempt is created without a recording process.
    assert [call[0] for call in recorder.calls] == ["start"]
    assert clips == []


@pytest.mark.asyncio
async def test_zero_threshold_initial_baseline_does_not_start_recording(
    tmp_path: Path, camera_preflight: None
) -> None:
    # Given: An initial baseline frame with a zero motion percentage threshold.
    session = [FrameStep(bytes(4))]

    # When: Running the source until input is exhausted.
    _, recorder, clips = await run_source(tmp_path, [session], min_changed_pct=0.0)

    # Then: Baseline establishment preserves false and creates no clip.
    assert recorder.calls == []
    assert clips == []


@pytest.mark.asyncio
async def test_common_debug_trace_reports_observations_once_per_hundred_frames(
    tmp_path: Path, camera_preflight: None, caplog: pytest.LogCaptureFixture
) -> None:
    # Given: Motion debugging enabled for one hundred consumed legacy observations.
    session = [FrameStep(bytes(4)) for _ in range(100)]

    # When: The source processes those observations through the common motion policy.
    with caplog.at_level(logging.DEBUG):
        await run_source(tmp_path, [session], debug_motion=True)

    # Then: Exactly one trace preserves the existing observation and configuration fields.
    traces = [record.getMessage() for record in caplog.records if "Motion check:" in record.msg]
    assert traces == [
        "Motion check: changed_pct=0.000% changed_px=0 pixel_threshold=1 min_changed_pct=30.000% blur=0"
    ]


@pytest.mark.asyncio
async def test_input_restart_resets_common_debug_cadence(
    tmp_path: Path, camera_preflight: None, caplog: pytest.LogCaptureFixture
) -> None:
    # Given: Two shorter sessions separated by a reconnect and its readiness discard.
    initial = [FrameStep(bytes(4)) for _ in range(60)]
    recovered = [FrameStep(bytes([255] * 4)) for _ in range(61)]

    # When: Each session supplies fewer than one hundred motion observations.
    with caplog.at_level(logging.DEBUG):
        _, recorder, clips = await run_source(tmp_path, [initial, recovered], debug_motion=True)

    # Then: Resetting the input also resets debug cadence and the comparison baseline.
    assert not [record for record in caplog.records if "Motion check:" in record.msg]
    assert recorder.calls == []
    assert clips == []


@pytest.mark.asyncio
async def test_default_source_consumes_typed_observations_without_pixels(
    tmp_path: Path,
    camera_preflight: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: A selected native input that exposes typed observations and no frame-byte API.
    native = ScriptedObservations(
        [
            (MotionObservation(motion=False, changed_pixels=0, changed_pct=0.0), 0.1),
            (MotionObservation(motion=True, changed_pixels=38400, changed_pct=50.0), 0.1),
            (MotionObservation(motion=True, changed_pixels=19200, changed_pct=25.0), 4.0),
            (MotionObservation(motion=False, changed_pixels=0, changed_pct=0.0), 2.0),
            (MotionObservation(motion=False, changed_pixels=7680, changed_pct=10.0), 3.1),
        ]
    )

    def select_input(
        *, fallback: MotionInput, on_frame: Callable[[], None], **settings: object
    ) -> MotionInput:
        native.on_frame = on_frame
        return native

    monkeypatch.setattr("homesec.sources.rtsp.core.build_motion_input", select_input)

    # When: Running the default source through its native motion-input boundary.
    with caplog.at_level(logging.INFO):
        _, recorder, clips = await run_source(tmp_path, [], owned_input=native)

    # Then: Python selects thresholds, timestamps decisions, emits clips, and retains typed stats.
    assert native.thresholds == [30.0, 30.0, 15.0, 15.0, 15.0, 30.0]
    assert ("profile", "rtsp://camera/detect") in native.calls
    assert recorder.calls == [("start", pytest.approx(1.2)), ("stop", pytest.approx(10.3))]
    assert len(clips) == 1
    stops = [record for record in caplog.records if record.getMessage() == "Recording stopped"]
    assert len(stops) == 1
    assert stops[0].last_changed_pct == 10.0
    assert stops[0].last_changed_pixels == 7680


@pytest.mark.asyncio
async def test_source_preflight_preserves_caller_owned_frame_profile(
    tmp_path: Path, camera_preflight: None
) -> None:
    # Given: An explicitly injected FFmpeg provider with its caller-owned configuration.
    clock = ControlledClock()
    frames = CallerConfiguredFrames(clock, [[FrameStep(bytes(4))]])
    config = RTSPSourceConfig.model_validate(
        {
            "rtsp_url": "rtsp://camera/main",
            "output_dir": str(tmp_path),
            "stream": {"disable_hwaccel": True},
            "motion": {"blur_kernel": 0},
            "reconnect": {"max_attempts": 1},
        }
    )
    source = RTSPSource(config, camera_name="cam", frame_pipeline=frames, clock=clock)

    # When: Source startup selects its own camera profiles and consumes injected input.
    await source.start()
    try:
        assert await asyncio.to_thread(frames.finished.wait, 2.0), "source did not finish"
    finally:
        source.stop()

    # Then: Preflight never overwrites a profile on an input the source does not own.
    assert frames.profile_calls == []
