"""Behavioral contracts for the source-private motion input seam."""

from collections import deque

import pytest
from pydantic import ValidationError

from homesec.sources.rtsp.motion import MotionDetector
from homesec.sources.rtsp.motion_input import FfmpegMotionInput, MotionObservation
from homesec.sources.rtsp.rust_motion import build_motion_input


class QueuedFrames:
    """A bounded frame-provider boundary with the existing oldest-frame drop policy."""

    frame_width: int | None = 2
    frame_height: int | None = 2

    def __init__(self, capacity: int = 20) -> None:
        self.frames: deque[bytes] = deque(maxlen=capacity)
        self.calls: list[tuple[str, str | float | None]] = []
        self.running = False

    def start(self, rtsp_url: str) -> None:
        self.calls.append(("start", rtsp_url))
        self.running = True

    def stop(self) -> None:
        self.calls.append(("stop", None))
        self.running = False
        self.frames.clear()

    def read_frame(self, timeout_s: float) -> bytes | None:
        self.calls.append(("read", timeout_s))
        return self.frames.popleft() if self.running and self.frames else None

    def is_running(self) -> bool:
        return self.running

    def exit_code(self) -> int | None:
        return None if self.running else 0


def make_input(frames: QueuedFrames) -> FfmpegMotionInput:
    return FfmpegMotionInput(
        frames,
        MotionDetector(pixel_threshold=1, min_changed_pct=30.0, blur_kernel=0, debug=False),
    )


def test_initial_frame_establishes_baseline_even_at_zero_threshold() -> None:
    # Given: A motion input with two identical frames and a zero threshold.
    frames = QueuedFrames()
    motion_input = make_input(frames)
    motion_input.start("rtsp://camera/detect")
    frames.frames.extend([bytes(4), bytes(4)])

    # When: Consuming both frames under the existing zero-threshold semantics.
    first = motion_input.read_motion(0.2, threshold=0.0)
    second = motion_input.read_motion(0.2, threshold=0.0)

    # Then: The baseline is false, while the second comparison satisfies zero percent.
    assert first == MotionObservation(motion=False, changed_pixels=0, changed_pct=0.0)
    assert second == MotionObservation(motion=True, changed_pixels=0, changed_pct=0.0)


def test_readiness_discard_does_not_advance_motion_baseline() -> None:
    # Given: A black readiness frame followed by two white frames after reconnect.
    frames = QueuedFrames()
    motion_input = make_input(frames)
    motion_input.start("rtsp://camera/detect")
    frames.frames.extend([bytes(4), bytes([255] * 4), bytes([255] * 4)])

    # When: Discarding readiness before consuming actual motion observations.
    ready = motion_input.discard_frame(0.5)
    first = motion_input.read_motion(0.2, threshold=30.0)
    second = motion_input.read_motion(0.2, threshold=30.0)

    # Then: White establishes the baseline instead of triggering against discarded black.
    assert ready
    assert first == MotionObservation(motion=False, changed_pixels=0, changed_pct=0.0)
    assert second == first


def test_oldest_frames_are_dropped_before_detection() -> None:
    # Given: A black baseline and a frame provider that retains only the newest frame.
    frames = QueuedFrames(capacity=1)
    motion_input = make_input(frames)
    motion_input.start("rtsp://camera/detect")
    frames.frames.append(bytes(4))
    assert motion_input.read_motion(0.2, threshold=30.0) is not None

    # When: White then black arrive before the next observation is requested.
    frames.frames.extend([bytes([255] * 4), bytes(4)])
    result = motion_input.read_motion(0.2, threshold=30.0)

    # Then: Black compares with the consumed black baseline, never with dropped white.
    assert result == MotionObservation(motion=False, changed_pixels=0, changed_pct=0.0)


def test_dropped_frames_do_not_initialize_detector() -> None:
    # Given: A single-slot frame provider before any detection frame has been consumed.
    frames = QueuedFrames(capacity=1)
    motion_input = make_input(frames)
    motion_input.start("rtsp://camera/detect")

    # When: Black is replaced by white before the first observation is requested.
    frames.frames.extend([bytes(4), bytes([255] * 4)])
    result = motion_input.read_motion(0.2, threshold=0.0)

    # Then: The retained white frame establishes a baseline, even with zero threshold.
    assert result == MotionObservation(motion=False, changed_pixels=0, changed_pct=0.0)


@pytest.mark.parametrize(("threshold", "expected_motion"), [(30.0, False), (15.0, True)])
def test_caller_threshold_controls_recording_sensitivity(
    threshold: float, expected_motion: bool
) -> None:
    # Given: A consumed black baseline and a subsequent one-pixel change (25 percent).
    frames = QueuedFrames()
    motion_input = make_input(frames)
    motion_input.start("rtsp://camera/detect")
    frames.frames.extend([bytes(4), bytes([255, 0, 0, 0])])
    assert motion_input.read_motion(0.2, threshold=threshold) is not None

    # When: Applying the threshold selected by the source for idle or recording state.
    result = motion_input.read_motion(0.2, threshold=threshold)

    # Then: Identical pixels produce the expected decision with precise observations.
    assert result == MotionObservation(motion=expected_motion, changed_pixels=1, changed_pct=25.0)


def test_restart_resets_baseline_and_retains_input_lifecycle() -> None:
    # Given: A running input that has observed both black and white frames.
    frames = QueuedFrames()
    motion_input = make_input(frames)
    motion_input.start("rtsp://camera/detect")
    frames.frames.extend([bytes(4), bytes([255] * 4)])
    assert motion_input.read_motion(0.2, threshold=30.0) is not None
    assert motion_input.read_motion(0.2, threshold=30.0) == MotionObservation(
        motion=True, changed_pixels=4, changed_pct=100.0
    )

    # When: Stopping and starting on a different stream with a black first frame.
    motion_input.stop()
    stopped = (motion_input.is_running(), motion_input.exit_code())
    motion_input.start("rtsp://camera/main")
    frames.frames.append(bytes(4))
    result = motion_input.read_motion(0.2, threshold=0.0)

    # Then: Lifecycle forwards to the provider, and the new stream establishes a baseline.
    assert stopped == (False, 0)
    assert motion_input.is_running()
    assert motion_input.exit_code() is None
    assert result == MotionObservation(motion=False, changed_pixels=0, changed_pct=0.0)
    assert [call for call in frames.calls if call[0] != "read"] == [
        ("start", "rtsp://camera/detect"),
        ("stop", None),
        ("start", "rtsp://camera/main"),
    ]


def test_missing_frame_does_not_advance_detector() -> None:
    # Given: A consumed black baseline followed by a period with no available input.
    frames = QueuedFrames()
    motion_input = make_input(frames)
    motion_input.start("rtsp://camera/detect")
    frames.frames.append(bytes(4))
    assert motion_input.read_motion(0.2, threshold=30.0) is not None

    # When: Waiting without a frame, then consuming a white frame.
    missing = motion_input.read_motion(0.5, threshold=30.0)
    frames.frames.append(bytes([255] * 4))
    result = motion_input.read_motion(0.2, threshold=30.0)

    # Then: Timeout remains a missing sample and white compares with the retained baseline.
    assert missing is None
    assert result == MotionObservation(motion=True, changed_pixels=4, changed_pct=100.0)
    assert ("read", 0.5) in frames.calls


@pytest.mark.parametrize(
    ("helper_available", "hwaccel_active", "pixel_threshold", "min_changed_pct", "timeout"),
    [
        (False, False, 1, 30.0, 2.0),
        (True, True, 1, 30.0, 2.0),
        (True, False, 2**64, 30.0, 2.0),
        (True, False, 1, float("inf"), 2.0),
        (True, False, 1, 30.0, 0.0),
        (True, False, 1, 30.0, 121.0),
    ],
)
def test_unavailable_or_unsupported_native_settings_keep_existing_motion_input(
    monkeypatch: pytest.MonkeyPatch,
    helper_available: bool,
    hwaccel_active: bool,
    pixel_threshold: int,
    min_changed_pct: float,
    timeout: float,
) -> None:
    # Given: An existing input and a native configuration the initial decoder cannot use.
    monkeypatch.setattr(
        "homesec.sources.rtsp.rust_motion.shutil.which",
        lambda _path: "/synthetic/helper" if helper_available else None,
    )
    frames = QueuedFrames()
    fallback = make_input(frames)

    # When: Selecting and consuming motion through the default source-private factory.
    selected = build_motion_input(
        fallback=fallback,
        pixel_threshold=pixel_threshold,
        min_changed_pct=min_changed_pct,
        blur_kernel=0,
        recording_sensitivity_factor=2.0,
        frame_queue_size=20,
        rtsp_connect_timeout_s=timeout,
        rtsp_io_timeout_s=2.0,
        hwaccel_active=hwaccel_active,
        on_frame=lambda: None,
    )
    selected.start("rtsp://camera/detect")
    frames.frames.extend([bytes(4), bytes([255] * 4)])
    first = selected.read_motion(0.2, threshold=30.0)
    second = selected.read_motion(0.2, threshold=30.0)

    # Then: The existing provider and detector remain operational without a native subprocess.
    assert selected is fallback
    assert first == MotionObservation(motion=False, changed_pixels=0, changed_pct=0.0)
    assert second == MotionObservation(motion=True, changed_pixels=4, changed_pct=100.0)
    selected.stop()


@pytest.mark.parametrize(
    "invalid",
    [
        {"motion": 1},
        {"changed_pixels": -1},
        {"changed_pixels": 76801},
        {"changed_pixels": "1"},
        {"changed_pct": float("nan")},
        {"changed_pct": float("inf")},
        {"changed_pct": 100.1},
        {"frames": "private media"},
    ],
)
def test_motion_observations_validate_boundary_payloads(invalid: dict[str, object]) -> None:
    # Given: A valid observation altered with an invalid or unexpected boundary field.
    payload: dict[str, object] = {"motion": False, "changed_pixels": 0, "changed_pct": 0.0}
    payload.update(invalid)

    # When: Validating the cross-process observation payload.
    with pytest.raises(ValidationError):
        MotionObservation.model_validate(payload)

    # Then: Unsafe values never become a motion observation.
