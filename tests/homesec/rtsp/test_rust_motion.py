"""Rust motion adapter contracts at the executable/JSON process boundary."""

import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from homesec.sources.rtsp.motion_input import MotionInput, MotionObservation
from homesec.sources.rtsp.recording_profile import MotionProfile
from homesec.sources.rtsp.rust_motion import build_motion_input

_HELPER = r"""#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

mode = MODE
invalid = INVALID
trace = Path(__file__).with_suffix(".calls")
pending = None
url = ""

def emit(value):
    print(json.dumps(value), flush=True)

emit({"event": "ready"})
for line in sys.stdin:
    request = json.loads(line)
    with trace.open("a") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "request": request}) + "\n")
    command = request["command"]
    reply = {"request_id": request["request_id"], "ok": True}
    if command == "start":
        url = request["rtsp_url"]
        if mode == "refuse_start":
            reply.update(ok=False, error_code="unsupported_codec")
        else:
            emit({"event": "frame"})
    elif command == "read_motion":
        if mode == "pending" or (mode == "restart" and url.endswith("/old")):
            pending = request["request_id"]
            continue
        if mode == "refuse_read":
            reply.update(ok=False, error_code="decode_failed")
        elif mode == "invalid":
            reply["observation"] = invalid
        elif mode != "empty":
            emit({"event": "frame"})
            reply["observation"] = {
                "motion": 25.0 >= request["threshold"],
                "changed_pixels": 19200,
                "changed_pct": 25.0,
            }
    elif command == "discard_frame":
        if mode == "refuse_read":
            reply.update(ok=False, error_code="decode_failed")
        else:
            reply["frame_available"] = True
    elif command == "stop":
        if pending is not None:
            emit({"request_id": pending, "ok": False, "error_code": "cancelled"})
            # Teardown events from the old generation must not renew source health.
            emit({"event": "frame"})
        emit(reply)
        break
    emit(reply)
"""


class LegacyInput:
    def __init__(self, *, fail_start: bool = False) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.running = False
        self.fail_start = fail_start
        self.observation = MotionObservation(motion=False, changed_pixels=0, changed_pct=0.0)

    def start(self, rtsp_url: str) -> None:
        self.calls.append(("start", rtsp_url))
        if self.fail_start:
            raise RuntimeError("synthetic legacy startup failed")
        self.running = True

    def stop(self) -> None:
        self.calls.append(("stop",))
        self.running = False

    def is_running(self) -> bool:
        return self.running

    def exit_code(self) -> int | None:
        return None if self.running else 0

    def read_motion(self, timeout_s: float, threshold: float) -> MotionObservation | None:
        self.calls.append(("read_motion", timeout_s, threshold))
        return self.observation if self.running else None

    def discard_frame(self, timeout_s: float) -> bool:
        self.calls.append(("discard", timeout_s))
        return self.running

    def set_motion_profile(self, profile: MotionProfile) -> None:
        self.calls.append(("profile", profile))


def helper_script(
    tmp_path: Path, mode: str = "ok", invalid: dict[str, object] | None = None
) -> Path:
    helper = tmp_path / "motion-helper"
    helper.write_text(_HELPER.replace("MODE", repr(mode)).replace("INVALID", repr(invalid)))
    helper.chmod(0o700)
    return helper


def trace_requests(helper: Path) -> list[dict[str, object]]:
    trace = helper.with_suffix(".calls")
    if not trace.exists():
        return []
    return [json.loads(line) for line in trace.read_text().splitlines() if line]


def wait_for_command(helper: Path, command: str) -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        records = trace_requests(helper)
        if any(record["request"]["command"] == command for record in records):
            return
        time.sleep(0.005)
    pytest.fail(f"synthetic helper did not receive {command}")


def selected_input(
    helper: Path, fallback: LegacyInput, on_frame: Callable[[], None] = lambda: None
) -> MotionInput:
    return build_motion_input(
        fallback=fallback,
        pixel_threshold=47,
        min_changed_pct=30.0,
        blur_kernel=7,
        recording_sensitivity_factor=2.0,
        frame_queue_size=3,
        rtsp_connect_timeout_s=2.0,
        rtsp_io_timeout_s=3.0,
        hwaccel_active=False,
        helper_path=str(helper),
        on_frame=on_frame,
    )


def test_native_helper_receives_existing_settings_and_typed_observations(tmp_path: Path) -> None:
    # Given: An executable motion helper and the selected detection stream profile.
    helper = helper_script(tmp_path)
    fallback = LegacyInput()
    health_events: list[str] = []
    motion_input = selected_input(helper, fallback, lambda: health_events.append("frame"))
    profile = MotionProfile(input_url="rtsp://synthetic-camera/detect")
    motion_input.set_motion_profile(profile)

    # When: Starting, consuming a readiness sample, and requesting both source thresholds.
    try:
        motion_input.start(profile.input_url)
        assert motion_input.is_running()
        assert motion_input.exit_code() is None
        assert motion_input.discard_frame(0.2)
        idle = motion_input.read_motion(0.2, threshold=30.0)
        recording = motion_input.read_motion(0.2, threshold=15.0)
    finally:
        motion_input.stop()

    # Then: The existing values and JSON-only controls reach the helper without pixel payloads.
    requests = [record["request"] for record in trace_requests(helper)]
    start = next(request for request in requests if request["command"] == "start")
    assert start["rtsp_url"] == profile.input_url
    assert start["motion_config"] == {
        "pixel_threshold": 47,
        "min_changed_pct": 30.0,
        "blur_kernel": 7,
        "recording_sensitivity_factor": 2.0,
    }
    assert start["frame_queue_size"] == 3
    assert start["connect_timeout_s"] == 2.0
    assert start["io_timeout_s"] == 3.0
    assert [
        request["threshold"] for request in requests if request["command"] == "read_motion"
    ] == [
        30.0,
        15.0,
    ]
    assert all("pixels" not in request and "frame" not in request for request in requests)
    assert idle == MotionObservation(motion=False, changed_pixels=19200, changed_pct=25.0)
    assert recording == MotionObservation(motion=True, changed_pixels=19200, changed_pct=25.0)
    assert health_events == ["frame", "frame", "frame"]
    assert not [call for call in fallback.calls if call[0] == "start"]
    assert not motion_input.is_running()


@pytest.mark.parametrize("mode", ["refuse_start", "refuse_read"])
def test_unsupported_or_failed_native_motion_uses_legacy_input(tmp_path: Path, mode: str) -> None:
    # Given: A native helper that refuses its camera format or fails when decoding.
    helper = helper_script(tmp_path, mode)
    fallback = LegacyInput()
    motion_input = selected_input(helper, fallback)
    url = "rtsp://synthetic-camera/detect"

    # When: Starting and reading with source-provided recording sensitivity.
    try:
        motion_input.start(url)
        observation = motion_input.read_motion(0.2, threshold=15.0)
        running = motion_input.is_running()
    finally:
        motion_input.stop()

    # Then: Legacy input receives the selected stream and unchanged read contract.
    assert ("start", url) in fallback.calls
    assert ("read_motion", 0.2, 15.0) in fallback.calls
    assert observation == fallback.observation
    assert running
    assert not motion_input.is_running()


@pytest.mark.parametrize("consume", ["read_motion", "discard_frame", "incompatible_deadline"])
def test_runtime_fallback_start_failure_allows_source_restart(
    tmp_path: Path, consume: str, caplog: pytest.LogCaptureFixture
) -> None:
    # Given: A failing native input and a transiently unavailable compatibility camera.
    helper = helper_script(tmp_path, "refuse_read")
    fallback = LegacyInput(fail_start=True)
    motion_input = selected_input(helper, fallback)
    url = "rtsp://synthetic-camera/detect"

    # When: Runtime fallback fails once, then the source restarts the selected input.
    try:
        motion_input.start(url)
        with caplog.at_level(logging.WARNING):
            if consume == "read_motion":
                assert motion_input.read_motion(0.2, threshold=30.0) is None
            elif consume == "discard_frame":
                assert not motion_input.discard_frame(0.2)
            else:
                assert not motion_input.discard_frame(121.0)
        assert not motion_input.is_running()
        fallback.fail_start = False
        motion_input.start(url)
        recovered = motion_input.read_motion(0.2, threshold=30.0)
    finally:
        motion_input.stop()

    # Then: Missing input permits retry, and the recovered legacy detector supplies motion.
    assert recovered == fallback.observation
    assert [call for call in fallback.calls if call[0] == "start"] == [
        ("start", url),
        ("start", url),
    ]
    assert sum(record["request"]["command"] == "start" for record in trace_requests(helper)) == 1
    failures = [record for record in caplog.records if "source will reconnect" in record.msg]
    assert len(failures) == 1
    assert failures[0].exc_info is None


@pytest.mark.parametrize(
    "invalid",
    [
        {"motion": "true", "changed_pixels": 0, "changed_pct": 0.0},
        {"motion": True, "changed_pixels": 76801, "changed_pct": 100.0},
        {"motion": True, "changed_pixels": 1, "changed_pct": 101.0},
        {"motion": True, "changed_pixels": 1, "changed_pct": 1.0, "pixels": "synthetic pixels"},
    ],
)
def test_invalid_observation_fails_closed_into_legacy_input(
    tmp_path: Path, invalid: dict[str, object]
) -> None:
    # Given: A helper that returns a malformed or over-limit typed observation.
    helper = helper_script(tmp_path, "invalid", invalid)
    fallback = LegacyInput()
    motion_input = selected_input(helper, fallback)

    # When: Reading through the validated helper transport.
    try:
        motion_input.start("rtsp://synthetic-camera/detect")
        result = motion_input.read_motion(0.2, threshold=30.0)
    finally:
        motion_input.stop()

    # Then: Invalid native data is discarded and the existing detector supplies the observation.
    assert result == fallback.observation
    assert [call for call in fallback.calls if call[0] == "start"] == [
        ("start", "rtsp://synthetic-camera/detect")
    ]


def test_custom_input_profile_retains_legacy_ffmpeg_semantics(tmp_path: Path) -> None:
    # Given: A negotiated FFmpeg input workaround the initial native decoder cannot reproduce.
    helper = helper_script(tmp_path)
    fallback = LegacyInput()
    motion_input = selected_input(helper, fallback)
    profile = MotionProfile(
        input_url="rtsp://synthetic-camera/detect", ffmpeg_input_args=["-fflags", "+discardcorrupt"]
    )

    # When: Applying that profile before starting the selected input.
    try:
        motion_input.set_motion_profile(profile)
        motion_input.start(profile.input_url)
        ready = motion_input.discard_frame(0.5)
        observation = motion_input.read_motion(0.2, threshold=15.0)
    finally:
        motion_input.stop()

    # Then: Profile, readiness discard, and sensitivity remain on the established path.
    assert ("profile", profile) in fallback.calls
    assert ("start", profile.input_url) in fallback.calls
    assert ("discard", 0.5) in fallback.calls
    assert observation == fallback.observation
    assert ready
    assert trace_requests(helper) == []


def test_native_empty_read_is_missing_input_without_immediate_fallback(tmp_path: Path) -> None:
    # Given: A live helper whose bounded wait finds no available prepared frame.
    helper = helper_script(tmp_path, "empty")
    fallback = LegacyInput()
    motion_input = selected_input(helper, fallback)

    # When: Waiting for an observation without a decoder or protocol failure.
    try:
        motion_input.start("rtsp://synthetic-camera/detect")
        result = motion_input.read_motion(0.2, threshold=30.0)
        running = motion_input.is_running()
    finally:
        motion_input.stop()

    # Then: The source receives missing input for its existing reconnect/stall policy.
    assert result is None
    assert running
    assert not [call for call in fallback.calls if call[0] == "start"]


def test_stop_cancels_pending_read_without_restarting_legacy(tmp_path: Path) -> None:
    # Given: A helper with a pending observation request and a source-health callback.
    helper = helper_script(tmp_path, "pending")
    fallback = LegacyInput()
    health_events: list[str] = []
    motion_input = selected_input(helper, fallback, lambda: health_events.append("frame"))
    motion_input.start("rtsp://synthetic-camera/detect")

    # When: Stopping while the executable is still awaiting a prepared detection frame.
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(motion_input.read_motion, 10.0, 30.0)
        wait_for_command(helper, "read_motion")
        motion_input.stop()
        result = pending.result(timeout=2.0)

    # Then: Cancellation releases the read and ignores teardown health/fallback activation.
    assert result is None
    assert health_events == ["frame"]
    assert not [call for call in fallback.calls if call[0] == "start"]
    assert not motion_input.is_running()


def test_stale_helper_cancellation_does_not_replace_new_generation(tmp_path: Path) -> None:
    # Given: A pending read against the old stream's helper generation.
    helper = helper_script(tmp_path, "restart")
    fallback = LegacyInput()
    health_events: list[str] = []
    motion_input = selected_input(helper, fallback, lambda: health_events.append("frame"))
    motion_input.start("rtsp://synthetic-camera/old")

    # When: Starting a new stream while teardown cancels the old helper's request.
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(motion_input.read_motion, 10.0, 30.0)
            wait_for_command(helper, "read_motion")
            motion_input.start("rtsp://synthetic-camera/new")
            old_result = pending.result(timeout=2.0)
        new_result = motion_input.read_motion(0.2, threshold=15.0)
    finally:
        motion_input.stop()

    # Then: Only the new helper controls motion; stale failures/events never activate legacy.
    assert old_result is None
    assert new_result == MotionObservation(motion=True, changed_pixels=19200, changed_pct=25.0)
    assert not [call for call in fallback.calls if call[0] == "start"]
    assert health_events == ["frame", "frame", "frame"]
    starts = [
        record for record in trace_requests(helper) if record["request"]["command"] == "start"
    ]
    assert len(starts) == 2
    assert starts[0]["pid"] != starts[1]["pid"]


def test_fallback_failure_does_not_log_handled_sensitive_validation_context(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Given: A synthetic oversized camera URL and a legacy input that also fails to start.
    helper = helper_script(tmp_path)
    fallback = LegacyInput(fail_start=True)
    motion_input = selected_input(helper, fallback)
    marker = "synthetic-password-private-marker"
    url = f"rtsp://synthetic-user:{marker * 600}@synthetic-camera/detect"

    # When: Reporting only the remaining fallback error after native request validation failed.
    try:
        with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="legacy startup"):
            try:
                motion_input.start(url)
            except RuntimeError:
                logging.getLogger(__name__).exception("Synthetic compatibility startup failed")
                raise
    finally:
        motion_input.stop()

    # Then: The handled validation exception and its synthetic credential input are absent.
    assert marker not in caplog.text
    assert "ValidationError" not in caplog.text
    assert "synthetic legacy startup failed" in caplog.text
