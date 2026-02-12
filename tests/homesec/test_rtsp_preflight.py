"""Tests for RTSP startup preflight orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from homesec.sources.rtsp.discovery import CameraProbeResult, ProbeStreamInfo
from homesec.sources.rtsp.preflight import (
    PreflightError,
    RecordingValidationResult,
    RecordingValidationSignals,
    RTSPStartupPreflight,
)
from homesec.sources.rtsp.recording_profile import RecordingProfile


class _FakeDiscovery:
    def __init__(self, streams: list[ProbeStreamInfo]) -> None:
        self._streams = streams
        self.last_candidate_urls: list[str] = []

    def probe(self, *, camera_key: str, candidate_urls: list[str]) -> CameraProbeResult:
        self.last_candidate_urls = list(candidate_urls)
        streams = [stream for stream in self._streams if stream.url in candidate_urls]
        return CameraProbeResult(
            camera_key=camera_key,
            streams=streams,
            attempted_urls=candidate_urls,
            duration_ms=12,
        )


def test_preflight_selects_low_motion_and_high_recording_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight should pick the cheapest stream for motion and best stream for recording."""
    # Given: two viable streams with different resolutions
    streams = [
        ProbeStreamInfo(
            url="rtsp://cam/high",
            video_codec="h264",
            audio_codec="aac",
            width=1920,
            height=1080,
            fps=20.0,
            fps_raw="20/1",
            probe_ok=True,
        ),
        ProbeStreamInfo(
            url="rtsp://cam/low",
            video_codec="h264",
            audio_codec="aac",
            width=640,
            height=360,
            fps=10.0,
            fps_raw="10/1",
            probe_ok=True,
        ),
    ]
    preflight = RTSPStartupPreflight(
        output_dir=tmp_path,
        rtsp_connect_timeout_s=2.0,
        rtsp_io_timeout_s=2.0,
        discovery=_FakeDiscovery(streams),
    )

    def _validate_profile(_profile: RecordingProfile) -> RecordingValidationResult:
        return RecordingValidationResult(ok=True)

    monkeypatch.setattr(preflight, "_validate_recording_profile", _validate_profile)
    monkeypatch.setattr(preflight, "_validate_session_limits", lambda _m, _r: True)

    # When: running startup preflight
    outcome = preflight.run(
        camera_name="garage",
        primary_rtsp_url="rtsp://cam/high",
        detect_rtsp_url="rtsp://cam/low",
    )

    # Then: motion uses low stream while recording uses high stream
    assert not isinstance(outcome, PreflightError)
    assert outcome.motion_profile.input_url == "rtsp://cam/low"
    assert outcome.recording_profile.input_url == "rtsp://cam/high"
    assert outcome.diagnostics.session_mode == "dual_stream"


def test_preflight_falls_back_to_single_stream_when_dual_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight should fallback to single-stream mode when dual-stream open fails."""
    # Given: dual-stream validation fails but single-stream succeeds
    streams = [
        ProbeStreamInfo(
            url="rtsp://cam/high",
            video_codec="h264",
            audio_codec="aac",
            width=1920,
            height=1080,
            fps=20.0,
            fps_raw="20/1",
            probe_ok=True,
        ),
        ProbeStreamInfo(
            url="rtsp://cam/low",
            video_codec="h264",
            audio_codec="aac",
            width=640,
            height=360,
            fps=10.0,
            fps_raw="10/1",
            probe_ok=True,
        ),
    ]
    preflight = RTSPStartupPreflight(
        output_dir=tmp_path,
        rtsp_connect_timeout_s=2.0,
        rtsp_io_timeout_s=2.0,
        discovery=_FakeDiscovery(streams),
    )

    def _validate_profile(_profile: RecordingProfile) -> RecordingValidationResult:
        return RecordingValidationResult(ok=True)

    call_count = 0

    def _validate_session(_motion_url: str, _recording_url: str) -> bool:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return False
        return True

    monkeypatch.setattr(preflight, "_validate_recording_profile", _validate_profile)
    monkeypatch.setattr(preflight, "_validate_session_limits", _validate_session)

    # When: running startup preflight
    outcome = preflight.run(
        camera_name="garage",
        primary_rtsp_url="rtsp://cam/high",
        detect_rtsp_url="rtsp://cam/low",
    )

    # Then: it falls back to single-stream mode for motion+recording
    assert not isinstance(outcome, PreflightError)
    assert outcome.motion_profile.input_url == outcome.recording_profile.input_url
    assert outcome.diagnostics.session_mode == "single_stream"


def test_preflight_returns_negotiation_error_for_unusable_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight should fail with negotiation error when all recording profiles fail."""
    # Given: one viable garage-style stream with pcm_alaw audio
    streams = [
        ProbeStreamInfo(
            url="rtsp://cam/garage",
            video_codec="h264",
            audio_codec="pcm_alaw",
            width=1280,
            height=720,
            fps=15.0,
            fps_raw="15/1",
            probe_ok=True,
        ),
    ]
    preflight = RTSPStartupPreflight(
        output_dir=tmp_path,
        rtsp_connect_timeout_s=2.0,
        rtsp_io_timeout_s=2.0,
        discovery=_FakeDiscovery(streams),
    )

    def _validate_profile(_profile: RecordingProfile) -> RecordingValidationResult:
        return RecordingValidationResult(ok=False, error="simulated ffmpeg failure")

    monkeypatch.setattr(preflight, "_validate_recording_profile", _validate_profile)

    # When: running startup preflight
    outcome = preflight.run(
        camera_name="garage",
        primary_rtsp_url="rtsp://cam/garage",
        detect_rtsp_url="rtsp://cam/garage",
    )

    # Then: a typed preflight negotiation error is returned
    assert isinstance(outcome, PreflightError)
    assert outcome.stage == "negotiation"
    assert outcome.diagnostics is not None
    assert outcome.diagnostics.negotiation_attempts == ["mp4:v=copy:a=aac", "mp4:v=copy:a=none"]


def test_preflight_discovers_stream2_candidate_from_stream1_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight should probe /stream2 when configured with only /stream1."""
    # Given: discovery metadata for stream1 (high) and stream2 (low)
    streams = [
        ProbeStreamInfo(
            url="rtsp://cam/stream1",
            video_codec="h264",
            audio_codec="aac",
            width=1920,
            height=1080,
            fps=20.0,
            fps_raw="20/1",
            probe_ok=True,
        ),
        ProbeStreamInfo(
            url="rtsp://cam/stream2",
            video_codec="h264",
            audio_codec="aac",
            width=640,
            height=360,
            fps=10.0,
            fps_raw="10/1",
            probe_ok=True,
        ),
    ]
    fake_discovery = _FakeDiscovery(streams)
    preflight = RTSPStartupPreflight(
        output_dir=tmp_path,
        rtsp_connect_timeout_s=2.0,
        rtsp_io_timeout_s=2.0,
        discovery=fake_discovery,
    )

    def _validate_profile(_profile: RecordingProfile) -> RecordingValidationResult:
        return RecordingValidationResult(ok=True)

    monkeypatch.setattr(preflight, "_validate_recording_profile", _validate_profile)
    monkeypatch.setattr(preflight, "_validate_session_limits", lambda _m, _r: True)

    # When: running startup preflight with only stream1 configured
    outcome = preflight.run(
        camera_name="garage",
        primary_rtsp_url="rtsp://cam/stream1",
        detect_rtsp_url="rtsp://cam/stream1",
    )

    # Then: stream2 is probed and selected for motion while stream1 stays for recording
    assert not isinstance(outcome, PreflightError)
    assert "rtsp://cam/stream2" in fake_discovery.last_candidate_urls
    assert outcome.motion_profile.input_url == "rtsp://cam/stream2"
    assert outcome.recording_profile.input_url == "rtsp://cam/stream1"


def test_preflight_enables_wallclock_timestamps_when_baseline_is_unstable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight should lock wallclock timestamps when validation shows DTS instability."""
    # Given: one viable stream with baseline timing instability warnings
    streams = [
        ProbeStreamInfo(
            url="rtsp://cam/high",
            video_codec="h264",
            audio_codec="aac",
            width=1920,
            height=1080,
            fps=20.0,
            fps_raw="20/1",
            probe_ok=True,
        ),
    ]
    preflight = RTSPStartupPreflight(
        output_dir=tmp_path,
        rtsp_connect_timeout_s=2.0,
        rtsp_io_timeout_s=2.0,
        discovery=_FakeDiscovery(streams),
    )

    def _validate_profile(profile: RecordingProfile) -> RecordingValidationResult:
        if profile.uses_wallclock_timestamps():
            return RecordingValidationResult(
                ok=True,
                signals=RecordingValidationSignals(
                    non_monotonic_dts=1,
                    queue_input_backward=0,
                    dts_discontinuity=0,
                ),
            )
        return RecordingValidationResult(
            ok=True,
            signals=RecordingValidationSignals(
                non_monotonic_dts=20,
                queue_input_backward=2,
                dts_discontinuity=0,
            ),
        )

    monkeypatch.setattr(preflight, "_validate_recording_profile", _validate_profile)
    monkeypatch.setattr(preflight, "_validate_session_limits", lambda _m, _r: True)

    # When: running startup preflight
    outcome = preflight.run(
        camera_name="garage",
        primary_rtsp_url="rtsp://cam/high",
        detect_rtsp_url="rtsp://cam/high",
    )

    # Then: wallclock timestamps are enabled for recording
    assert not isinstance(outcome, PreflightError)
    assert outcome.recording_profile.uses_wallclock_timestamps()
    assert outcome.diagnostics.selected_recording_profile == "mp4:v=copy:a=copy:ts=wallclock"
    assert "mp4:v=copy:a=copy:ts=wallclock" in outcome.diagnostics.negotiation_attempts


def test_validate_recording_profile_collects_stability_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recording validation should capture timestamp instability warning signals."""

    class _FakeResult:
        def __init__(self, returncode: int, stderr: str) -> None:
            self.returncode = returncode
            self.stderr = stderr
            self.stdout = ""

    preflight = RTSPStartupPreflight(
        output_dir=tmp_path,
        rtsp_connect_timeout_s=2.0,
        rtsp_io_timeout_s=2.0,
        discovery=_FakeDiscovery([]),
    )
    profile = RecordingProfile(
        input_url="rtsp://cam/high",
        audio_mode="aac",
        ffmpeg_output_args=["-c:v", "copy", "-c:a", "aac", "-f", "mp4"],
    )

    def _fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
    ) -> _FakeResult:
        # Given: a validation ffmpeg command that logs warnings and succeeds
        _ = capture_output
        _ = text
        _ = timeout
        _ = check
        idx = cmd.index("-loglevel")
        assert cmd[idx + 1] == "warning"
        output_path = Path(cmd[-1])
        output_path.write_bytes(b"ok")
        return _FakeResult(
            returncode=0,
            stderr=(
                "Non-monotonic DTS\\n"
                "Non-monotonic DTS\\n"
                "Queue input is backward in time\\n"
                "DTS discontinuity\\n"
                "SEI type 764 size\\n"
            ),
        )

    monkeypatch.setattr("homesec.sources.rtsp.preflight.subprocess.run", _fake_run)

    # When: validating a recording profile
    result = preflight._validate_recording_profile(profile)

    # Then: warning counts are captured and returned with a successful validation
    assert result.ok
    assert result.signals.non_monotonic_dts == 2
    assert result.signals.queue_input_backward == 1
    assert result.signals.dts_discontinuity == 1
    assert result.signals.sei_truncated == 1


def test_validate_recording_profile_requires_output_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation should fail if ffmpeg reports success but no output clip exists."""

    class _FakeResult:
        def __init__(self, returncode: int, stderr: str = "") -> None:
            self.returncode = returncode
            self.stderr = stderr
            self.stdout = ""

    preflight = RTSPStartupPreflight(
        output_dir=tmp_path,
        rtsp_connect_timeout_s=2.0,
        rtsp_io_timeout_s=2.0,
        discovery=_FakeDiscovery([]),
    )
    profile = RecordingProfile(
        input_url="rtsp://cam/high",
        audio_mode="aac",
        ffmpeg_output_args=["-c:v", "copy", "-c:a", "aac", "-f", "mp4"],
    )

    def _fake_run(
        _cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
    ) -> _FakeResult:
        _ = capture_output
        _ = text
        _ = timeout
        _ = check
        return _FakeResult(returncode=0)

    monkeypatch.setattr("homesec.sources.rtsp.preflight.subprocess.run", _fake_run)

    # When: validating a profile with no output produced
    result = preflight._validate_recording_profile(profile)

    # Then: validation fails closed
    assert not result.ok
    assert result.error is not None
    assert "empty" in result.error


def test_session_limit_validation_retries_without_timeout_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session-limit validation should retry when ffmpeg rejects timeout options."""

    class _FakeProc:
        def __init__(self, *, returncode: int | None, stderr_text: str) -> None:
            self._returncode = returncode
            self._stderr_text = stderr_text

        def poll(self) -> int | None:
            return self._returncode

        def terminate(self) -> None:
            if self._returncode is None:
                self._returncode = 0

        def communicate(self, timeout: float) -> tuple[str, str]:
            _ = timeout
            if self._returncode is None:
                self._returncode = 0
            return ("", self._stderr_text)

        def kill(self) -> None:
            self._returncode = -9

    calls: list[list[str]] = []

    def _fake_popen(cmd: list[str], **_kwargs: object) -> _FakeProc:
        calls.append(list(cmd))
        if "-rw_timeout" in cmd or "-stimeout" in cmd:
            return _FakeProc(returncode=1, stderr_text="Option rw_timeout not found")
        return _FakeProc(returncode=None, stderr_text="")

    preflight = RTSPStartupPreflight(
        output_dir=tmp_path,
        rtsp_connect_timeout_s=2.0,
        rtsp_io_timeout_s=2.0,
        discovery=_FakeDiscovery([]),
    )
    monkeypatch.setattr("homesec.sources.rtsp.preflight.subprocess.Popen", _fake_popen)
    monkeypatch.setattr("homesec.sources.rtsp.preflight.time.sleep", lambda _seconds: None)

    # Given: ffmpeg fails with timeout-option errors and then succeeds without them
    motion_url = "rtsp://cam/motion"
    recording_url = "rtsp://cam/recording"

    # When: validating concurrent session limits
    ok = preflight._validate_session_limits(motion_url, recording_url)

    # Then: validation succeeds after retry without timeout options
    assert ok
    assert len(calls) == 4
    assert "-rw_timeout" in calls[0] or "-stimeout" in calls[0]
    assert "-rw_timeout" not in calls[-1]
    assert "-stimeout" not in calls[-1]


def test_session_limit_validation_non_timeout_failure_does_not_retry_without_timeouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session-limit validation should fail fast on non-timeout launch failures."""

    class _FakeProc:
        def __init__(self, *, returncode: int | None, stderr_text: str) -> None:
            self._returncode = returncode
            self._stderr_text = stderr_text

        def poll(self) -> int | None:
            return self._returncode

        def terminate(self) -> None:
            if self._returncode is None:
                self._returncode = 0

        def communicate(self, timeout: float) -> tuple[str, str]:
            _ = timeout
            if self._returncode is None:
                self._returncode = 0
            return ("", self._stderr_text)

        def kill(self) -> None:
            self._returncode = -9

    calls: list[list[str]] = []

    def _fake_popen(cmd: list[str], **_kwargs: object) -> _FakeProc:
        calls.append(list(cmd))
        return _FakeProc(returncode=1, stderr_text="Connection refused")

    preflight = RTSPStartupPreflight(
        output_dir=tmp_path,
        rtsp_connect_timeout_s=2.0,
        rtsp_io_timeout_s=2.0,
        discovery=_FakeDiscovery([]),
    )
    monkeypatch.setattr("homesec.sources.rtsp.preflight.subprocess.Popen", _fake_popen)
    monkeypatch.setattr("homesec.sources.rtsp.preflight.time.sleep", lambda _seconds: None)

    # Given: concurrent stream opens fail for non-timeout reasons
    motion_url = "rtsp://cam/motion"
    recording_url = "rtsp://cam/recording"

    # When: validating concurrent session limits
    ok = preflight._validate_session_limits(motion_url, recording_url)

    # Then: validation fails without a no-timeouts retry
    assert not ok
    assert len(calls) == 2
    assert "-rw_timeout" in calls[0] or "-stimeout" in calls[0]


def test_session_limit_validation_requires_clean_exit_after_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session-limit validation should fail when opens die after overlap window."""

    class _FakeProc:
        def __init__(self, *, final_exit: int, stderr_text: str) -> None:
            self._returncode: int | None = None
            self._final_exit = final_exit
            self._stderr_text = stderr_text

        def poll(self) -> int | None:
            return self._returncode

        def terminate(self) -> None:
            if self._returncode is None:
                self._returncode = 0

        def communicate(self, timeout: float) -> tuple[str, str]:
            _ = timeout
            if self._returncode is None:
                self._returncode = self._final_exit
            return ("", self._stderr_text)

        def kill(self) -> None:
            self._returncode = -9

    calls: list[list[str]] = []

    def _fake_popen(cmd: list[str], **_kwargs: object) -> _FakeProc:
        calls.append(list(cmd))
        return _FakeProc(final_exit=1, stderr_text="Session ended")

    preflight = RTSPStartupPreflight(
        output_dir=tmp_path,
        rtsp_connect_timeout_s=2.0,
        rtsp_io_timeout_s=2.0,
        discovery=_FakeDiscovery([]),
    )
    monkeypatch.setattr("homesec.sources.rtsp.preflight.subprocess.Popen", _fake_popen)
    monkeypatch.setattr("homesec.sources.rtsp.preflight.time.sleep", lambda _seconds: None)

    # Given: both ffmpeg session checks appear alive during overlap but exit non-zero
    motion_url = "rtsp://cam/motion"
    recording_url = "rtsp://cam/recording"

    # When: validating concurrent session limits
    ok = preflight._validate_session_limits(motion_url, recording_url)

    # Then: validation fails closed instead of treating overlap-only liveness as success
    assert not ok
    assert len(calls) == 2
