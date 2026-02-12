"""Tests for RTSP startup preflight orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from homesec.sources.rtsp.discovery import CameraProbeResult, ProbeStreamInfo
from homesec.sources.rtsp.preflight import PreflightError, RTSPStartupPreflight
from homesec.sources.rtsp.recording_profile import RecordingProfile


class _FakeDiscovery:
    def __init__(self, streams: list[ProbeStreamInfo]) -> None:
        self._streams = streams

    def probe(self, *, camera_key: str, candidate_urls: list[str]) -> CameraProbeResult:
        return CameraProbeResult(
            camera_key=camera_key,
            streams=list(self._streams),
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

    def _validate_profile(_profile: RecordingProfile) -> tuple[bool, str | None]:
        return True, None

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

    def _validate_profile(_profile: RecordingProfile) -> tuple[bool, str | None]:
        return True, None

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

    def _validate_profile(_profile: RecordingProfile) -> tuple[bool, str | None]:
        return False, "simulated ffmpeg failure"

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
