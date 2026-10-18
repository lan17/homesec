"""Tests for RTSP startup preflight orchestration."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from homesec.sources.rtsp.discovery import CameraProbeResult, ProbeStreamInfo
from homesec.sources.rtsp.preflight import (
    ConcurrentStreamOpenResult,
    PreflightError,
    RecordingValidationResult,
    RecordingValidationSignals,
    RTSPStartupPreflight,
)
from homesec.sources.rtsp.recording_profile import RecordingProfile
from tests.homesec.inspection_helpers import inspection_helper, inspection_requests


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


def test_preflight_classifies_concurrent_preview_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight should validate the motion, recording, and preview topology on request."""
    # Given: dual stream validation succeeds and preview is allowed during recording
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
    preview_checks: list[tuple[str, str, str, bool]] = []

    def _validate_profile(_profile: RecordingProfile) -> RecordingValidationResult:
        return RecordingValidationResult(ok=True)

    def _validate_preview(
        motion_url: str,
        recording_url: str,
        preview_url: str,
        *,
        preview_audio_enabled: bool,
    ) -> ConcurrentStreamOpenResult:
        preview_checks.append((motion_url, recording_url, preview_url, preview_audio_enabled))
        return ConcurrentStreamOpenResult.SUPPORTED

    monkeypatch.setattr(preflight, "_validate_recording_profile", _validate_profile)
    monkeypatch.setattr(preflight, "_validate_session_limits", lambda _m, _r: True)
    monkeypatch.setattr(
        preflight,
        "_validate_concurrent_preview_session_limits",
        _validate_preview,
    )

    # When: running startup preflight with the actual preview input URL requested
    outcome = preflight.run(
        camera_name="garage",
        primary_rtsp_url="rtsp://cam/high",
        detect_rtsp_url="rtsp://cam/low",
        preview_rtsp_url="rtsp://cam/high",
        preview_audio_enabled=True,
    )

    # Then: the preview check uses motion low, recording high, preview high, and preview audio
    assert not isinstance(outcome, PreflightError)
    assert outcome.concurrent_preview_supported is True
    assert outcome.concurrent_preview_downgrade_reason is None
    assert outcome.diagnostics.selected_preview_url == "rtsp://cam/high"
    assert preview_checks == [("rtsp://cam/low", "rtsp://cam/high", "rtsp://cam/high", True)]


def test_preflight_validates_preview_startup_probe_when_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight should validate the preview startup probe used by auto codecs."""
    # Given: concurrent preview is requested and preview startup still needs stream probing
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
    probe_checks: list[tuple[str, str, str]] = []
    preview_checks: list[tuple[str, str, str, bool]] = []

    def _validate_profile(_profile: RecordingProfile) -> RecordingValidationResult:
        return RecordingValidationResult(ok=True)

    def _validate_probe(
        motion_url: str, recording_url: str, preview_probe_url: str
    ) -> ConcurrentStreamOpenResult:
        probe_checks.append((motion_url, recording_url, preview_probe_url))
        return ConcurrentStreamOpenResult.SUPPORTED

    def _validate_preview(
        motion_url: str,
        recording_url: str,
        preview_url: str,
        *,
        preview_audio_enabled: bool,
    ) -> ConcurrentStreamOpenResult:
        preview_checks.append((motion_url, recording_url, preview_url, preview_audio_enabled))
        return ConcurrentStreamOpenResult.SUPPORTED

    monkeypatch.setattr(preflight, "_validate_recording_profile", _validate_profile)
    monkeypatch.setattr(preflight, "_validate_session_limits", lambda _m, _r: True)
    monkeypatch.setattr(
        preflight,
        "_validate_concurrent_preview_probe_session_limits",
        _validate_probe,
    )
    monkeypatch.setattr(
        preflight,
        "_validate_concurrent_preview_session_limits",
        _validate_preview,
    )

    # When: running startup preflight with the preview probe URL requested
    outcome = preflight.run(
        camera_name="garage",
        primary_rtsp_url="rtsp://cam/high",
        detect_rtsp_url="rtsp://cam/low",
        preview_rtsp_url="rtsp://cam/high",
        preview_probe_rtsp_url="rtsp://cam/high",
        preview_audio_enabled=True,
    )

    # Then: the probe and preview checks both validate the current runtime topology
    assert not isinstance(outcome, PreflightError)
    assert outcome.concurrent_preview_supported is True
    assert probe_checks == [("rtsp://cam/low", "rtsp://cam/high", "rtsp://cam/high")]
    assert preview_checks == [("rtsp://cam/low", "rtsp://cam/high", "rtsp://cam/high", True)]
    assert outcome.diagnostics.selected_preview_probe_url == "rtsp://cam/high"


def test_preflight_continues_preview_classification_after_probe_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe-only failures should not skip the actual preview stream capability check."""
    # Given: startup probe classification is generic but the runtime preview stream hits a session limit
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
    preview_checks: list[tuple[str, str, str, bool]] = []

    def _validate_profile(_profile: RecordingProfile) -> RecordingValidationResult:
        return RecordingValidationResult(ok=True)

    def _validate_preview(
        motion_url: str,
        recording_url: str,
        preview_url: str,
        *,
        preview_audio_enabled: bool,
    ) -> ConcurrentStreamOpenResult:
        preview_checks.append((motion_url, recording_url, preview_url, preview_audio_enabled))
        return ConcurrentStreamOpenResult.SESSION_LIMIT

    monkeypatch.setattr(preflight, "_validate_recording_profile", _validate_profile)
    monkeypatch.setattr(preflight, "_validate_session_limits", lambda _m, _r: True)
    monkeypatch.setattr(
        preflight,
        "_validate_concurrent_preview_probe_session_limits",
        lambda _m, _r, _p: ConcurrentStreamOpenResult.FAILED,
    )
    monkeypatch.setattr(
        preflight,
        "_validate_concurrent_preview_session_limits",
        _validate_preview,
    )

    # When: running startup preflight with both auto-codec probe and preview validation requested
    outcome = preflight.run(
        camera_name="garage",
        primary_rtsp_url="rtsp://cam/high",
        detect_rtsp_url="rtsp://cam/low",
        preview_rtsp_url="rtsp://cam/high",
        preview_probe_rtsp_url="rtsp://cam/high",
        preview_audio_enabled=True,
    )

    # Then: the actual preview stream check still classifies the process-local downgrade
    assert not isinstance(outcome, PreflightError)
    assert preview_checks == [("rtsp://cam/low", "rtsp://cam/high", "rtsp://cam/high", True)]
    assert outcome.concurrent_preview_supported is False
    assert (
        outcome.concurrent_preview_downgrade_reason
        == "concurrent_preview_unsupported_by_startup_preflight"
    )


def test_preflight_downgrades_concurrent_preview_when_three_stream_check_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight should mark concurrent preview unsupported without failing startup."""
    # Given: motion plus recording works but the three-stream preview check fails
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
    monkeypatch.setattr(
        preflight,
        "_validate_concurrent_preview_session_limits",
        lambda _m, _r, _p, **_kwargs: ConcurrentStreamOpenResult.SESSION_LIMIT,
    )

    # When: running startup preflight with concurrent preview requested
    outcome = preflight.run(
        camera_name="garage",
        primary_rtsp_url="rtsp://cam/high",
        detect_rtsp_url="rtsp://cam/low",
        preview_rtsp_url="rtsp://cam/high",
    )

    # Then: startup succeeds while the process-local preview downgrade is classified
    assert not isinstance(outcome, PreflightError)
    assert outcome.concurrent_preview_supported is False
    assert (
        outcome.concurrent_preview_downgrade_reason
        == "concurrent_preview_unsupported_by_startup_preflight"
    )
    assert outcome.diagnostics.concurrent_preview_supported is False
    assert outcome.diagnostics.concurrent_preview_downgrade_reason == (
        "concurrent_preview_unsupported_by_startup_preflight"
    )


def test_preflight_leaves_concurrent_preview_unclassified_on_generic_three_stream_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight should not downgrade preview concurrency on transient validation failures."""
    # Given: motion plus recording works but the preview classification has a generic failure
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
    monkeypatch.setattr(
        preflight,
        "_validate_concurrent_preview_session_limits",
        lambda _m, _r, _p, **_kwargs: ConcurrentStreamOpenResult.FAILED,
    )

    # When: running startup preflight with concurrent preview requested
    outcome = preflight.run(
        camera_name="garage",
        primary_rtsp_url="rtsp://cam/high",
        detect_rtsp_url="rtsp://cam/low",
        preview_rtsp_url="rtsp://cam/high",
    )

    # Then: startup succeeds and preview concurrency remains process-local/unclassified
    assert not isinstance(outcome, PreflightError)
    assert outcome.concurrent_preview_supported is None
    assert outcome.concurrent_preview_downgrade_reason is None
    assert outcome.diagnostics.concurrent_preview_supported is None
    assert outcome.diagnostics.concurrent_preview_downgrade_reason is None


def test_preflight_skips_three_stream_check_when_preview_not_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight should not run preview capability checks unless explicitly requested."""
    # Given: a viable RTSP camera where preview concurrency was not requested
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

    def _validate_profile(_profile: RecordingProfile) -> RecordingValidationResult:
        return RecordingValidationResult(ok=True)

    def _unexpected_preview_check(
        _motion_url: str,
        _recording_url: str,
        _preview_url: str,
        **_kwargs: object,
    ) -> ConcurrentStreamOpenResult:
        raise AssertionError("three-stream preview check should not run")

    monkeypatch.setattr(preflight, "_validate_recording_profile", _validate_profile)
    monkeypatch.setattr(preflight, "_validate_session_limits", lambda _m, _r: True)
    monkeypatch.setattr(
        preflight,
        "_validate_concurrent_preview_session_limits",
        _unexpected_preview_check,
    )

    # When: running startup preflight without a preview input URL
    outcome = preflight.run(
        camera_name="garage",
        primary_rtsp_url="rtsp://cam/high",
        detect_rtsp_url="rtsp://cam/high",
    )

    # Then: preview capability remains unclassified for this startup run
    assert not isinstance(outcome, PreflightError)
    assert outcome.concurrent_preview_supported is None
    assert outcome.diagnostics.concurrent_preview_supported is None
    assert outcome.diagnostics.selected_preview_url is None


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


def test_native_trials_and_overlap_preserve_independent_roles(tmp_path: Path) -> None:
    # Given: an executable native protocol peer and equal URLs for checked roles
    helper = inspection_helper(tmp_path)
    preflight = RTSPStartupPreflight(
        output_dir=tmp_path, rtsp_connect_timeout_s=2, rtsp_io_timeout_s=2, helper_path=str(helper)
    )
    # When: recording and preview negotiation succeeds
    outcome = preflight.run(
        camera_name="test",
        primary_rtsp_url="rtsp://camera/live",
        detect_rtsp_url="rtsp://camera/live",
        preview_rtsp_url="rtsp://camera/live",
        preview_probe_rtsp_url="rtsp://camera/live",
        preview_audio_enabled=True,
    )
    # Then: no URL equality collapses conservative independent checks and trials are removed
    assert not isinstance(outcome, PreflightError)
    assert outcome.recording_profile.audio_mode == "copy"
    requests = inspection_requests(tmp_path)
    overlap = [request for request in requests if request["command"] == "inspect_overlap"]
    assert [len(request["roles"]) for request in overlap] == [2, 3, 3]
    assert overlap[1]["roles"][2]["kind"] == "metadata"
    assert overlap[2]["roles"][2]["audio_mode"] == "copy"
    for request in requests:
        if request["command"] == "inspect_recording":
            assert not Path(request["output_path"]).exists()


def test_native_missing_aac_capability_cannot_silently_remove_audio(tmp_path: Path) -> None:
    # Given: conversion support is missing in the linked build
    helper = inspection_helper(
        tmp_path,
        behavior="if command == 'inspect_recording': error = 'local_capability_unavailable'",
    )
    preflight = RTSPStartupPreflight(
        output_dir=tmp_path, rtsp_connect_timeout_s=2, rtsp_io_timeout_s=2, helper_path=str(helper)
    )
    # When: preflight attempts audio negotiation
    outcome = preflight.run(
        camera_name="test",
        primary_rtsp_url="rtsp://camera/live",
        detect_rtsp_url="rtsp://camera/live",
    )
    # Then: this is an explicit local delivery failure, with no silent candidate attempted
    assert isinstance(outcome, PreflightError)
    assert "local_capability_unavailable" in outcome.message
    trials = [
        request
        for request in inspection_requests(tmp_path)
        if request["command"] == "inspect_recording"
    ]
    assert len(trials) == 1
    assert trials[0]["audio_mode"] == "copy"


def test_native_header_only_trial_cannot_pass(tmp_path: Path) -> None:
    # Given: a helper reports a finalized nonempty container with no video packets
    helper = inspection_helper(
        tmp_path,
        behavior="if command == 'inspect_recording':\n    pathlib.Path(request['output_path']).write_bytes(b'header')\n    inspection = dict(kind='recording',finalized=True,video_packets=0,audio_packets=1,output_bytes=6,signals={})",
    )
    preflight = RTSPStartupPreflight(
        output_dir=tmp_path, rtsp_connect_timeout_s=2, rtsp_io_timeout_s=2, helper_path=str(helper)
    )
    # When: negotiating recording profiles
    outcome = preflight.run(
        camera_name="test",
        primary_rtsp_url="rtsp://camera/live",
        detect_rtsp_url="rtsp://camera/live",
    )
    # Then: empty-media output is rejected despite a successful helper reply
    assert isinstance(outcome, PreflightError)
    assert outcome.stage == "negotiation"


def test_startup_budget_allows_multiple_operations_beyond_one_command_cap(tmp_path: Path) -> None:
    # Given: several healthy operations each finish within the one-second operation cap
    helper = inspection_helper(tmp_path, behavior="time.sleep(.3)")
    preflight = RTSPStartupPreflight(
        output_dir=tmp_path,
        rtsp_connect_timeout_s=1,
        rtsp_io_timeout_s=1,
        command_timeout_s=1,
        helper_path=str(helper),
    )
    # When: the finite discovery/trial/overlap plan exceeds a single operation's duration
    outcome = preflight.run(
        camera_name="test",
        primary_rtsp_url="rtsp://camera/stream1",
        detect_rtsp_url="rtsp://camera/stream2",
        preview_rtsp_url="rtsp://camera/stream1",
    )
    # Then: startup completes under the derived plan budget, preserving per-operation caps
    assert not isinstance(outcome, PreflightError)
    requests = inspection_requests(tmp_path)
    metadata = [request for request in requests if request["command"] == "inspect_metadata"]
    assert len(metadata) >= 2
    assert all(request["operation_timeout_s"] <= 1 for request in metadata)
    assert sum(request["command"] != "stop" for request in requests) * 0.3 > 1


def test_native_failed_audio_trial_retains_conversion_and_silent_order(tmp_path: Path) -> None:
    # Given: genuine camera/media profile failures rather than missing build capabilities
    helper = inspection_helper(
        tmp_path,
        behavior="if command == 'inspect_recording' and request['audio_mode'] != 'none': error = 'profile_validation_failed'",
    )
    preflight = RTSPStartupPreflight(
        output_dir=tmp_path, rtsp_connect_timeout_s=2, rtsp_io_timeout_s=2, helper_path=str(helper)
    )
    # When: profiles are negotiated
    outcome = preflight.run(
        camera_name="test",
        primary_rtsp_url="rtsp://camera/live",
        detect_rtsp_url="rtsp://camera/live",
    )
    # Then: copy, conversion, and silent candidates retain their established ordering
    assert not isinstance(outcome, PreflightError)
    assert outcome.recording_profile.audio_mode == "none"
    assert [
        request["audio_mode"]
        for request in inspection_requests(tmp_path)
        if request["command"] == "inspect_recording"
    ] == ["copy", "aac", "none"]


@pytest.mark.parametrize("retry", ["better", "same", "failed"])
def test_native_timestamp_retry_retains_only_the_more_stable_success(
    tmp_path: Path, retry: str
) -> None:
    # Given: native packet timing signals cross the existing baseline retry threshold
    helper = inspection_helper(
        tmp_path,
        behavior=(
            "if command == 'inspect_recording':\n"
            "    pathlib.Path(request['output_path']).write_bytes(b'media')\n"
            "    wallclock = request['timestamp_mode'] == 'wallclock'\n"
            f"    if wallclock and {retry!r} == 'failed': error = 'profile_validation_failed'\n"
            "    else:\n"
            f"        score = 0 if wallclock and {retry!r} == 'better' else 8\n"
            "        inspection = dict(kind='recording',finalized=True,video_packets=5,audio_packets=3,output_bytes=5,signals=dict(non_monotonic_dts=score))"
        ),
    )
    preflight = RTSPStartupPreflight(
        output_dir=tmp_path, rtsp_connect_timeout_s=2, rtsp_io_timeout_s=2, helper_path=str(helper)
    )
    # When: Python compares the bounded native timing evidence
    outcome = preflight.run(
        camera_name="test",
        primary_rtsp_url="rtsp://camera/live",
        detect_rtsp_url="rtsp://camera/live",
    )
    # Then: only a successful strictly better retry changes timestamp mode
    assert not isinstance(outcome, PreflightError)
    assert outcome.recording_profile.uses_wallclock_timestamps() == (retry == "better")
    assert [
        request["timestamp_mode"]
        for request in inspection_requests(tmp_path)
        if request["command"] == "inspect_recording"
    ] == ["stream", "wallclock"]


@pytest.mark.parametrize(
    "failure,denied",
    [
        ("session_limit", True),
        ("authentication_failed", False),
        ("connection_failed", False),
        ("timeout", False),
    ],
)
@pytest.mark.parametrize("metadata_role", [True, False])
def test_native_overlap_only_explicit_capacity_rejection_downgrades_preview(
    tmp_path: Path, failure: str, denied: bool, metadata_role: bool
) -> None:
    # Given: the optional metadata or preview role fails during conservative overlap
    selected_role = "preview_probe" if metadata_role else "preview"
    helper = inspection_helper(
        tmp_path,
        behavior=f"if command == 'inspect_overlap' and any(role['role'] == {selected_role!r} for role in request['roles']):\n    inspection = dict(kind='overlap',status={'session_limit' if denied else 'failed'!r},roles=[dict(role=role['role'],ok=False,error_code={failure!r},frames_decoded=0,video_packets=0,audio_packets=0) for role in request['roles']])",
    )
    preflight = RTSPStartupPreflight(
        output_dir=tmp_path, rtsp_connect_timeout_s=2, rtsp_io_timeout_s=2, helper_path=str(helper)
    )
    # When: startup classifies preview coexistence with recording
    outcome = preflight.run(
        camera_name="test",
        primary_rtsp_url="rtsp://camera/live",
        detect_rtsp_url="rtsp://camera/live",
        preview_rtsp_url="rtsp://camera/live",
        preview_probe_rtsp_url="rtsp://camera/live" if metadata_role else None,
    )
    # Then: only explicit session denial selects recording-first behavior
    assert not isinstance(outcome, PreflightError)
    assert outcome.concurrent_preview_supported is (
        False if denied else (True if metadata_role else None)
    )
    assert (outcome.concurrent_preview_downgrade_reason is not None) == denied


def test_preflight_credentials_tokens_and_private_paths_stay_outside_logs_and_argv(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: camera secrets are supplied through the private control pipe
    helper = inspection_helper(tmp_path)
    real_popen = subprocess.Popen
    args_seen: list[str] = []

    def capture(args: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
        args_seen.extend(args)
        return real_popen(args, **kwargs)

    monkeypatch.setattr("homesec.sources.rtsp.helper_client.subprocess.Popen", capture)
    url = "rtsp://secret-user:secret-password@camera/secret-path?token=secret-token"
    preflight = RTSPStartupPreflight(
        output_dir=tmp_path, rtsp_connect_timeout_s=2, rtsp_io_timeout_s=2, helper_path=str(helper)
    )
    # When: successful native startup selects a recording profile
    with caplog.at_level("DEBUG"):
        outcome = preflight.run(camera_name="test", primary_rtsp_url=url, detect_rtsp_url=url)
    # Then: process arguments and diagnostics logs contain no camera URL fragments
    assert not isinstance(outcome, PreflightError)
    for secret in ("secret-user", "secret-password", "secret-path", "secret-token"):
        assert secret not in caplog.text
        assert secret not in " ".join(args_seen)
    assert args_seen == [str(helper), "--inspect"]


def test_native_overlap_local_capability_failure_cannot_become_session_defaults(
    tmp_path: Path,
) -> None:
    # Given: metadata/trial work, but a required overlap decoder is missing locally
    helper = inspection_helper(
        tmp_path,
        behavior="if command == 'inspect_overlap': inspection = dict(kind='overlap',status='failed',roles=[dict(role=role['role'],ok=False,error_code='local_capability_unavailable',frames_decoded=0,video_packets=0,audio_packets=0) for role in request['roles']])",
    )
    preflight = RTSPStartupPreflight(
        output_dir=tmp_path, rtsp_connect_timeout_s=2, rtsp_io_timeout_s=2, helper_path=str(helper)
    )
    # When: native overlap validation cannot perform a required operation
    outcome = preflight.run(
        camera_name="test",
        primary_rtsp_url="rtsp://camera/live",
        detect_rtsp_url="rtsp://camera/live",
    )
    # Then: the incomplete local build is reported without a camera-capacity fallback
    assert isinstance(outcome, PreflightError)
    assert outcome.stage == "inspection"
    assert "local_capability_unavailable" in outcome.message
