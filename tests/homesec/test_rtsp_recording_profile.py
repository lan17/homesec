"""Tests for RTSP recording profile candidate generation."""

from __future__ import annotations

from homesec.sources.rtsp.recording_profile import (
    build_default_recording_profile,
    build_recording_profile_candidates,
)


def test_candidates_skip_audio_copy_for_incompatible_codec() -> None:
    """Incompatible audio codecs should not produce mp4 audio-copy candidate."""
    # Given: an RTSP stream with pcm_alaw audio
    input_url = "rtsp://camera/garage"

    # When: building recording profile candidates
    candidates = build_recording_profile_candidates(input_url=input_url, audio_codec="pcm_alaw")

    # Then: copy-audio is skipped and fallback modes remain
    assert [candidate.audio_mode for candidate in candidates] == ["aac", "none"]


def test_candidates_prefer_audio_copy_for_compatible_codec() -> None:
    """MP4-safe audio codecs should keep copy-audio as first candidate."""
    # Given: an RTSP stream with AAC audio
    input_url = "rtsp://camera/front"

    # When: building recording profile candidates
    candidates = build_recording_profile_candidates(input_url=input_url, audio_codec="aac")

    # Then: copy-audio is first, then transcode fallback, then no-audio fallback
    assert [candidate.audio_mode for candidate in candidates] == ["copy", "aac", "none"]


def test_candidates_include_passthrough_vsync_defaults() -> None:
    """All generated recording profiles should default to passthrough timing mode."""
    # Given: a stream with audio where all output profiles are generated
    candidates = build_recording_profile_candidates(
        input_url="rtsp://camera/front",
        audio_codec="aac",
    )

    # When/Then: each candidate includes -vsync 0
    for candidate in candidates:
        assert "-vsync" in candidate.ffmpeg_output_args
        idx = candidate.ffmpeg_output_args.index("-vsync")
        assert candidate.ffmpeg_output_args[idx + 1] == "0"


def test_default_profile_includes_passthrough_vsync_defaults() -> None:
    """Default recording profile should also preserve passthrough timing mode."""
    # Given: default profile selection before preflight
    profile = build_default_recording_profile("rtsp://camera/front")

    # Then: default output args include -vsync 0
    assert "-vsync" in profile.ffmpeg_output_args
    idx = profile.ffmpeg_output_args.index("-vsync")
    assert profile.ffmpeg_output_args[idx + 1] == "0"
