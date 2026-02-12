"""Tests for RTSP recording profile candidate generation."""

from __future__ import annotations

from homesec.sources.rtsp.recording_profile import build_recording_profile_candidates


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
