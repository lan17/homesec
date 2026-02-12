"""Tests for RTSP timeout capability tracking."""

from __future__ import annotations

from homesec.sources.rtsp.capabilities import (
    RTSPTimeoutCapabilities,
    TimeoutOptionSupport,
)


def test_ffmpeg_timeout_args_disabled_after_unsupported_marker() -> None:
    """ffmpeg timeout flags should stop once unsupported is observed."""
    # Given: unknown timeout-option support
    capabilities = RTSPTimeoutCapabilities()

    # When: timeout args are requested before and after unsupported detection
    initial_args = capabilities.build_ffmpeg_timeout_args(connect_timeout_s=2.0, io_timeout_s=2.0)
    changed = capabilities.mark_ffmpeg_timeout_unsupported()
    after_args = capabilities.build_ffmpeg_timeout_args(connect_timeout_s=2.0, io_timeout_s=2.0)

    # Then: timeout args are initially present and then disabled
    assert changed
    assert "-stimeout" in initial_args
    assert "-rw_timeout" in initial_args
    assert after_args == []


def test_ffprobe_timeout_support_is_tracked_independently() -> None:
    """ffprobe timeout capability should be tracked independently from ffmpeg."""
    # Given: unknown timeout-option support for both tools
    capabilities = RTSPTimeoutCapabilities()

    # When: ffmpeg is marked unsupported while ffprobe marks success
    _ = capabilities.mark_ffmpeg_timeout_unsupported()
    capabilities.note_ffprobe_timeout_success()
    ffmpeg_state, ffprobe_state = capabilities.snapshot()

    # Then: each tool keeps its own support state
    assert ffmpeg_state == TimeoutOptionSupport.UNSUPPORTED
    assert ffprobe_state == TimeoutOptionSupport.SUPPORTED
