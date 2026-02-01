"""Tests for RTSPSource helper methods."""

from __future__ import annotations

from pathlib import Path

from homesec.sources.rtsp.core import RTSPSource, RTSPSourceConfig


def _make_source(tmp_path: Path) -> RTSPSource:
    config = RTSPSourceConfig(
        rtsp_url="rtsp://user:pass@camera/stream?subtype=0",
        output_dir=str(tmp_path),
        stream={"disable_hwaccel": True},
    )
    return RTSPSource(config, camera_name="Front Door #1")


def test_sanitize_camera_name(tmp_path: Path) -> None:
    """Sanitize camera names into safe identifiers."""
    # Given an RTSPSource instance
    source = _make_source(tmp_path)

    # When sanitizing a name with spaces and symbols
    cleaned = source._sanitize_camera_name("Front Door #1")

    # Then the name is normalized
    assert cleaned == "Front_Door_1"


def test_derive_detect_rtsp_url(tmp_path: Path) -> None:
    """Derive a detect stream URL from subtype=0 URLs."""
    # Given an RTSPSource instance
    source = _make_source(tmp_path)

    # When deriving a detect URL from subtype=0
    derived = source._derive_detect_rtsp_url("rtsp://host/stream?subtype=0")

    # Then subtype=1 is returned
    assert derived == "rtsp://host/stream?subtype=1"


def test_derive_detect_rtsp_url_no_change(tmp_path: Path) -> None:
    """Return None when subtype=0 is not present."""
    # Given an RTSPSource instance
    source = _make_source(tmp_path)

    # When deriving from a URL without subtype=0
    derived = source._derive_detect_rtsp_url("rtsp://host/stream")

    # Then no derived URL is produced
    assert derived is None


def test_normalize_blur_kernel(tmp_path: Path) -> None:
    """Normalize blur kernel to odd or zero values."""
    # Given an RTSPSource instance
    source = _make_source(tmp_path)

    # When normalizing even and negative values
    even = source._normalize_blur_kernel(4)
    negative = source._normalize_blur_kernel(-1)

    # Then even increments and negative clamps to zero
    assert even == 5
    assert negative == 0


def test_redact_rtsp_url_credentials(tmp_path: Path) -> None:
    """Redact credentials in RTSP URLs."""
    # Given an RTSPSource instance
    source = _make_source(tmp_path)

    # When redacting a URL with credentials
    redacted = source._redact_rtsp_url("rtsp://user:pass@host/stream")

    # Then credentials are hidden
    assert redacted == "rtsp://***:***@host/stream"
