"""Tests for hardware acceleration detection."""

from __future__ import annotations

import subprocess

import pytest

from homesec.sources.rtsp.hardware import HardwareAccelConfig, HardwareAccelDetector


def test_test_hwaccel_returns_false_when_ffmpeg_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing ffmpeg should disable hwaccel detection."""

    # Given: ffmpeg is not available
    def _missing(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("ffmpeg not found")

    monkeypatch.setattr("homesec.sources.rtsp.hardware.subprocess.run", _missing)

    # When: checking for hwaccel support
    ok = HardwareAccelDetector._test_hwaccel("vaapi")

    # Then: hwaccel is unavailable
    assert not ok


def test_test_decode_returns_false_on_known_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known decode errors should return False."""

    # Given: ffmpeg returns a known decode error
    def _run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["ffmpeg"],
            returncode=1,
            stdout="",
            stderr="No VA display found",
        )

    monkeypatch.setattr("homesec.sources.rtsp.hardware.subprocess.run", _run)
    config = HardwareAccelConfig(hwaccel="vaapi", hwaccel_device="/dev/dri/renderD128")

    # When: testing decode
    ok = HardwareAccelDetector._test_decode("rtsp://host/stream", config)

    # Then: decode fails
    assert not ok


def test_test_decode_treats_timeout_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeouts should be treated as a successful decode check."""

    # Given: ffmpeg times out
    def _timeout(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=10)

    monkeypatch.setattr("homesec.sources.rtsp.hardware.subprocess.run", _timeout)
    config = HardwareAccelConfig(hwaccel="cuda")

    # When: testing decode
    ok = HardwareAccelDetector._test_decode("rtsp://host/stream", config)

    # Then: decode is treated as successful
    assert ok


def test_check_nvidia_returns_false_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NVIDIA check should return False when nvidia-smi fails."""

    # Given: nvidia-smi fails
    def _fail(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(returncode=1, cmd="nvidia-smi")

    monkeypatch.setattr("homesec.sources.rtsp.hardware.subprocess.run", _fail)

    # When: checking for NVIDIA availability
    ok = HardwareAccelDetector._check_nvidia()

    # Then: NVIDIA is unavailable
    assert not ok


def test_detect_returns_software_when_no_accel_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detector should fall back to software when no accel works."""
    # Given: no hwaccel methods are available
    monkeypatch.setattr("homesec.sources.rtsp.hardware.Path.exists", lambda _p: False)
    monkeypatch.setattr("homesec.sources.rtsp.hardware.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "homesec.sources.rtsp.hardware.HardwareAccelDetector._check_nvidia",
        lambda: False,
    )
    monkeypatch.setattr(
        "homesec.sources.rtsp.hardware.HardwareAccelDetector._test_hwaccel",
        lambda _method: False,
    )

    # When: detecting hardware accel
    config = HardwareAccelDetector.detect("rtsp://host/stream")

    # Then: software decode is selected
    assert config.hwaccel is None
