"""Native device availability remains distinct from actual decode evidence."""

from pathlib import Path

import pytest
from tests.homesec.inspection_helpers import inspection_helper, inspection_requests

from homesec.sources.rtsp.hardware import HardwareAccelConfig, HardwareAccelDetector


def test_native_hardware_requires_actual_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a listed device whose decoder produced no frames
    helper = inspection_helper(
        tmp_path,
        behavior="if command == 'inspect_decode': inspection = dict(kind='decode',backend=request['backend'],frames_decoded=0)",
    )
    monkeypatch.setattr("homesec.sources.rtsp.hardware.platform.system", lambda: "Darwin")
    # When: selecting native hardware
    config = HardwareAccelDetector.detect("rtsp://camera/live", helper_path=str(helper))
    # Then: enumeration does not establish working decode
    assert not config.is_available
    assert any(request["command"] == "inspect_decode" for request in inspection_requests(tmp_path))


def test_native_hardware_requires_requested_backend(tmp_path: Path) -> None:
    # Given: a helper incorrectly claims software frames for a hardware trial
    helper = inspection_helper(
        tmp_path,
        behavior="if command == 'inspect_decode': inspection = dict(kind='decode',backend='software',frames_decoded=5)",
    )
    # When: selecting native hardware
    config = HardwareAccelDetector.detect("rtsp://camera/live", helper_path=str(helper))
    # Then: hidden software fallback does not pass hardware validation
    assert not config.is_available


def test_native_hardware_accepts_five_frames_on_requested_backend(tmp_path: Path) -> None:
    # Given: available hardware and actual frame evidence
    helper = inspection_helper(tmp_path)
    # When: selecting hardware from the native inspector
    config = HardwareAccelDetector.detect("rtsp://camera/live", helper_path=str(helper))
    # Then: hardware is used only after the decoded sample completed
    assert config.is_available
    assert config.hwaccel == inspection_requests(tmp_path)[1]["backend"]


def test_native_hardware_timeout_does_not_pass(tmp_path: Path) -> None:
    # Given: a device decode operation times out
    helper = inspection_helper(
        tmp_path, behavior="if command == 'inspect_decode': error = 'timeout'"
    )
    # When: selecting native hardware
    config = HardwareAccelDetector.detect("rtsp://camera/live", helper_path=str(helper))
    # Then: timeout has no frame evidence and software is selected
    assert not config.is_available


@pytest.mark.parametrize("backend,encoder", [("videotoolbox", "h264_videotoolbox"), ("cuda", None)])
def test_encoder_selection_uses_linked_build_capabilities(
    tmp_path: Path, backend: str, encoder: str | None
) -> None:
    # Given: the linked build advertises only VideoToolbox's H.264 encoder
    helper = inspection_helper(tmp_path)
    # When: choosing the matching compatible execution encoder
    result = HardwareAccelDetector.select_h264_encoder(
        HardwareAccelConfig(hwaccel=backend), helper_path=str(helper)
    )
    # Then: encoder availability comes from the native build
    assert (result.ffmpeg_encoder if result else None) == encoder
