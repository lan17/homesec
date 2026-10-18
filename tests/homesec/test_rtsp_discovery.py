"""Native metadata protocol preserves discovery policy without media CLI tools."""

from pathlib import Path

from homesec.sources.rtsp.discovery import (
    CameraProbeResult,
    NativeStreamDiscovery,
    build_camera_key,
)
from tests.homesec.inspection_helpers import inspection_helper, inspection_requests


def test_build_camera_key_normalizes_inputs() -> None:
    # Given: mixed case and URL credentials
    url = "rtsp://User:Pass@LENOVO:554/Streaming/Channels/101?subtype=0"
    # When: resolving the stable camera key
    key = build_camera_key("Garage Cam #1", url)
    # Then: no secret is included
    assert key == "garage_cam_1:lenovo:streaming/channels/101"


def test_probe_preserves_order_partial_results_and_codec_metadata(tmp_path: Path) -> None:
    # Given: independent native metadata succeeds for HEVC and rejects one candidate
    helper = inspection_helper(
        tmp_path,
        behavior="if request.get('rtsp_url') == 'rtsp://bad': error = 'authentication_failed'",
    )
    discovery = NativeStreamDiscovery(
        helper_path=str(helper),
        rtsp_connect_timeout_s=2.0,
        rtsp_io_timeout_s=2.0,
    )
    # When: discovering repeated candidates
    result = discovery.probe(
        camera_key="camera",
        candidate_urls=["rtsp://good", "rtsp://bad", "rtsp://good", "rtsp://last"],
    )
    # Then: original order, first audio/video metadata, and partial failures survive
    assert isinstance(result, CameraProbeResult)
    assert result.attempted_urls == ["rtsp://good", "rtsp://bad", "rtsp://last"]
    assert [stream.probe_ok for stream in result.streams] == [True, False, True]
    assert result.streams[0].video_codec == "hevc"
    assert result.streams[0].audio_codec == "aac"
    assert result.streams[0].fps == 15.0
    assert result.streams[0].fps_raw == "15/1"
    assert result.streams[1].error == "Native inspection: authentication_failed"
    requests = inspection_requests(tmp_path)
    assert [
        request["rtsp_url"] for request in requests if request["command"] == "inspect_metadata"
    ] == result.attempted_urls


def test_missing_frame_rate_remains_absent(tmp_path: Path) -> None:
    # Given: dimensions and codec are available without usable frame-rate metadata
    helper = inspection_helper(
        tmp_path,
        behavior="inspection = dict(kind='metadata', video_codec='mpeg4', width=320, height=240)",
    )
    # When: metadata discovery completes
    result = NativeStreamDiscovery(
        helper_path=str(helper), rtsp_connect_timeout_s=1, rtsp_io_timeout_s=1
    ).probe(camera_key="camera", candidate_urls=["rtsp://x"])
    # Then: discovery does not invent frame-rate or gate on H.264 eligibility
    assert isinstance(result, CameraProbeResult)
    assert result.streams[0].probe_ok
    assert result.streams[0].fps is None
    assert result.streams[0].fps_raw is None


def test_invalid_metadata_and_malicious_errors_fail_safely(tmp_path: Path) -> None:
    # Given: a helper returns an untrusted error which might contain credentials
    helper = inspection_helper(tmp_path, behavior="error = 'rtsp://user:secret@camera' ")
    # When: probing the camera
    result = NativeStreamDiscovery(
        helper_path=str(helper), rtsp_connect_timeout_s=1, rtsp_io_timeout_s=1
    ).probe(camera_key="camera", candidate_urls=["rtsp://x"])
    # Then: only bounded stable failure detail is returned
    assert isinstance(result, CameraProbeResult)
    assert result.streams[0].error == "Native inspection: helper_failure"
    assert "secret" not in result.model_dump_json()
