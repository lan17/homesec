"""Tests for RTSP ffprobe-based stream discovery."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from homesec.sources.rtsp.discovery import FfprobeStreamDiscovery, build_camera_key


class _FakeResult:
    def __init__(self, *, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_build_camera_key_normalizes_inputs() -> None:
    """Camera key should normalize camera name, host, and path."""
    # Given: camera metadata with mixed case and punctuation
    camera_name = "Garage Cam #1"
    rtsp_url = "rtsp://User:Pass@LENOVO:554/Streaming/Channels/101?subtype=0"

    # When: building the stable camera key
    camera_key = build_camera_key(camera_name, rtsp_url)

    # Then: key contains normalized name, host, and path
    assert camera_key == "garage_cam_1:lenovo:streaming/channels/101"


def test_probe_parses_video_and_audio_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Discovery should parse codec, dimensions, and fps from ffprobe output."""
    # Given: ffprobe returns one video and one audio stream
    payload = (
        '{"streams":['
        '{"codec_type":"video","codec_name":"h264","width":1280,'
        '"height":720,"avg_frame_rate":"15/1"},'
        '{"codec_type":"audio","codec_name":"aac"}'
        "]}"
    )

    def _fake_run(_cmd: Sequence[str], **_kwargs: object) -> _FakeResult:
        return _FakeResult(returncode=0, stdout=payload)

    monkeypatch.setattr("homesec.sources.rtsp.discovery.subprocess.run", _fake_run)
    discovery = FfprobeStreamDiscovery(rtsp_connect_timeout_s=2.0, rtsp_io_timeout_s=2.0)

    # When: probing candidate URLs
    result = discovery.probe(camera_key="garage:lenovo:stream", candidate_urls=["rtsp://x"])

    # Then: stream metadata is populated
    assert result.streams[0].probe_ok
    assert result.streams[0].video_codec == "h264"
    assert result.streams[0].audio_codec == "aac"
    assert result.streams[0].width == 1280
    assert result.streams[0].height == 720
    assert result.streams[0].fps == 15.0


def test_probe_retries_without_timeout_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Discovery should retry without timeout flags when ffprobe rejects them."""
    # Given: ffprobe fails on timeout flags then succeeds without them
    calls: list[list[str]] = []

    def _fake_run(cmd: Sequence[str], **_kwargs: object) -> _FakeResult:
        calls.append(list(cmd))
        if "-rw_timeout" in cmd or "-stimeout" in cmd:
            return _FakeResult(returncode=1, stderr="Option rw_timeout not found")
        return _FakeResult(
            returncode=0,
            stdout=(
                '{"streams":['
                '{"codec_type":"video","codec_name":"h264","width":640,'
                '"height":360,"avg_frame_rate":"10/1"}'
                "]}"
            ),
        )

    monkeypatch.setattr("homesec.sources.rtsp.discovery.subprocess.run", _fake_run)
    discovery = FfprobeStreamDiscovery(rtsp_connect_timeout_s=2.0, rtsp_io_timeout_s=2.0)

    # When: probing a URL
    result = discovery.probe(camera_key="garage:lenovo:stream", candidate_urls=["rtsp://x"])

    # Then: probe succeeds after timeout-option fallback
    assert result.streams[0].probe_ok
    assert len(calls) == 2
    assert "-rw_timeout" in calls[0] or "-stimeout" in calls[0]
    assert "-rw_timeout" not in calls[1]
    assert "-stimeout" not in calls[1]


def test_probe_disables_timeout_flags_after_first_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery should stop retrying timeout flags once unsupported is detected."""
    # Given: ffprobe rejects timeout flags on first probe call
    calls: list[list[str]] = []

    def _fake_run(cmd: Sequence[str], **_kwargs: object) -> _FakeResult:
        calls.append(list(cmd))
        if "-rw_timeout" in cmd or "-stimeout" in cmd:
            return _FakeResult(returncode=1, stderr="Unrecognized option 'stimeout'")
        return _FakeResult(
            returncode=0,
            stdout=(
                '{"streams":['
                '{"codec_type":"video","codec_name":"h264","width":640,'
                '"height":360,"avg_frame_rate":"10/1"}'
                "]}"
            ),
        )

    monkeypatch.setattr("homesec.sources.rtsp.discovery.subprocess.run", _fake_run)
    discovery = FfprobeStreamDiscovery(rtsp_connect_timeout_s=2.0, rtsp_io_timeout_s=2.0)

    # When: probing the same camera twice
    first = discovery.probe(camera_key="garage:lenovo:stream", candidate_urls=["rtsp://x"])
    second = discovery.probe(camera_key="garage:lenovo:stream", candidate_urls=["rtsp://x"])

    # Then: second probe skips timeout-option attempt entirely
    assert first.streams[0].probe_ok
    assert second.streams[0].probe_ok
    assert len(calls) == 3
    assert "-rw_timeout" in calls[0] or "-stimeout" in calls[0]
    assert "-rw_timeout" not in calls[1]
    assert "-stimeout" not in calls[1]
    assert "-rw_timeout" not in calls[2]
    assert "-stimeout" not in calls[2]
