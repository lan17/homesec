from __future__ import annotations

import pytest

from homesec.sources.rtsp.talk.errors import (
    CameraBackchannelUnsupportedError,
    UnsupportedTalkCodecError,
)
from homesec.sources.rtsp.talk.sdp import (
    advertised_audio_backchannel_codecs,
    parse_sdp,
    select_audio_backchannel,
)

_BACKCHANNEL_SDP = """v=0
o=- 0 0 IN IP4 127.0.0.1
s=HomeSec fake camera
t=0 0
m=video 0 RTP/AVP 96
a=recvonly
a=rtpmap:96 H264/90000
a=control:trackID=video
m=audio 0 RTP/AVP 0 8
a=sendonly
a=rtpmap:0 PCMU/8000
a=rtpmap:8 PCMA/8000
a=control:trackID=backchannel
"""


def test_parse_sdp_collects_sendonly_audio_codecs() -> None:
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
    description = parse_sdp(_BACKCHANNEL_SDP)

    audio = description.media[1]
    assert audio.media == "audio"
    assert audio.direction == "sendonly"
    assert audio.payload_types == [0, 8]
    assert audio.codec_for_payload(0).normalized_name == "PCMU/8000"  # type: ignore[union-attr]
    assert audio.codec_for_payload(8).normalized_name == "PCMA/8000"  # type: ignore[union-attr]


def test_advertised_audio_backchannel_codecs_returns_sendonly_audio_codecs() -> None:
    """Capability probes should expose camera-advertised talk codecs."""
    # Given: SDP with a sendonly audio backchannel
    description = parse_sdp(_BACKCHANNEL_SDP)

    # When: Extracting advertised backchannel codecs
    codecs = advertised_audio_backchannel_codecs(description)

    # Then: The codec list preserves camera SDP order
    assert codecs == ["PCMU/8000", "PCMA/8000"]


def test_select_audio_backchannel_honors_codec_order_and_control_url() -> None:
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
    selected = select_audio_backchannel(
        _BACKCHANNEL_SDP,
        preferred_codecs=["PCMA/8000", "PCMU/8000"],
        base_control_url="rtsp://camera.example/Streaming/Channels/101",
    )

    assert selected.payload_type == 8
    assert selected.selected_codec == "PCMA/8000"
    assert selected.control == "rtsp://camera.example/Streaming/Channels/101/trackID=backchannel"


def test_select_audio_backchannel_uses_session_level_sendonly() -> None:
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
    sdp = """v=0
s=session-level direction

a=sendonly
a=control:rtsp://camera.example/base
m=audio 0 RTP/AVP 0
a=control:trackID=backchannel
"""

    selected = select_audio_backchannel(
        sdp,
        preferred_codecs=["PCMU/8000"],
        base_control_url="rtsp://camera.example/base",
    )

    assert selected.payload_type == 0
    assert selected.control == "rtsp://camera.example/base/trackID=backchannel"


def test_select_audio_backchannel_uses_static_payload_type_without_rtpmap() -> None:
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
    sdp = """v=0
s=static payload
m=audio 0 RTP/AVP 0
a=sendonly
a=control:backchannel
"""

    selected = select_audio_backchannel(
        sdp,
        preferred_codecs=["PCMU/8000"],
        base_control_url="rtsp://camera.example/live",
    )

    assert selected.selected_codec == "PCMU/8000"
    assert selected.control == "rtsp://camera.example/live/backchannel"


def test_select_audio_backchannel_preserves_absolute_control_url() -> None:
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
    sdp = _BACKCHANNEL_SDP.replace(
        "a=control:trackID=backchannel",
        "a=control:rtsp://media.example/talk",
    )

    selected = select_audio_backchannel(
        sdp,
        preferred_codecs=["PCMU/8000"],
        base_control_url="rtsp://camera.example/live",
    )

    assert selected.control == "rtsp://media.example/talk"


def test_select_audio_backchannel_maps_aggregate_control_to_base_url() -> None:
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
    sdp = _BACKCHANNEL_SDP.replace("a=control:trackID=backchannel", "a=control:*")

    selected = select_audio_backchannel(
        sdp,
        preferred_codecs=["PCMU/8000"],
        base_control_url="rtsp://camera.example/live",
    )

    assert selected.control == "rtsp://camera.example/live"


def test_select_audio_backchannel_handles_leading_slash_control_path() -> None:
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
    sdp = _BACKCHANNEL_SDP.replace(
        "a=control:trackID=backchannel",
        "a=control:/Streaming/Channels/101/trackID=backchannel",
    )

    selected = select_audio_backchannel(
        sdp,
        preferred_codecs=["PCMU/8000"],
        base_control_url="rtsp://camera.example/live?profile=main",
    )

    assert selected.control == "rtsp://camera.example/Streaming/Channels/101/trackID=backchannel"


def test_select_audio_backchannel_appends_relative_control_before_base_query() -> None:
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
    selected = select_audio_backchannel(
        _BACKCHANNEL_SDP,
        preferred_codecs=["PCMU/8000"],
        base_control_url="rtsp://camera.example/live?profile=main",
    )

    assert selected.control == "rtsp://camera.example/live/trackID=backchannel"


def test_select_audio_backchannel_uses_session_control_as_base_for_media_control() -> None:
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
    sdp = """v=0
s=session control base
a=sendonly
a=control:rtsp://camera.example/base/
m=audio 0 RTP/AVP 0
a=control:trackID=backchannel
"""

    selected = select_audio_backchannel(
        sdp,
        preferred_codecs=["PCMU/8000"],
        base_control_url="rtsp://camera.example/fallback",
    )

    assert selected.control == "rtsp://camera.example/base/trackID=backchannel"


def test_parse_sdp_ignores_malformed_lines_and_applies_fmtp_to_known_codec() -> None:
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
    sdp = """v=0
s=malformed resilience
m=audio nope RTP/AVP 0
m=audio 0 RTP/AVP nope
m=audio 0 RTP/AVP 99 98 0
a=sendonly
a=rtpmap:bad
a=rtpmap:nope PCMU/8000
a=rtpmap:97 BROKEN
a=rtpmap:96 PCMU/nope
a=rtpmap:98 PCMU/8000/2
a=fmtp:bad
a=fmtp:nope mode=ignore
a=fmtp:99 orphan=true
a=fmtp:98 mode=probe
a=control:talk
"""

    description = parse_sdp(sdp)

    assert len(description.media) == 1
    audio = description.media[0]
    assert audio.payload_types == [99, 98, 0]
    assert audio.codec_for_payload(99) is None
    assert audio.codec_for_payload(98).normalized_name == "PCMU/8000/2"  # type: ignore[union-attr]
    assert audio.codec_for_payload(98).fmtp == "mode=probe"  # type: ignore[union-attr]
    assert audio.codec_for_payload(0).normalized_name == "PCMU/8000"  # type: ignore[union-attr]


def test_select_audio_backchannel_rejects_missing_sendonly_audio() -> None:
    # Given: The test setup represents the scenario named by this test.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
    sdp = _BACKCHANNEL_SDP.replace("a=sendonly", "a=recvonly")

    with pytest.raises(CameraBackchannelUnsupportedError):
        select_audio_backchannel(sdp, preferred_codecs=["PCMU/8000"])


def test_select_audio_backchannel_rejects_unsupported_codec() -> None:
    # Given/When/Then: The error includes both camera-advertised and requested codecs.
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
    with pytest.raises(UnsupportedTalkCodecError, match="advertised: PCMU/8000, PCMA/8000"):
        select_audio_backchannel(_BACKCHANNEL_SDP, preferred_codecs=["OPUS/48000"])


def test_select_audio_backchannel_rejects_payloads_without_codec_mapping() -> None:
    # When: The behavior under test is exercised.
    # Then: The observable result should match the expected contract.
    sdp = """v=0
s=unknown dynamic payload
m=audio 0 RTP/AVP 121
a=sendonly
a=control:talk
"""

    # Given/When/Then: Unknown dynamic payloads remain visible in diagnostics.
    with pytest.raises(UnsupportedTalkCodecError, match="payload:121"):
        select_audio_backchannel(sdp, preferred_codecs=["PCMU/8000"])
