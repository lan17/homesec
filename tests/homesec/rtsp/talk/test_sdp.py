from __future__ import annotations

import pytest

from homesec.sources.rtsp.talk.errors import (
    CameraBackchannelUnsupportedError,
    UnsupportedTalkCodecError,
)
from homesec.sources.rtsp.talk.sdp import parse_sdp, select_audio_backchannel

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
    description = parse_sdp(_BACKCHANNEL_SDP)

    audio = description.media[1]
    assert audio.media == "audio"
    assert audio.direction == "sendonly"
    assert audio.payload_types == [0, 8]
    assert audio.codec_for_payload(0).normalized_name == "PCMU/8000"  # type: ignore[union-attr]
    assert audio.codec_for_payload(8).normalized_name == "PCMA/8000"  # type: ignore[union-attr]


def test_select_audio_backchannel_honors_codec_order_and_control_url() -> None:
    selected = select_audio_backchannel(
        _BACKCHANNEL_SDP,
        preferred_codecs=["PCMA/8000", "PCMU/8000"],
        base_control_url="rtsp://camera.example/Streaming/Channels/101",
    )

    assert selected.payload_type == 8
    assert selected.selected_codec == "PCMA/8000"
    assert selected.control == "rtsp://camera.example/Streaming/Channels/101/trackID=backchannel"


def test_select_audio_backchannel_rejects_missing_sendonly_audio() -> None:
    sdp = _BACKCHANNEL_SDP.replace("a=sendonly", "a=recvonly")

    with pytest.raises(CameraBackchannelUnsupportedError):
        select_audio_backchannel(sdp, preferred_codecs=["PCMU/8000"])


def test_select_audio_backchannel_rejects_unsupported_codec() -> None:
    with pytest.raises(UnsupportedTalkCodecError):
        select_audio_backchannel(_BACKCHANNEL_SDP, preferred_codecs=["OPUS/48000"])
