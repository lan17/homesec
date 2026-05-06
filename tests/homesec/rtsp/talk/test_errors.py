from __future__ import annotations

import pytest

from homesec.sources.rtsp.talk.errors import (
    CameraBackchannelUnsupportedError,
    CameraRejectedTalkSessionError,
    CameraTalkStreamFailedError,
    RTSPAuthenticationError,
    RTSPProtocolError,
    TalkProtocolError,
    TalkProtocolErrorCode,
    UnsupportedTalkCodecError,
)


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (
            CameraBackchannelUnsupportedError(),
            TalkProtocolErrorCode.CAMERA_BACKCHANNEL_UNSUPPORTED,
        ),
        (UnsupportedTalkCodecError(), TalkProtocolErrorCode.UNSUPPORTED_CODEC),
        (CameraRejectedTalkSessionError(), TalkProtocolErrorCode.CAMERA_REJECTED_SESSION),
        (CameraTalkStreamFailedError(), TalkProtocolErrorCode.CAMERA_STREAM_FAILED),
        (RTSPAuthenticationError(), TalkProtocolErrorCode.RTSP_AUTH_FAILED),
        (RTSPProtocolError(), TalkProtocolErrorCode.RTSP_PROTOCOL_ERROR),
    ],
)
def test_protocol_errors_expose_stable_machine_codes(
    error: TalkProtocolError,
    code: TalkProtocolErrorCode,
) -> None:
    # Given: A concrete talk protocol error instance.
    # When: Its stable machine-code contract is inspected.
    # Then: The error exposes the expected code and human-readable message.
    assert error.code == code
    assert str(error)
