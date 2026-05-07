"""Push-to-talk RTSP/ONVIF error types."""

from __future__ import annotations

from enum import StrEnum


class TalkProtocolErrorCode(StrEnum):
    """Machine-readable RTSP talk failure codes."""

    CAMERA_BACKCHANNEL_UNSUPPORTED = "camera_backchannel_unsupported"
    UNSUPPORTED_CODEC = "unsupported_codec"
    CAMERA_REJECTED_SESSION = "camera_rejected_session"
    CAMERA_STREAM_FAILED = "camera_stream_failed"
    RTSP_AUTH_FAILED = "rtsp_auth_failed"
    RTSP_PROTOCOL_ERROR = "rtsp_protocol_error"


class TalkProtocolError(RuntimeError):
    """Base class for RTSP talk failures safe to map at source/runtime boundaries."""

    def __init__(self, message: str, *, code: TalkProtocolErrorCode) -> None:
        super().__init__(message)
        self.code = code


class CameraBackchannelUnsupportedError(TalkProtocolError):
    """Raised when a camera does not advertise a usable ONVIF audio backchannel."""

    def __init__(
        self, message: str = "Camera does not advertise an ONVIF audio backchannel"
    ) -> None:
        super().__init__(message, code=TalkProtocolErrorCode.CAMERA_BACKCHANNEL_UNSUPPORTED)


class UnsupportedTalkCodecError(TalkProtocolError):
    """Raised when no advertised backchannel codec matches HomeSec's encoder set."""

    def __init__(
        self, message: str = "Camera backchannel does not advertise a supported codec"
    ) -> None:
        super().__init__(message, code=TalkProtocolErrorCode.UNSUPPORTED_CODEC)


class CameraRejectedTalkSessionError(TalkProtocolError):
    """Raised when RTSP SETUP/PLAY or another session command is rejected."""

    def __init__(self, message: str = "Camera rejected the talk session") -> None:
        super().__init__(message, code=TalkProtocolErrorCode.CAMERA_REJECTED_SESSION)


class CameraTalkStreamFailedError(TalkProtocolError):
    """Raised when RTP audio can no longer be sent to the camera."""

    def __init__(self, message: str = "Camera talk stream failed") -> None:
        super().__init__(message, code=TalkProtocolErrorCode.CAMERA_STREAM_FAILED)


class RTSPAuthenticationError(TalkProtocolError):
    """Raised when RTSP authentication cannot be satisfied."""

    def __init__(self, message: str = "RTSP authentication failed") -> None:
        super().__init__(message, code=TalkProtocolErrorCode.RTSP_AUTH_FAILED)


class RTSPProtocolError(TalkProtocolError):
    """Raised for malformed RTSP data or protocol-state violations."""

    def __init__(self, message: str = "Malformed RTSP response") -> None:
        super().__init__(message, code=TalkProtocolErrorCode.RTSP_PROTOCOL_ERROR)
