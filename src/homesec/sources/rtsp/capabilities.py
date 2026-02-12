from __future__ import annotations

from enum import Enum
from threading import Lock


class TimeoutOptionSupport(str, Enum):
    """Support status for ffmpeg/ffprobe RTSP timeout options."""

    UNKNOWN = "unknown"
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


class RTSPTimeoutCapabilities:
    """Process-wide RTSP timeout-option compatibility state.

    The state is intentionally one-way during runtime:
    - unknown -> supported
    - unknown/supported -> unsupported
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._ffmpeg = TimeoutOptionSupport.UNKNOWN
        self._ffprobe = TimeoutOptionSupport.UNKNOWN

    def build_ffmpeg_timeout_args(
        self,
        *,
        connect_timeout_s: float,
        io_timeout_s: float,
    ) -> list[str]:
        with self._lock:
            if self._ffmpeg == TimeoutOptionSupport.UNSUPPORTED:
                return []
        return self._build_timeout_args(
            connect_timeout_s=connect_timeout_s, io_timeout_s=io_timeout_s
        )

    def build_ffprobe_timeout_args(
        self,
        *,
        connect_timeout_s: float,
        io_timeout_s: float,
    ) -> list[str]:
        with self._lock:
            if self._ffprobe == TimeoutOptionSupport.UNSUPPORTED:
                return []
        return self._build_timeout_args(
            connect_timeout_s=connect_timeout_s, io_timeout_s=io_timeout_s
        )

    def build_ffmpeg_timeout_args_for_user_flags(
        self,
        *,
        connect_timeout_s: float,
        io_timeout_s: float,
        user_flags: list[str],
    ) -> list[str]:
        """Build timeout args while respecting explicit user timeout options."""
        timeout_args = self.build_ffmpeg_timeout_args(
            connect_timeout_s=connect_timeout_s,
            io_timeout_s=io_timeout_s,
        )
        if not timeout_args:
            return []

        skip_options = {flag for flag in user_flags if flag in {"-stimeout", "-rw_timeout"}}
        if not skip_options:
            return timeout_args

        filtered: list[str] = []
        for idx in range(0, len(timeout_args), 2):
            if idx + 1 >= len(timeout_args):
                continue
            option = timeout_args[idx]
            if option in skip_options:
                continue
            filtered.extend([option, timeout_args[idx + 1]])
        return filtered

    def note_ffmpeg_timeout_success(self) -> None:
        with self._lock:
            if self._ffmpeg == TimeoutOptionSupport.UNKNOWN:
                self._ffmpeg = TimeoutOptionSupport.SUPPORTED

    def note_ffprobe_timeout_success(self) -> None:
        with self._lock:
            if self._ffprobe == TimeoutOptionSupport.UNKNOWN:
                self._ffprobe = TimeoutOptionSupport.SUPPORTED

    def mark_ffmpeg_timeout_unsupported(self) -> bool:
        with self._lock:
            if self._ffmpeg == TimeoutOptionSupport.UNSUPPORTED:
                return False
            self._ffmpeg = TimeoutOptionSupport.UNSUPPORTED
            return True

    def mark_ffprobe_timeout_unsupported(self) -> bool:
        with self._lock:
            if self._ffprobe == TimeoutOptionSupport.UNSUPPORTED:
                return False
            self._ffprobe = TimeoutOptionSupport.UNSUPPORTED
            return True

    def snapshot(self) -> tuple[TimeoutOptionSupport, TimeoutOptionSupport]:
        with self._lock:
            return self._ffmpeg, self._ffprobe

    def reset(self) -> None:
        """Reset state to unknown for tests."""
        with self._lock:
            self._ffmpeg = TimeoutOptionSupport.UNKNOWN
            self._ffprobe = TimeoutOptionSupport.UNKNOWN

    @staticmethod
    def _build_timeout_args(*, connect_timeout_s: float, io_timeout_s: float) -> list[str]:
        args: list[str] = []
        if connect_timeout_s > 0:
            timeout_us_connect = str(int(max(0.1, connect_timeout_s) * 1_000_000))
            args.extend(["-stimeout", timeout_us_connect])
        if io_timeout_s > 0:
            timeout_us_io = str(int(max(0.1, io_timeout_s) * 1_000_000))
            args.extend(["-rw_timeout", timeout_us_io])
        return args


_GLOBAL_RTSP_TIMEOUT_CAPABILITIES = RTSPTimeoutCapabilities()


def get_global_rtsp_timeout_capabilities() -> RTSPTimeoutCapabilities:
    return _GLOBAL_RTSP_TIMEOUT_CAPABILITIES
