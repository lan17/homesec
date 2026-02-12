from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Sequence
from urllib.parse import urlsplit

from pydantic import BaseModel

from homesec.sources.rtsp.utils import _format_cmd, _is_timeout_option_error, _redact_rtsp_url

logger = logging.getLogger(__name__)


class ProbeStreamInfo(BaseModel):
    """Stream metadata discovered from ffprobe for one RTSP URL."""

    model_config = {"extra": "forbid"}

    url: str
    video_codec: str | None = None
    audio_codec: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    fps_raw: str | None = None
    probe_ok: bool
    error: str | None = None


class CameraProbeResult(BaseModel):
    """Aggregate probe results for all candidate URLs of one camera."""

    model_config = {"extra": "forbid"}

    camera_key: str
    streams: list[ProbeStreamInfo]
    attempted_urls: list[str]
    duration_ms: int


class ProbeError(Exception):
    """Startup probe failure for a camera."""

    def __init__(self, camera_key: str, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.camera_key = camera_key
        self.message = message
        self.cause = cause
        self.__cause__ = cause


class FfprobeStreamDiscovery:
    """RTSP stream discovery using ffprobe (TCP transport only)."""

    def __init__(
        self,
        *,
        rtsp_connect_timeout_s: float,
        rtsp_io_timeout_s: float,
        command_timeout_s: float = 10.0,
    ) -> None:
        self._rtsp_connect_timeout_s = rtsp_connect_timeout_s
        self._rtsp_io_timeout_s = rtsp_io_timeout_s
        self._command_timeout_s = command_timeout_s

    def probe(
        self,
        *,
        camera_key: str,
        candidate_urls: Sequence[str],
    ) -> CameraProbeResult | ProbeError:
        started = time.monotonic()
        ordered_urls = _dedupe_urls(candidate_urls)
        if not ordered_urls:
            return ProbeError(camera_key, "No candidate RTSP URLs provided for probing")

        streams: list[ProbeStreamInfo] = []
        for rtsp_url in ordered_urls:
            streams.append(self._probe_single(rtsp_url))

        return CameraProbeResult(
            camera_key=camera_key,
            streams=streams,
            attempted_urls=list(ordered_urls),
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    def _probe_single(self, rtsp_url: str) -> ProbeStreamInfo:
        base_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,avg_frame_rate",
            "-of",
            "json",
            "-rtsp_transport",
            "tcp",
        ]

        timeout_args: list[str] = []
        if self._rtsp_connect_timeout_s > 0:
            timeout_us_connect = str(int(max(0.1, self._rtsp_connect_timeout_s) * 1_000_000))
            timeout_args.extend(["-stimeout", timeout_us_connect])
        if self._rtsp_io_timeout_s > 0:
            timeout_us_io = str(int(max(0.1, self._rtsp_io_timeout_s) * 1_000_000))
            timeout_args.extend(["-rw_timeout", timeout_us_io])

        attempts: list[tuple[str, list[str]]] = []
        if timeout_args:
            attempts.append(("timeouts", base_cmd + timeout_args + [rtsp_url]))
        attempts.append(("no_timeouts" if timeout_args else "default", base_cmd + [rtsp_url]))

        for label, cmd in attempts:
            redacted_cmd = _format_probe_cmd(cmd)
            logger.debug("Running RTSP probe (%s): %s", label, redacted_cmd)

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._command_timeout_s,
                    check=False,
                )
            except Exception as exc:
                return ProbeStreamInfo(
                    url=rtsp_url,
                    probe_ok=False,
                    error=f"ffprobe execution failed: {type(exc).__name__}",
                )

            if result.returncode != 0:
                stderr_text = _redact_stderr(result.stderr, rtsp_url)
                if _is_timeout_option_error(stderr_text):
                    logger.debug(
                        "ffprobe timeout options unsupported for %s", _redact_rtsp_url(rtsp_url)
                    )
                    continue
                return ProbeStreamInfo(
                    url=rtsp_url,
                    probe_ok=False,
                    error=f"ffprobe failed ({result.returncode}): {stderr_text[:240]}",
                )

            parsed = _parse_ffprobe_payload(result.stdout, rtsp_url)
            if parsed.probe_ok:
                return parsed

            # Parsed payload may still indicate timeout-option issue in stderr after exit=0,
            # but that's unusual; we intentionally do not retry here.
            return parsed

        return ProbeStreamInfo(
            url=rtsp_url,
            probe_ok=False,
            error="ffprobe failed after retries",
        )


def build_camera_key(camera_name: str, rtsp_url: str) -> str:
    """Stable per-camera key: camera + normalized host + normalized path."""

    parsed = urlsplit(rtsp_url)
    host = (parsed.hostname or parsed.netloc or "unknown-host").strip().lower()
    path = (parsed.path or "/").strip().lower().strip("/")

    normalized_name = _normalize_name(camera_name)
    normalized_path = path or "root"
    return f"{normalized_name}:{host}:{normalized_path}"


def _normalize_name(name: str) -> str:
    out: list[str] = []
    for ch in name.strip().lower():
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
            continue
        out.append("_")

    cleaned = "".join(out).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "unknown-camera"


def _dedupe_urls(urls: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        ordered.append(url)
    return ordered


def _format_probe_cmd(cmd: list[str]) -> str:
    redacted = list(cmd)
    if redacted:
        redacted[-1] = _redact_rtsp_url(str(redacted[-1]))
    return _format_cmd(redacted)


def _redact_stderr(stderr_text: str, rtsp_url: str) -> str:
    if not stderr_text:
        return ""
    return stderr_text.replace(rtsp_url, _redact_rtsp_url(rtsp_url)).strip()


def _parse_ffprobe_payload(payload: str, rtsp_url: str) -> ProbeStreamInfo:
    try:
        raw = json.loads(payload)
    except Exception:
        return ProbeStreamInfo(
            url=rtsp_url,
            probe_ok=False,
            error="ffprobe returned invalid JSON",
        )

    streams: list[dict[str, object]] = []
    raw_streams = raw.get("streams")
    if isinstance(raw_streams, list):
        for item in raw_streams:
            if isinstance(item, dict):
                streams.append(item)

    video_stream: dict[str, object] | None = None
    audio_stream: dict[str, object] | None = None

    for stream in streams:
        codec_type = stream.get("codec_type")
        if codec_type == "video" and video_stream is None:
            video_stream = stream
        if codec_type == "audio" and audio_stream is None:
            audio_stream = stream

    if video_stream is None:
        return ProbeStreamInfo(
            url=rtsp_url,
            probe_ok=False,
            error="No video stream found",
        )

    width = _coerce_int(video_stream.get("width"))
    height = _coerce_int(video_stream.get("height"))
    fps_raw = _coerce_str(video_stream.get("avg_frame_rate"))

    return ProbeStreamInfo(
        url=rtsp_url,
        video_codec=_coerce_str(video_stream.get("codec_name")),
        audio_codec=_coerce_str(audio_stream.get("codec_name")) if audio_stream else None,
        width=width,
        height=height,
        fps=_parse_fps(fps_raw),
        fps_raw=fps_raw,
        probe_ok=width is not None and height is not None,
        error=None if width is not None and height is not None else "Video dimensions unavailable",
    )


def _coerce_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _parse_fps(raw_fps: str | None) -> float | None:
    if raw_fps is None:
        return None

    value = raw_fps.strip()
    if not value:
        return None

    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            den = float(denominator)
            if den == 0:
                return None
            return float(numerator) / den
        except ValueError:
            return None

    try:
        return float(value)
    except ValueError:
        return None
