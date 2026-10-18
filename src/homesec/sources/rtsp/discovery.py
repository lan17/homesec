from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from urllib.parse import urlsplit

from pydantic import BaseModel

from homesec.sources.rtsp.inspection import (
    InspectionFailure,
    InspectionGroup,
    MetadataEvidence,
)

logger = logging.getLogger(__name__)


class ProbeStreamInfo(BaseModel):
    """Stream metadata discovered through native inspection for one RTSP URL."""

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


class NativeStreamDiscovery:
    """Bounded RTSP metadata inspection, independent of recorder eligibility."""

    def __init__(
        self,
        *,
        rtsp_connect_timeout_s: float,
        rtsp_io_timeout_s: float,
        command_timeout_s: float = 10.0,
        helper_path: str = "homesec-webrtc",
        inspection: InspectionGroup | None = None,
    ) -> None:
        self._rtsp_connect_timeout_s = rtsp_connect_timeout_s
        self._rtsp_io_timeout_s = rtsp_io_timeout_s
        self._command_timeout_s = command_timeout_s
        self._helper_path = helper_path
        self._inspection = inspection

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
        try:
            if self._inspection is not None:
                streams = [self._probe_single(url, self._inspection) for url in ordered_urls]
            else:
                with InspectionGroup(
                    helper_path=self._helper_path,
                    total_timeout_s=(len(ordered_urls) * self._command_timeout_s) + 3.0,
                    connect_timeout_s=self._rtsp_connect_timeout_s,
                    io_timeout_s=self._rtsp_io_timeout_s,
                ) as inspection:
                    streams = [self._probe_single(url, inspection) for url in ordered_urls]
        except InspectionFailure as exc:
            return ProbeError(camera_key, str(exc), cause=exc)
        return CameraProbeResult(
            camera_key=camera_key,
            streams=streams,
            attempted_urls=ordered_urls,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    def _probe_single(self, rtsp_url: str, inspection: InspectionGroup) -> ProbeStreamInfo:
        try:
            evidence = inspection.request(
                "inspect_metadata",
                timeout_s=self._command_timeout_s,
                rtsp_url=rtsp_url,
            )
            if not isinstance(evidence, MetadataEvidence):
                raise InspectionFailure("helper_failure")
            usable = evidence.width is not None and evidence.height is not None
            return ProbeStreamInfo(
                url=rtsp_url,
                video_codec=evidence.video_codec,
                audio_codec=evidence.audio_codec,
                width=evidence.width,
                height=evidence.height,
                fps=evidence.fps,
                fps_raw=evidence.fps_raw,
                probe_ok=usable,
                error=None if usable else "Native inspection: invalid_video",
            )
        except InspectionFailure as exc:
            if exc.terminal:
                raise
            return ProbeStreamInfo(url=rtsp_url, probe_ok=False, error=str(exc))


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
