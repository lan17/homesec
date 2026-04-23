from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol, cast
from unittest.mock import patch

from homesec.sources.rtsp.discovery import CameraProbeResult, ProbeStreamInfo
from homesec.sources.rtsp.live_publisher import (
    HLSLivePublisher,
    LivePublisherRefusalReason,
    LivePublisherStartRefusal,
    LivePublisherState,
    LivePublisherStatus,
)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    def now(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self._now += float(seconds)


class FakeDiscovery:
    def __init__(self, stream: ProbeStreamInfo | None) -> None:
        self._stream = stream
        self.calls = 0

    def probe(
        self,
        *,
        camera_key: str,
        candidate_urls: list[str],
    ) -> CameraProbeResult:
        self.calls += 1
        return CameraProbeResult(
            camera_key=camera_key,
            streams=[] if self._stream is None else [self._stream],
            attempted_urls=list(candidate_urls),
            duration_ms=0,
        )


class FakeProc:
    def __init__(self, *, pid: int, returncode: int | None = None) -> None:
        self.pid = pid
        self.returncode = returncode
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_timeouts: list[float | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int | None:
        self.wait_timeouts.append(timeout)
        return self.returncode

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9


class _Writable(Protocol):
    def write(self, text: str) -> object: ...

    def flush(self) -> None: ...


def _probe_stream(
    *,
    video_codec: str,
    audio_codec: str | None,
) -> ProbeStreamInfo:
    return ProbeStreamInfo(
        url="rtsp://host/main",
        video_codec=video_codec,
        audio_codec=audio_codec,
        width=1920,
        height=1080,
        fps=15.0,
        fps_raw="15/1",
        probe_ok=True,
        error=None,
    )


def _make_publisher(
    tmp_path: Path,
    *,
    clock: FakeClock | None = None,
    discovery: FakeDiscovery | None = None,
    idle_timeout_s: float = 5.0,
    audio_codec: Literal["auto", "copy", "aac"] = "auto",
    video_codec: Literal["auto", "copy", "h264"] = "auto",
) -> HLSLivePublisher:
    return HLSLivePublisher(
        camera_name="Front Door #1",
        rtsp_url="rtsp://host/main",
        storage_dir=tmp_path,
        segment_duration_ms=1000,
        live_window_segments=4,
        idle_timeout_s=idle_timeout_s,
        audio_enabled=True,
        audio_codec=audio_codec,
        video_codec=video_codec,
        rtsp_connect_timeout_s=1.0,
        rtsp_io_timeout_s=1.0,
        clock=clock or FakeClock(),
        stream_discovery=discovery
        or FakeDiscovery(_probe_stream(video_codec="h264", audio_codec="aac")),
        background_maintenance=False,
    )


def _fake_popen_factory(
    calls: list[dict[str, object]],
    *,
    make_output: bool = True,
    returncode: int | None = None,
    stderr_text: str = "",
) -> Callable[..., FakeProc]:
    next_pid = 1000

    def fake_popen(
        cmd: list[str],
        *,
        stdout: object = None,
        stderr: object = None,
        start_new_session: bool = False,
        **_: object,
    ) -> FakeProc:
        nonlocal next_pid
        proc = FakeProc(pid=next_pid, returncode=returncode)
        next_pid += 1

        calls.append(
            {
                "cmd": list(cmd),
                "stdout": stdout,
                "stderr": stderr,
                "start_new_session": start_new_session,
                "proc": proc,
            }
        )

        if stderr_text and hasattr(stderr, "write") and hasattr(stderr, "flush"):
            stderr_handle = cast(_Writable, stderr)
            stderr_handle.write(stderr_text)
            stderr_handle.flush()

        if make_output:
            playlist_path = Path(cmd[-1])
            segment_pattern = Path(cmd[cmd.index("-hls_segment_filename") + 1])
            playlist_path.parent.mkdir(parents=True, exist_ok=True)
            playlist_path.write_text(
                "#EXTM3U\n#EXT-X-VERSION:3\n#EXTINF:1.0,\nsegment_000000.ts\n",
                encoding="utf-8",
            )
            Path(str(segment_pattern).replace("%06d", "000000")).write_bytes(b"segment")

        return proc

    return fake_popen


def test_ensure_active_is_idempotent_and_cleans_stale_window(tmp_path: Path) -> None:
    """ensure_active should start once and replace stale playlist state."""
    # Given: A publisher with stale preview files from a prior session
    clock = FakeClock()
    publisher = _make_publisher(tmp_path, clock=clock)
    stale_dir = tmp_path / "homesec" / "Front_Door_1"
    stale_dir.mkdir(parents=True, exist_ok=True)
    stale_file = stale_dir / "old.ts"
    stale_file.write_text("stale", encoding="utf-8")
    calls: list[dict[str, object]] = []

    # When: Activating preview twice
    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=_fake_popen_factory(calls),
    ):
        first = publisher.ensure_active()
        second = publisher.ensure_active()

    # Then: ffmpeg starts once and the live window is recreated in the camera-scoped path
    assert first == LivePublisherStatus(
        state=LivePublisherState.READY,
        viewer_count=0,
        idle_shutdown_at=clock.now() + 5.0,
    )
    assert second == first
    assert len(calls) == 1
    assert calls[0]["start_new_session"] is True
    assert not stale_file.exists()
    assert (stale_dir / "playlist.m3u8").exists()
    assert (stale_dir / "segment_000000.ts").exists()
    cmd = cast(list[str], calls[0]["cmd"])
    assert "-f" in cmd
    assert "hls" in cmd
    assert "delete_segments+append_list+omit_endlist+program_date_time+temp_file" in cmd


def test_auto_codec_prefers_copy_for_h264_and_aac(tmp_path: Path) -> None:
    """auto codec mode should copy browser-safe source codecs."""
    # Given: A publisher whose source probe reports H.264 video and AAC audio
    publisher = _make_publisher(
        tmp_path,
        discovery=FakeDiscovery(_probe_stream(video_codec="h264", audio_codec="aac")),
    )
    calls: list[dict[str, object]] = []

    # When: Activating preview
    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=_fake_popen_factory(calls),
    ):
        result = publisher.ensure_active()

    # Then: The HLS ffmpeg path copy-remuxes both tracks
    assert result == LivePublisherStatus(
        state=LivePublisherState.READY,
        viewer_count=0,
        idle_shutdown_at=5.0,
    )
    cmd = calls[0]["cmd"]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert "libx264" not in cmd


def test_auto_codec_transcodes_non_browser_safe_source(tmp_path: Path) -> None:
    """auto codec mode should transcode unsupported source codecs to browser-safe outputs."""
    # Given: A publisher whose source probe reports H.265 video and PCMA audio
    publisher = _make_publisher(
        tmp_path,
        discovery=FakeDiscovery(_probe_stream(video_codec="hevc", audio_codec="pcm_alaw")),
    )
    calls: list[dict[str, object]] = []

    # When: Activating preview
    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=_fake_popen_factory(calls),
    ):
        result = publisher.ensure_active()

    # Then: The HLS ffmpeg path transcodes to H.264 video and AAC audio
    assert result == LivePublisherStatus(
        state=LivePublisherState.READY,
        viewer_count=0,
        idle_shutdown_at=5.0,
    )
    cmd = calls[0]["cmd"]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-c:a") + 1] == "aac"
    assert cmd[cmd.index("-b:a") + 1] == "128k"


def test_request_stop_terminates_process_and_removes_live_window(tmp_path: Path) -> None:
    """Explicit stop should terminate ffmpeg and remove playlist and segments."""
    # Given: An active preview publisher
    publisher = _make_publisher(tmp_path)
    calls: list[dict[str, object]] = []
    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=_fake_popen_factory(calls),
    ):
        _ = publisher.ensure_active()

    proc = calls[0]["proc"]
    assert isinstance(proc, FakeProc)
    camera_dir = tmp_path / "homesec" / "Front_Door_1"

    # When: Force-stopping preview
    publisher.request_stop()

    # Then: ffmpeg is terminated and the live files are cleaned up
    assert proc.terminate_calls == 1
    assert not camera_dir.exists()
    assert publisher.status() == LivePublisherStatus(state=LivePublisherState.IDLE, viewer_count=0)


def test_idle_shutdown_stops_preview_after_viewers_go_inactive(tmp_path: Path) -> None:
    """Idle timeout should stop the publisher after recent viewers disappear."""
    # Given: An active preview publisher with one recent viewer
    clock = FakeClock()
    publisher = _make_publisher(tmp_path, clock=clock, idle_timeout_s=5.0)
    calls: list[dict[str, object]] = []
    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=_fake_popen_factory(calls),
    ):
        _ = publisher.ensure_active()
    publisher.note_viewer_activity("viewer-1")
    proc = calls[0]["proc"]
    assert isinstance(proc, FakeProc)

    # When: The viewer ages out of the recent-consumer window but idle timeout has not fired yet
    clock.sleep(2.1)
    status_before_timeout = publisher.status()

    # Then: Preview is still ready with no active viewers and a pending idle shutdown time
    assert status_before_timeout.state == LivePublisherState.READY
    assert status_before_timeout.viewer_count == 0
    assert status_before_timeout.idle_shutdown_at == 5.0

    # When: Advancing beyond the idle timeout
    clock.sleep(3.0)
    status_after_timeout = publisher.status()

    # Then: Preview stops and the live window is removed
    assert status_after_timeout == LivePublisherStatus(
        state=LivePublisherState.IDLE,
        viewer_count=0,
    )
    assert proc.terminate_calls == 1
    assert not (tmp_path / "homesec" / "Front_Door_1").exists()


def test_recording_priority_sheds_active_preview_and_refuses_restart(tmp_path: Path) -> None:
    """Recording-active sync should stop preview and refuse new activation."""
    # Given: An active preview publisher
    publisher = _make_publisher(tmp_path)
    calls: list[dict[str, object]] = []
    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=_fake_popen_factory(calls),
    ):
        _ = publisher.ensure_active()
    proc = calls[0]["proc"]
    assert isinstance(proc, FakeProc)

    # When: Recording becomes active and preview is requested again
    publisher.sync_recording_active(True)
    restart_result = publisher.ensure_active()

    # Then: The existing preview is stopped and future starts are refused
    assert proc.terminate_calls == 1
    assert isinstance(restart_result, LivePublisherStartRefusal)
    assert restart_result.reason == LivePublisherRefusalReason.RECORDING_PRIORITY
    assert publisher.status() == LivePublisherStatus(state=LivePublisherState.IDLE, viewer_count=0)


def test_start_failure_maps_session_budget_refusal_and_cleans_storage(tmp_path: Path) -> None:
    """Immediate ffmpeg failure with session-limit hints should return a session-budget refusal."""
    # Given: A publisher whose ffmpeg start fails with a session-limit error
    publisher = _make_publisher(tmp_path)
    calls: list[dict[str, object]] = []

    # When: Activating preview
    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=_fake_popen_factory(
            calls,
            make_output=False,
            returncode=1,
            stderr_text="Too many clients already connected.\n",
        ),
    ):
        result = publisher.ensure_active()

    # Then: The publisher surfaces a session-budget refusal and removes partial live output
    assert isinstance(result, LivePublisherStartRefusal)
    assert result.reason == LivePublisherRefusalReason.SESSION_BUDGET_EXHAUSTED
    status = publisher.status()
    assert status.state == LivePublisherState.ERROR
    assert status.last_error is not None
    assert "Too many clients" in status.last_error
    assert not (tmp_path / "homesec" / "Front_Door_1").exists()
