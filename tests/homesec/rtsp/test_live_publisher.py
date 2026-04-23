from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread, current_thread
from typing import Literal, Protocol, cast
from unittest.mock import patch

from homesec.sources.rtsp.discovery import CameraProbeResult, ProbeStreamInfo
from homesec.sources.rtsp.hardware import HardwareAccelConfig
from homesec.sources.rtsp.live_publisher import (
    HLSLivePublisher,
    LivePublisherRefusalReason,
    LivePublisherStartRefusal,
    LivePublisherState,
    LivePublisherStatus,
)


class FakeClock:
    def __init__(
        self,
        start: float = 0.0,
        *,
        on_sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._now = float(start)
        self._on_sleep = on_sleep

    def now(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self._now += float(seconds)
        if self._on_sleep is not None:
            self._on_sleep(self._now)


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


class SlowDiscovery:
    def __init__(self, stream: ProbeStreamInfo) -> None:
        self._stream = stream
        self.started = Event()
        self.release = Event()

    def probe(
        self,
        *,
        camera_key: str,
        candidate_urls: list[str],
    ) -> CameraProbeResult:
        self.started.set()
        self.release.wait(timeout=1.0)
        return CameraProbeResult(
            camera_key=camera_key,
            streams=[self._stream],
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


def _write_live_output(*, playlist_path: Path, segment_pattern: Path) -> None:
    playlist_path.parent.mkdir(parents=True, exist_ok=True)
    playlist_path.write_text(
        "#EXTM3U\n#EXT-X-VERSION:3\n#EXTINF:1.0,\nsegment_000000.ts\n",
        encoding="utf-8",
    )
    Path(str(segment_pattern).replace("%06d", "000000")).write_bytes(b"segment")


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
    recording_policy: Literal["stop_on_recording", "allow_during_recording"] = (
        "stop_on_recording"
    ),
    audio_codec: Literal["auto", "copy", "aac"] = "auto",
    video_codec: Literal["auto", "copy", "h264"] = "auto",
    hwaccel_config: HardwareAccelConfig | None = None,
    background_maintenance: bool = False,
    maintenance_interval_s: float | None = None,
) -> HLSLivePublisher:
    return HLSLivePublisher(
        camera_name="Front Door #1",
        rtsp_url="rtsp://host/main",
        storage_dir=tmp_path,
        segment_duration_ms=1000,
        live_window_segments=4,
        idle_timeout_s=idle_timeout_s,
        recording_policy=recording_policy,
        audio_enabled=True,
        audio_codec=audio_codec,
        video_codec=video_codec,
        rtsp_connect_timeout_s=1.0,
        rtsp_io_timeout_s=1.0,
        hwaccel_config=hwaccel_config,
        clock=clock or FakeClock(),
        stream_discovery=discovery
        or FakeDiscovery(_probe_stream(video_codec="h264", audio_codec="aac")),
        background_maintenance=background_maintenance,
        maintenance_interval_s=maintenance_interval_s,
    )


def _fake_popen_factory(
    calls: list[dict[str, object]],
    *,
    make_output: bool = True,
    returncode: int | None = None,
    stderr_text: str = "",
    on_spawn: Callable[[FakeProc, list[str]], None] | None = None,
) -> Callable[..., FakeProc]:
    # Use an implausibly high PID so test cleanup never risks signaling a real process group.
    next_pid = 900_000_000

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

        if on_spawn is not None:
            on_spawn(proc, list(cmd))

        if stderr_text and hasattr(stderr, "write") and hasattr(stderr, "flush"):
            stderr_handle = cast(_Writable, stderr)
            stderr_handle.write(stderr_text)
            stderr_handle.flush()

        if make_output:
            playlist_path = Path(cmd[-1])
            segment_pattern = Path(cmd[cmd.index("-hls_segment_filename") + 1])
            _write_live_output(
                playlist_path=playlist_path,
                segment_pattern=segment_pattern,
            )

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


def test_ensure_active_refuses_when_stale_window_cannot_be_cleaned(tmp_path: Path) -> None:
    """Startup should fail closed when stale preview output cannot be removed."""
    # Given: A publisher with stale HLS files that cannot be cleaned before restart
    publisher = _make_publisher(tmp_path)
    stale_dir = tmp_path / "homesec" / "Front_Door_1"
    _write_live_output(
        playlist_path=stale_dir / "playlist.m3u8",
        segment_pattern=stale_dir / "segment_%06d.ts",
    )
    calls: list[dict[str, object]] = []

    # When: Activation is attempted while directory cleanup fails
    with (
        patch(
            "homesec.sources.rtsp.live_publisher.shutil.rmtree",
            side_effect=OSError("device busy"),
        ),
        patch(
            "homesec.sources.rtsp.live_publisher.subprocess.Popen",
            side_effect=_fake_popen_factory(calls, make_output=False),
        ),
    ):
        result = publisher.ensure_active()

    # Then: Startup is refused before ffmpeg spawn instead of reusing stale readiness signals
    assert isinstance(result, LivePublisherStartRefusal)
    assert result.reason == LivePublisherRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE
    assert calls == []
    assert publisher.status() == LivePublisherStatus(
        state=LivePublisherState.ERROR,
        viewer_count=0,
        last_error="Preview storage directory could not be prepared",
    )


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


def test_auto_codec_transcodes_mp4_safe_but_hls_unsafe_audio(tmp_path: Path) -> None:
    """auto codec mode should transcode non-browser HLS audio even if recording can copy it."""
    # Given: A publisher whose source probe reports H.264 video and AC-3 audio
    publisher = _make_publisher(
        tmp_path,
        discovery=FakeDiscovery(_probe_stream(video_codec="h264", audio_codec="ac3")),
    )
    calls: list[dict[str, object]] = []

    # When: Activating preview
    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=_fake_popen_factory(calls),
    ):
        result = publisher.ensure_active()

    # Then: The preview keeps video copy but transcodes audio to AAC for browser playback
    assert result == LivePublisherStatus(
        state=LivePublisherState.READY,
        viewer_count=0,
        idle_shutdown_at=5.0,
    )
    cmd = calls[0]["cmd"]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[cmd.index("-c:a") + 1] == "aac"
    assert cmd[cmd.index("-b:a") + 1] == "128k"


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


def test_transcode_prefers_matching_hardware_encoder_when_available(tmp_path: Path) -> None:
    """Transcode mode should use the matching hardware H.264 encoder before software."""
    # Given: A non-H.264 preview source and a supported VideoToolbox encoder
    publisher = _make_publisher(
        tmp_path,
        discovery=FakeDiscovery(_probe_stream(video_codec="hevc", audio_codec="aac")),
        hwaccel_config=HardwareAccelConfig(hwaccel="videotoolbox"),
    )
    calls: list[dict[str, object]] = []

    # When: Activating preview
    with (
        patch(
            "homesec.sources.rtsp.hardware.HardwareAccelDetector._test_encoder",
            return_value=True,
        ),
        patch(
            "homesec.sources.rtsp.live_publisher.subprocess.Popen",
            side_effect=_fake_popen_factory(calls),
        ),
    ):
        result = publisher.ensure_active()

    # Then: The preview transcodes with the matching hardware encoder
    assert result == LivePublisherStatus(
        state=LivePublisherState.READY,
        viewer_count=0,
        idle_shutdown_at=5.0,
    )
    cmd = calls[0]["cmd"]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("-c:v") + 1] == "h264_videotoolbox"
    assert "libx264" not in cmd


def test_hardware_transcode_falls_back_to_software_when_encoder_startup_fails(
    tmp_path: Path,
) -> None:
    """Hardware transcode failures should retry once with the software H.264 path."""
    # Given: A non-H.264 preview source whose supported NVENC path fails immediately
    publisher = _make_publisher(
        tmp_path,
        discovery=FakeDiscovery(_probe_stream(video_codec="hevc", audio_codec="aac")),
        hwaccel_config=HardwareAccelConfig(hwaccel="cuda"),
    )
    calls: list[dict[str, object]] = []
    hardware_spawn = _fake_popen_factory(
        calls,
        make_output=False,
        returncode=1,
        stderr_text="Error initializing output stream",
    )
    software_spawn = _fake_popen_factory(calls)

    def _fake_popen(
        cmd: list[str],
        *,
        stdout: object = None,
        stderr: object = None,
        start_new_session: bool = False,
        **kwargs: object,
    ) -> FakeProc:
        video_encoder = cmd[cmd.index("-c:v") + 1]
        if video_encoder == "h264_nvenc":
            return hardware_spawn(
                cmd,
                stdout=stdout,
                stderr=stderr,
                start_new_session=start_new_session,
                **kwargs,
            )
        return software_spawn(
            cmd,
            stdout=stdout,
            stderr=stderr,
            start_new_session=start_new_session,
            **kwargs,
        )

    # When: Activating preview
    with (
        patch(
            "homesec.sources.rtsp.hardware.HardwareAccelDetector._test_encoder",
            return_value=True,
        ),
        patch(
            "homesec.sources.rtsp.live_publisher.subprocess.Popen",
            side_effect=_fake_popen,
        ),
    ):
        result = publisher.ensure_active()

    # Then: Preview retries with libx264 and becomes ready
    assert result == LivePublisherStatus(
        state=LivePublisherState.READY,
        viewer_count=0,
        idle_shutdown_at=5.0,
    )
    assert len(calls) == 2
    first_cmd = calls[0]["cmd"]
    second_cmd = calls[1]["cmd"]
    assert isinstance(first_cmd, list)
    assert isinstance(second_cmd, list)
    assert first_cmd[first_cmd.index("-c:v") + 1] == "h264_nvenc"
    assert second_cmd[second_cmd.index("-c:v") + 1] == "libx264"


def test_hardware_transcode_does_not_retry_software_on_session_budget_exhaustion(
    tmp_path: Path,
) -> None:
    """Session-limit refusals should not hide behind a software fallback retry."""
    # Given: A non-H.264 preview source whose hardware attempt hits the camera session limit
    publisher = _make_publisher(
        tmp_path,
        discovery=FakeDiscovery(_probe_stream(video_codec="hevc", audio_codec="aac")),
        hwaccel_config=HardwareAccelConfig(hwaccel="cuda"),
    )
    calls: list[dict[str, object]] = []

    # When: Activating preview
    with (
        patch(
            "homesec.sources.rtsp.hardware.HardwareAccelDetector._test_encoder",
            return_value=True,
        ),
        patch(
            "homesec.sources.rtsp.live_publisher.subprocess.Popen",
            side_effect=_fake_popen_factory(
                calls,
                make_output=False,
                returncode=1,
                stderr_text="Too many clients already connected",
            ),
        ),
    ):
        result = publisher.ensure_active()

    # Then: Preview reports the session-budget refusal without retrying software encode
    assert isinstance(result, LivePublisherStartRefusal)
    assert result.reason == LivePublisherRefusalReason.SESSION_BUDGET_EXHAUSTED
    assert len(calls) == 1
    cmd = calls[0]["cmd"]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("-c:v") + 1] == "h264_nvenc"


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


def test_request_stop_reports_error_when_process_does_not_stop_cleanly(tmp_path: Path) -> None:
    """Explicit stop should fail closed when ffmpeg teardown does not complete."""
    # Given: An active preview publisher whose ffmpeg stop path reports failure
    publisher = _make_publisher(tmp_path)
    calls: list[dict[str, object]] = []
    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=_fake_popen_factory(calls),
    ):
        _ = publisher.ensure_active()

    camera_dir = tmp_path / "homesec" / "Front_Door_1"

    # When: Force-stopping preview while process teardown reports failure
    with patch.object(
        publisher,
        "_terminate_process",
        return_value="Preview ffmpeg could not be stopped cleanly",
    ):
        publisher.request_stop()

    # Then: The publisher surfaces the stop failure instead of reporting a clean idle stop
    assert not camera_dir.exists()
    assert publisher.status() == LivePublisherStatus(
        state=LivePublisherState.ERROR,
        viewer_count=0,
        last_error="Preview ffmpeg could not be stopped cleanly",
    )


def test_request_stop_reports_error_when_live_window_cleanup_fails(tmp_path: Path) -> None:
    """Explicit stop should fail closed when live HLS cleanup cannot finish."""
    # Given: An active preview publisher whose live window cannot be removed
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

    # When: Force-stopping preview while live-window cleanup fails
    with patch(
        "homesec.sources.rtsp.live_publisher.shutil.rmtree",
        side_effect=OSError("device busy"),
    ):
        publisher.request_stop()

    # Then: The publisher preserves an error state instead of claiming preview is idle
    assert proc.terminate_calls == 1
    assert camera_dir.exists()
    assert publisher.status() == LivePublisherStatus(
        state=LivePublisherState.ERROR,
        viewer_count=0,
        last_error="Preview live output could not be removed",
    )


def test_request_stop_allows_maintenance_thread_to_exit(tmp_path: Path) -> None:
    """Explicit stop should not leave a permanent maintenance thread behind."""
    # Given: An active preview publisher with background maintenance enabled
    publisher = _make_publisher(
        tmp_path,
        background_maintenance=True,
        maintenance_interval_s=0.01,
    )
    calls: list[dict[str, object]] = []
    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=_fake_popen_factory(calls),
    ):
        _ = publisher.ensure_active()

    maintenance_thread = publisher._maintenance_thread
    assert maintenance_thread is not None
    assert maintenance_thread.is_alive()

    # When: Force-stopping preview
    publisher.request_stop()
    maintenance_thread.join(timeout=0.5)

    # Then: The maintenance thread exits once the publisher is idle
    assert not maintenance_thread.is_alive()
    assert publisher._maintenance_thread is None


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


def test_recording_priority_sheds_active_preview_and_refuses_restart(
    tmp_path: Path,
    caplog,
) -> None:
    """Recording-active sync should stop preview and refuse new activation."""
    # Given: An active preview publisher
    publisher = _make_publisher(tmp_path)
    calls: list[dict[str, object]] = []
    with caplog.at_level(logging.INFO, logger="homesec.sources.rtsp.live_publisher"):
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
    assert any(
        "Stopping preview because recording became active" in record.message
        for record in caplog.records
    )
    assert any(
        "Refusing preview activation because recording is active" in record.message
        for record in caplog.records
    )


def test_allow_during_recording_keeps_active_preview_and_allows_restart(
    tmp_path: Path,
    caplog,
) -> None:
    """Concurrent-preview mode should keep preview alive when recording becomes active."""
    # Given: An active preview publisher configured to allow preview during recording
    publisher = _make_publisher(tmp_path, recording_policy="allow_during_recording")
    calls: list[dict[str, object]] = []
    with (
        caplog.at_level(logging.INFO, logger="homesec.sources.rtsp.live_publisher"),
        patch(
            "homesec.sources.rtsp.live_publisher.subprocess.Popen",
            side_effect=_fake_popen_factory(calls),
        ),
    ):
        first = publisher.ensure_active()
        proc = calls[0]["proc"]
        assert isinstance(proc, FakeProc)

        # When: Recording becomes active and preview is requested again
        publisher.sync_recording_active(True)
        second = publisher.ensure_active()

    # Then: The existing preview keeps running and no refusal is returned
    assert first == LivePublisherStatus(
        state=LivePublisherState.READY,
        viewer_count=0,
        idle_shutdown_at=5.0,
    )
    assert second == first
    assert proc.terminate_calls == 0
    assert publisher.status() == first
    assert any(
        "preview remains allowed by configuration" in record.message for record in caplog.records
    )
    assert not any(
        "Stopping preview because recording became active" in record.message
        or "Refusing preview activation because recording is active" in record.message
        for record in caplog.records
    )


def test_allow_during_recording_allows_start_while_recording_is_active(tmp_path: Path) -> None:
    """Concurrent-preview mode should allow startup even when recording is already active."""
    # Given: A publisher configured to allow preview while recording is active
    publisher = _make_publisher(tmp_path, recording_policy="allow_during_recording")
    calls: list[dict[str, object]] = []
    publisher.sync_recording_active(True)

    # When: Activating preview during recording
    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=_fake_popen_factory(calls),
    ):
        result = publisher.ensure_active()

    # Then: Preview starts normally instead of returning a recording-priority refusal
    assert result == LivePublisherStatus(
        state=LivePublisherState.READY,
        viewer_count=0,
        idle_shutdown_at=5.0,
    )
    assert len(calls) == 1
    assert publisher.status() == result


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


def test_timeout_retry_aborts_when_failed_start_teardown_cannot_clean_up(tmp_path: Path) -> None:
    """Timeout-option fallback should stop retrying when failed-start cleanup leaves stale state."""
    # Given: A publisher whose failed start cannot fully clean its live window
    publisher = _make_publisher(tmp_path)
    calls: list[dict[str, object]] = []
    cleanup_calls = 0
    original_cleanup = publisher._cleanup_live_dir_locked

    def cleanup_with_failed_stop() -> bool:
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 2:
            return False
        return original_cleanup()

    # When: Activating preview with ffmpeg rejecting timeout options
    with (
        patch.object(
            publisher,
            "_cleanup_live_dir_locked",
            side_effect=cleanup_with_failed_stop,
        ),
        patch(
            "homesec.sources.rtsp.live_publisher.subprocess.Popen",
            side_effect=_fake_popen_factory(
                calls,
                make_output=False,
                returncode=1,
                stderr_text="Unrecognized option 'rw_timeout'.\n",
            ),
        ),
    ):
        result = publisher.ensure_active()

    # Then: Startup fails closed instead of retrying a second ffmpeg launch
    assert isinstance(result, LivePublisherStartRefusal)
    assert result.reason == LivePublisherRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE
    assert len(calls) == 1
    status = publisher.status()
    assert status.state == LivePublisherState.ERROR
    assert status.last_error is not None
    assert "rw_timeout" in status.last_error
    assert "Preview live output could not be removed" in status.last_error


def test_non_session_limit_failure_does_not_map_generic_too_many_text(tmp_path: Path) -> None:
    """Non-session ffmpeg errors should not be misclassified as session-budget refusals."""
    # Given: A publisher whose ffmpeg start fails with a non-session "too many" error
    publisher = _make_publisher(tmp_path)
    calls: list[dict[str, object]] = []

    # When: Activating preview
    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=_fake_popen_factory(
            calls,
            make_output=False,
            returncode=1,
            stderr_text="Too many packets buffered for output stream 0:1.\n",
        ),
    ):
        result = publisher.ensure_active()

    # Then: The refusal stays generic instead of claiming the camera session budget is exhausted
    assert isinstance(result, LivePublisherStartRefusal)
    assert result.reason == LivePublisherRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE
    assert publisher.status().state == LivePublisherState.ERROR


def test_startup_requires_running_process_when_ready_deadline_expires(tmp_path: Path) -> None:
    """Startup should refuse readiness when ffmpeg exits before the deadline check completes."""
    # Given: A publisher whose ffmpeg produces HLS files but exits at the startup deadline
    calls: list[dict[str, object]] = []
    spawn: dict[str, FakeProc | Path] = {}

    def on_sleep(now: float) -> None:
        proc = spawn.get("proc")
        playlist_path = spawn.get("playlist_path")
        segment_pattern = spawn.get("segment_pattern")
        if (
            now < 6.0
            or not isinstance(proc, FakeProc)
            or not isinstance(playlist_path, Path)
            or not isinstance(segment_pattern, Path)
            or proc.returncode is not None
        ):
            return

        _write_live_output(
            playlist_path=playlist_path,
            segment_pattern=segment_pattern,
        )
        proc.returncode = 1

    clock = FakeClock(on_sleep=on_sleep)
    publisher = _make_publisher(tmp_path, clock=clock)

    def on_spawn(proc: FakeProc, cmd: list[str]) -> None:
        spawn["proc"] = proc
        spawn["playlist_path"] = Path(cmd[-1])
        spawn["segment_pattern"] = Path(cmd[cmd.index("-hls_segment_filename") + 1])

    # When: Activating preview
    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=_fake_popen_factory(
            calls,
            make_output=False,
            on_spawn=on_spawn,
        ),
    ):
        result = publisher.ensure_active()

    # Then: Startup fails closed instead of reporting READY for a dead ffmpeg process
    assert isinstance(result, LivePublisherStartRefusal)
    assert result.reason == LivePublisherRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE
    status = publisher.status()
    assert status.state == LivePublisherState.ERROR
    assert status.last_error == "Preview publisher exited before producing HLS output"
    assert not (tmp_path / "homesec" / "Front_Door_1").exists()


def test_slow_startup_arms_idle_timeout_from_ready_time(tmp_path: Path) -> None:
    """Slow startup should not age out the idle window before preview becomes ready."""
    # Given: A publisher whose HLS output does not appear until the readiness deadline
    calls: list[dict[str, object]] = []
    spawn: dict[str, FakeProc | Path] = {}

    def on_sleep(now: float) -> None:
        playlist_path = spawn.get("playlist_path")
        segment_pattern = spawn.get("segment_pattern")
        if (
            now < 6.0
            or not isinstance(playlist_path, Path)
            or not isinstance(segment_pattern, Path)
            or playlist_path.exists()
        ):
            return

        _write_live_output(
            playlist_path=playlist_path,
            segment_pattern=segment_pattern,
        )

    clock = FakeClock(on_sleep=on_sleep)
    publisher = _make_publisher(tmp_path, clock=clock, idle_timeout_s=5.0)

    def on_spawn(proc: FakeProc, cmd: list[str]) -> None:
        spawn["proc"] = proc
        spawn["playlist_path"] = Path(cmd[-1])
        spawn["segment_pattern"] = Path(cmd[cmd.index("-hls_segment_filename") + 1])

    # When: Activating preview after a slow HLS startup
    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=_fake_popen_factory(
            calls,
            make_output=False,
            on_spawn=on_spawn,
        ),
    ):
        result = publisher.ensure_active()

    # Then: The idle timeout starts from readiness instead of expiring during startup
    assert result.state == LivePublisherState.READY
    assert result.viewer_count == 0
    assert result.idle_shutdown_at == clock.now() + 5.0

    status_while_ready = publisher.status()
    assert status_while_ready.state == LivePublisherState.READY
    assert status_while_ready.viewer_count == 0
    assert status_while_ready.idle_shutdown_at == result.idle_shutdown_at
    proc = calls[0]["proc"]
    assert isinstance(proc, FakeProc)
    assert proc.terminate_calls == 0

    clock.sleep(4.9)
    assert publisher.status() == status_while_ready

    clock.sleep(0.2)
    assert publisher.status() == LivePublisherStatus(
        state=LivePublisherState.IDLE,
        viewer_count=0,
    )
    assert proc.terminate_calls == 1


def test_waiting_activation_does_not_return_ready_until_start_completes(tmp_path: Path) -> None:
    """Concurrent callers should queue behind startup (even after spawn) and share its result."""

    # Given: A publisher whose ffmpeg spawns but never produces HLS output
    class _ThreadClock:
        def now(self) -> float:
            return time.monotonic()

        def sleep(self, seconds: float) -> None:
            time.sleep(seconds)

    publisher = _make_publisher(tmp_path, clock=_ThreadClock())
    publisher._startup_timeout_s = 0.3
    calls: list[dict[str, object]] = []
    spawned = Event()
    start_results: dict[str, LivePublisherStatus | LivePublisherStartRefusal] = {}

    def on_spawn(proc: FakeProc, cmd: list[str]) -> None:
        _ = proc
        _ = cmd
        spawned.set()

    def run_ensure_active(name: str) -> None:
        start_results[name] = publisher.ensure_active()

    # When: A second caller arrives while the first startup is still waiting for readiness
    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=_fake_popen_factory(
            calls,
            make_output=False,
            on_spawn=on_spawn,
        ),
    ):
        first_thread = Thread(target=run_ensure_active, args=("first",))
        second_thread = Thread(target=run_ensure_active, args=("second",))
        first_thread.start()
        assert spawned.wait(timeout=0.5)
        second_thread.start()
        first_thread.join(timeout=1.0)
        second_thread.join(timeout=1.0)

    # Then: Both callers share the eventual failed start instead of returning READY early
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert len(calls) == 1
    first_result = start_results["first"]
    second_result = start_results["second"]
    assert isinstance(first_result, LivePublisherStartRefusal)
    assert second_result == first_result


def test_waiting_activation_restarts_when_process_exits_before_waiter_wakes(tmp_path: Path) -> None:
    """Queued activation callers should not return a stale READY after ffmpeg has already exited."""

    # Given: A publisher whose first startup is slow enough for a second caller to queue behind it
    class _ControlledClock:
        def __init__(self) -> None:
            self.waiter_sleep_started = Event()
            self.waiter_sleep_release = Event()

        def now(self) -> float:
            return time.monotonic()

        def sleep(self, seconds: float) -> None:
            if current_thread().name == "waiter":
                self.waiter_sleep_started.set()
                self.waiter_sleep_release.wait()
                return
            time.sleep(seconds)

    clock = _ControlledClock()
    publisher = _make_publisher(tmp_path, clock=clock)
    publisher._startup_timeout_s = 1.0

    calls: list[dict[str, object]] = []
    spawned = Event()
    first_proc: FakeProc | None = None
    start_results: dict[str, LivePublisherStatus | LivePublisherStartRefusal] = {}

    def on_spawn(proc: FakeProc, cmd: list[str]) -> None:
        _ = cmd
        nonlocal first_proc
        if first_proc is None:
            first_proc = proc
            spawned.set()

    first_spawn = _fake_popen_factory(
        calls,
        make_output=False,
        on_spawn=on_spawn,
    )
    second_spawn = _fake_popen_factory(calls)

    def fake_popen(
        cmd: list[str],
        *,
        stdout: object = None,
        stderr: object = None,
        start_new_session: bool = False,
        **kwargs: object,
    ) -> FakeProc:
        factory = first_spawn if len(calls) == 0 else second_spawn
        return factory(
            cmd,
            stdout=stdout,
            stderr=stderr,
            start_new_session=start_new_session,
            **kwargs,
        )

    def run_ensure_active(name: str) -> None:
        start_results[name] = publisher.ensure_active()

    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=fake_popen,
    ):
        starter_thread = Thread(
            target=run_ensure_active,
            args=("starter",),
            name="starter",
            daemon=True,
        )
        waiter_thread = Thread(
            target=run_ensure_active,
            args=("waiter",),
            name="waiter",
            daemon=True,
        )
        try:
            starter_thread.start()
            assert spawned.wait(timeout=0.5)
            waiter_thread.start()
            assert clock.waiter_sleep_started.wait(timeout=0.5)

            camera_dir = tmp_path / "homesec" / "Front_Door_1"
            _write_live_output(
                playlist_path=camera_dir / "playlist.m3u8",
                segment_pattern=camera_dir / "segment_%06d.ts",
            )

            starter_thread.join(timeout=1.0)
            assert not starter_thread.is_alive()
            starter_result = start_results["starter"]
            assert isinstance(starter_result, LivePublisherStatus)
            assert starter_result.state == LivePublisherState.READY
            assert starter_result.viewer_count == 0

            # When: The underlying process exits before the queued waiter observes the completed start
            assert isinstance(first_proc, FakeProc)
            first_proc.returncode = 1
            clock.waiter_sleep_release.set()
            waiter_thread.join(timeout=1.0)
        finally:
            clock.waiter_sleep_release.set()
            starter_thread.join(timeout=1.0)
            waiter_thread.join(timeout=1.0)

    # Then: The queued waiter restarts preview instead of returning a stale READY result
    assert not waiter_thread.is_alive()
    assert len(calls) == 2
    waiter_result = start_results["waiter"]
    assert isinstance(waiter_result, LivePublisherStatus)
    assert waiter_result.state == LivePublisherState.READY


def test_request_stop_preempts_slow_startup_without_error(tmp_path: Path) -> None:
    """Explicit stop should cancel a slow startup instead of surfacing preview failure."""

    # Given: A publisher whose ffmpeg process stays alive but never produces HLS output
    class _ThreadClock:
        def now(self) -> float:
            return time.monotonic()

        def sleep(self, seconds: float) -> None:
            time.sleep(seconds)

    publisher = _make_publisher(tmp_path, clock=_ThreadClock())
    publisher._startup_timeout_s = 0.5
    calls: list[dict[str, object]] = []
    spawned = Event()
    start_result: dict[str, LivePublisherStatus | LivePublisherStartRefusal] = {}

    def on_spawn(proc: FakeProc, cmd: list[str]) -> None:
        _ = proc
        _ = cmd
        spawned.set()

    def run_ensure_active() -> None:
        with patch(
            "homesec.sources.rtsp.live_publisher.subprocess.Popen",
            side_effect=_fake_popen_factory(
                calls,
                make_output=False,
                on_spawn=on_spawn,
            ),
        ):
            start_result["result"] = publisher.ensure_active()

    ensure_thread = Thread(target=run_ensure_active)
    ensure_thread.start()
    assert spawned.wait(timeout=0.2)

    # When: Preview is explicitly stopped while startup is still waiting for HLS readiness
    publisher.request_stop()
    ensure_thread.join(timeout=0.3)

    # Then: Startup resolves as an idle cancellation instead of forcing ERROR state
    assert not ensure_thread.is_alive()
    result = start_result["result"]
    assert result == LivePublisherStatus(state=LivePublisherState.IDLE, viewer_count=0)
    proc = calls[0]["proc"]
    assert isinstance(proc, FakeProc)
    assert proc.terminate_calls == 1
    assert publisher.status() == LivePublisherStatus(state=LivePublisherState.IDLE, viewer_count=0)


def test_recording_priority_preempts_slow_startup(tmp_path: Path, caplog) -> None:
    """Recording activation should preempt a slow preview startup without blocking."""

    # Given: A publisher whose ffmpeg process stays alive but never produces HLS output
    class _ThreadClock:
        def now(self) -> float:
            return time.monotonic()

        def sleep(self, seconds: float) -> None:
            time.sleep(seconds)

    publisher = _make_publisher(tmp_path, clock=_ThreadClock())
    publisher._startup_timeout_s = 0.5
    calls: list[dict[str, object]] = []
    spawned = Event()
    start_result: dict[str, LivePublisherStatus | LivePublisherStartRefusal] = {}

    def on_spawn(proc: FakeProc, cmd: list[str]) -> None:
        _ = proc
        _ = cmd
        spawned.set()

    def run_ensure_active() -> None:
        with patch(
            "homesec.sources.rtsp.live_publisher.subprocess.Popen",
            side_effect=_fake_popen_factory(
                calls,
                make_output=False,
                on_spawn=on_spawn,
            ),
        ):
            start_result["result"] = publisher.ensure_active()

    with caplog.at_level(logging.INFO, logger="homesec.sources.rtsp.live_publisher"):
        ensure_thread = Thread(target=run_ensure_active)
        ensure_thread.start()
        assert spawned.wait(timeout=0.2)

        sync_completed = Event()

        # When: Recording becomes active while startup is still waiting for HLS readiness
        def run_sync_recording_active() -> None:
            publisher.sync_recording_active(True)
            sync_completed.set()

        sync_thread = Thread(target=run_sync_recording_active)
        sync_thread.start()

        # Then: Recording-priority preemption does not block on the startup wait loop
        sync_thread.join(timeout=0.3)
        ensure_thread.join(timeout=0.3)

    assert sync_completed.is_set()
    assert not sync_thread.is_alive()
    assert not ensure_thread.is_alive()
    result = start_result["result"]
    assert isinstance(result, LivePublisherStartRefusal)
    assert result.reason == LivePublisherRefusalReason.RECORDING_PRIORITY
    proc = calls[0]["proc"]
    assert isinstance(proc, FakeProc)
    assert proc.terminate_calls == 1
    assert publisher.status() == LivePublisherStatus(state=LivePublisherState.IDLE, viewer_count=0)
    assert any(
        (
            "Stopping preview because recording became active" in record.message
            or "Cancelling preview startup because recording became active" in record.message
        )
        for record in caplog.records
    )


def test_request_stop_clears_racing_start_failure_error(tmp_path: Path) -> None:
    """Explicit stop should win over a concurrent startup-failure error transition."""

    # Given: A publisher timing out during startup while error reporting is delayed
    class _ThreadClock:
        def now(self) -> float:
            return time.monotonic()

        def sleep(self, seconds: float) -> None:
            time.sleep(seconds)

    publisher = _make_publisher(tmp_path, clock=_ThreadClock())
    publisher._startup_timeout_s = 0.2
    calls: list[dict[str, object]] = []
    spawned = Event()
    error_transition_started = Event()
    allow_error_transition = Event()

    original_set_error_locked = publisher._set_error_locked

    def on_spawn(proc: FakeProc, cmd: list[str]) -> None:
        _ = proc
        _ = cmd
        spawned.set()

    def delayed_set_error_locked(
        *,
        reason: LivePublisherRefusalReason,
        message: str,
        last_error: str,
    ) -> LivePublisherStartRefusal:
        error_transition_started.set()
        allow_error_transition.wait(timeout=1.0)
        return original_set_error_locked(
            reason=reason,
            message=message,
            last_error=last_error,
        )

    def run_ensure_active() -> None:
        with (
            patch(
                "homesec.sources.rtsp.live_publisher.subprocess.Popen",
                side_effect=_fake_popen_factory(
                    calls,
                    make_output=False,
                    on_spawn=on_spawn,
                ),
            ),
            patch.object(
                publisher,
                "_set_error_locked",
                side_effect=delayed_set_error_locked,
            ),
        ):
            _ = publisher.ensure_active()

    ensure_thread = Thread(target=run_ensure_active)
    ensure_thread.start()
    assert spawned.wait(timeout=0.2)
    assert error_transition_started.wait(timeout=1.0)

    stop_completed = Event()

    # When: Explicit stop races the startup-failure error transition
    def run_request_stop() -> None:
        publisher.request_stop()
        stop_completed.set()

    stop_thread = Thread(target=run_request_stop)
    stop_thread.start()
    time.sleep(0.05)
    allow_error_transition.set()
    ensure_thread.join(timeout=1.0)
    stop_thread.join(timeout=1.0)

    # Then: The later stop request leaves the publisher idle instead of ERROR
    assert stop_completed.is_set()
    assert not ensure_thread.is_alive()
    assert not stop_thread.is_alive()
    assert publisher.status() == LivePublisherStatus(state=LivePublisherState.IDLE, viewer_count=0)


def test_request_stop_preempts_slow_probe_before_spawn(tmp_path: Path) -> None:
    """Explicit stop should cancel startup before ffmpeg spawn when probing is slow."""

    # Given: A publisher whose codec probe is still in flight
    class _ThreadClock:
        def now(self) -> float:
            return time.monotonic()

        def sleep(self, seconds: float) -> None:
            time.sleep(seconds)

    discovery = SlowDiscovery(_probe_stream(video_codec="h264", audio_codec="aac"))
    publisher = _make_publisher(tmp_path, clock=_ThreadClock(), discovery=discovery)
    calls: list[dict[str, object]] = []
    start_result: dict[str, LivePublisherStatus | LivePublisherStartRefusal] = {}

    def run_ensure_active() -> None:
        with patch(
            "homesec.sources.rtsp.live_publisher.subprocess.Popen",
            side_effect=_fake_popen_factory(calls),
        ):
            start_result["result"] = publisher.ensure_active()

    ensure_thread = Thread(target=run_ensure_active)
    ensure_thread.start()
    assert discovery.started.wait(timeout=0.2)

    # When: Preview is explicitly stopped before the probe has released
    publisher.request_stop()
    discovery.release.set()
    ensure_thread.join(timeout=0.3)

    # Then: Startup finishes idle without spawning preview ffmpeg
    assert not ensure_thread.is_alive()
    result = start_result["result"]
    assert result == LivePublisherStatus(state=LivePublisherState.IDLE, viewer_count=0)
    assert calls == []
    assert publisher.status() == LivePublisherStatus(state=LivePublisherState.IDLE, viewer_count=0)


def test_request_stop_cancels_waiting_activation_queue(tmp_path: Path) -> None:
    """Explicit stop should prevent queued activation callers from immediately restarting preview."""

    # Given: One preview activation probing slowly while a second caller waits behind it
    class _ThreadClock:
        def __init__(self) -> None:
            self.wait_started = Event()

        def now(self) -> float:
            return time.monotonic()

        def sleep(self, seconds: float) -> None:
            self.wait_started.set()
            time.sleep(seconds)

    clock = _ThreadClock()
    discovery = SlowDiscovery(_probe_stream(video_codec="h264", audio_codec="aac"))
    publisher = _make_publisher(tmp_path, clock=clock, discovery=discovery)
    calls: list[dict[str, object]] = []
    start_results: dict[str, LivePublisherStatus | LivePublisherStartRefusal] = {}

    def run_ensure_active(name: str) -> None:
        start_results[name] = publisher.ensure_active()

    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=_fake_popen_factory(calls),
    ):
        first_thread = Thread(target=run_ensure_active, args=("first",))
        second_thread = Thread(target=run_ensure_active, args=("second",))
        first_thread.start()
        assert discovery.started.wait(timeout=0.2)
        second_thread.start()
        assert clock.wait_started.wait(timeout=0.2)

        # When: Preview is explicitly stopped while a second caller is queued behind startup
        publisher.request_stop()
        discovery.release.set()
        first_thread.join(timeout=1.0)
        second_thread.join(timeout=1.0)

    # Then: Both in-flight activation calls resolve idle and no new ffmpeg process is spawned
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert start_results["first"] == LivePublisherStatus(
        state=LivePublisherState.IDLE,
        viewer_count=0,
    )
    assert start_results["second"] == LivePublisherStatus(
        state=LivePublisherState.IDLE,
        viewer_count=0,
    )
    assert calls == []
    assert publisher.status() == LivePublisherStatus(state=LivePublisherState.IDLE, viewer_count=0)


def test_request_stop_cancels_activation_that_queued_after_stop(tmp_path: Path) -> None:
    """Explicit stop should fence callers that queue behind a start already being cancelled."""

    # Given: One preview activation probing slowly while a later caller arrives after stop
    class _ThreadClock:
        def __init__(self) -> None:
            self.wait_started = Event()

        def now(self) -> float:
            return time.monotonic()

        def sleep(self, seconds: float) -> None:
            self.wait_started.set()
            time.sleep(seconds)

    clock = _ThreadClock()
    discovery = SlowDiscovery(_probe_stream(video_codec="h264", audio_codec="aac"))
    publisher = _make_publisher(tmp_path, clock=clock, discovery=discovery)
    calls: list[dict[str, object]] = []
    start_results: dict[str, LivePublisherStatus | LivePublisherStartRefusal] = {}

    def run_ensure_active(name: str) -> None:
        start_results[name] = publisher.ensure_active()

    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=_fake_popen_factory(calls),
    ):
        first_thread = Thread(target=run_ensure_active, args=("first",))
        second_thread = Thread(target=run_ensure_active, args=("second",))
        first_thread.start()
        assert discovery.started.wait(timeout=0.2)

        # When: Preview is explicitly stopped before a second caller queues behind startup
        publisher.request_stop()
        second_thread.start()
        assert clock.wait_started.wait(timeout=0.2)

        discovery.release.set()
        first_thread.join(timeout=1.0)
        second_thread.join(timeout=1.0)

    # Then: The late-queued activation also resolves idle and does not restart ffmpeg
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert start_results["first"] == LivePublisherStatus(
        state=LivePublisherState.IDLE,
        viewer_count=0,
    )
    assert start_results["second"] == LivePublisherStatus(
        state=LivePublisherState.IDLE,
        viewer_count=0,
    )
    assert calls == []
    assert publisher.status() == LivePublisherStatus(state=LivePublisherState.IDLE, viewer_count=0)


def test_request_stop_keeps_cancelled_queue_from_observing_later_restart(tmp_path: Path) -> None:
    """Explicit stop should keep stale queued callers from inheriting a later restart."""

    # Given: One queued activation is already waiting when an explicit stop cancels startup
    class _ThreadClock:
        def __init__(self) -> None:
            self.wait_started = Event()
            self.release_wait = Event()

        def now(self) -> float:
            return time.monotonic()

        def sleep(self, seconds: float) -> None:
            _ = seconds
            self.wait_started.set()
            self.release_wait.wait(timeout=1.0)

    clock = _ThreadClock()
    discovery = SlowDiscovery(_probe_stream(video_codec="h264", audio_codec="aac"))
    publisher = _make_publisher(tmp_path, clock=clock, discovery=discovery)
    calls: list[dict[str, object]] = []
    start_results: dict[str, LivePublisherStatus | LivePublisherStartRefusal] = {}

    def run_ensure_active(name: str) -> None:
        start_results[name] = publisher.ensure_active()

    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=_fake_popen_factory(calls),
    ):
        first_thread = Thread(target=run_ensure_active, args=("first",))
        second_thread = Thread(target=run_ensure_active, args=("second",))
        third_thread = Thread(target=run_ensure_active, args=("third",))
        first_thread.start()
        assert discovery.started.wait(timeout=0.2)
        second_thread.start()
        assert clock.wait_started.wait(timeout=0.2)

        # When: A later restart succeeds before the stale queued caller wakes up
        publisher.request_stop()
        discovery.release.set()
        first_thread.join(timeout=1.0)
        third_thread.start()
        third_thread.join(timeout=1.0)
        clock.release_wait.set()
        second_thread.join(timeout=1.0)

    # Then: The stale queued activation resolves to the stop result instead of piggybacking on restart
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not third_thread.is_alive()
    assert start_results["first"] == LivePublisherStatus(
        state=LivePublisherState.IDLE,
        viewer_count=0,
    )
    assert start_results["second"] == LivePublisherStatus(
        state=LivePublisherState.IDLE,
        viewer_count=0,
    )
    third_result = start_results["third"]
    assert isinstance(third_result, LivePublisherStatus)
    assert third_result.state == LivePublisherState.READY
    assert third_result.viewer_count == 0
    assert third_result.idle_shutdown_at is not None
    assert len(calls) == 1
    assert publisher.status() == third_result


def test_waiting_activation_inherits_failed_start_result(tmp_path: Path) -> None:
    """Queued activation callers should share the failed start they were waiting on."""

    # Given: One preview activation probing slowly while a second caller waits behind it
    class _ThreadClock:
        def __init__(self) -> None:
            self.wait_started = Event()

        def now(self) -> float:
            return time.monotonic()

        def sleep(self, seconds: float) -> None:
            self.wait_started.set()
            time.sleep(seconds)

    clock = _ThreadClock()
    discovery = SlowDiscovery(_probe_stream(video_codec="h264", audio_codec="aac"))
    publisher = _make_publisher(tmp_path, clock=clock, discovery=discovery)
    calls: list[dict[str, object]] = []
    start_results: dict[str, LivePublisherStatus | LivePublisherStartRefusal] = {}

    def run_ensure_active(name: str) -> None:
        start_results[name] = publisher.ensure_active()

    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=_fake_popen_factory(
            calls,
            make_output=False,
            returncode=1,
            stderr_text="Too many clients already connected.\n",
        ),
    ):
        first_thread = Thread(target=run_ensure_active, args=("first",))
        second_thread = Thread(target=run_ensure_active, args=("second",))
        first_thread.start()
        assert discovery.started.wait(timeout=0.2)
        second_thread.start()
        assert clock.wait_started.wait(timeout=0.2)

        # When: The in-flight activation fails with a session-budget refusal
        discovery.release.set()
        first_thread.join(timeout=1.0)
        second_thread.join(timeout=1.0)

    # Then: The waiting caller receives the same refusal without starting ffmpeg again
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    first_result = start_results["first"]
    second_result = start_results["second"]
    assert isinstance(first_result, LivePublisherStartRefusal)
    assert first_result.reason == LivePublisherRefusalReason.SESSION_BUDGET_EXHAUSTED
    assert second_result == first_result
    assert len(calls) == 1
    assert publisher.status().state == LivePublisherState.ERROR


def test_waiting_activation_keeps_failed_result_after_later_restart(tmp_path: Path) -> None:
    """Queued activation callers should not inherit a later restart after their start already failed."""

    # Given: One preview activation probing slowly while a second caller waits behind it
    class _ThreadClock:
        def __init__(self) -> None:
            self.wait_started = Event()
            self.release_wait = Event()

        def now(self) -> float:
            return time.monotonic()

        def sleep(self, seconds: float) -> None:
            _ = seconds
            self.wait_started.set()
            self.release_wait.wait(timeout=1.0)

    clock = _ThreadClock()
    discovery = SlowDiscovery(_probe_stream(video_codec="h264", audio_codec="aac"))
    publisher = _make_publisher(tmp_path, clock=clock, discovery=discovery)
    calls: list[dict[str, object]] = []
    start_results: dict[str, LivePublisherStatus | LivePublisherStartRefusal] = {}
    failed_start = _fake_popen_factory(
        calls,
        make_output=False,
        returncode=1,
        stderr_text="Too many clients already connected.\n",
    )
    successful_restart = _fake_popen_factory(calls)

    def run_ensure_active(name: str) -> None:
        start_results[name] = publisher.ensure_active()

    def fake_popen(
        cmd: list[str],
        *,
        stdout: object = None,
        stderr: object = None,
        start_new_session: bool = False,
        **kwargs: object,
    ) -> FakeProc:
        factory = failed_start if len(calls) == 0 else successful_restart
        return factory(
            cmd,
            stdout=stdout,
            stderr=stderr,
            start_new_session=start_new_session,
            **kwargs,
        )

    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=fake_popen,
    ):
        first_thread = Thread(target=run_ensure_active, args=("first",))
        second_thread = Thread(target=run_ensure_active, args=("second",))
        third_thread = Thread(target=run_ensure_active, args=("third",))
        first_thread.start()
        assert discovery.started.wait(timeout=0.2)
        second_thread.start()
        assert clock.wait_started.wait(timeout=0.2)

        # When: The original start fails and a later caller restarts preview before the waiter wakes
        discovery.release.set()
        first_thread.join(timeout=1.0)
        third_thread.start()
        third_thread.join(timeout=1.0)
        clock.release_wait.set()
        second_thread.join(timeout=1.0)

    # Then: The waiting caller still receives the failed start instead of inheriting the restart
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not third_thread.is_alive()
    first_result = start_results["first"]
    second_result = start_results["second"]
    third_result = start_results["third"]
    assert isinstance(first_result, LivePublisherStartRefusal)
    assert first_result.reason == LivePublisherRefusalReason.SESSION_BUDGET_EXHAUSTED
    assert second_result == first_result
    assert isinstance(third_result, LivePublisherStatus)
    assert third_result.state == LivePublisherState.READY
    assert third_result.viewer_count == 0
    assert third_result.idle_shutdown_at is not None
    assert len(calls) == 2
    assert publisher.status() == third_result


def test_recording_priority_preempts_slow_probe_before_spawn(tmp_path: Path) -> None:
    """Recording activation should interrupt startup before ffmpeg spawn when probing is slow."""

    # Given: A publisher whose codec probe is still in flight
    class _ThreadClock:
        def now(self) -> float:
            return time.monotonic()

        def sleep(self, seconds: float) -> None:
            time.sleep(seconds)

    discovery = SlowDiscovery(_probe_stream(video_codec="h264", audio_codec="aac"))
    publisher = _make_publisher(tmp_path, clock=_ThreadClock(), discovery=discovery)
    calls: list[dict[str, object]] = []
    start_result: dict[str, LivePublisherStatus | LivePublisherStartRefusal] = {}

    def run_ensure_active() -> None:
        with patch(
            "homesec.sources.rtsp.live_publisher.subprocess.Popen",
            side_effect=_fake_popen_factory(calls),
        ):
            start_result["result"] = publisher.ensure_active()

    ensure_thread = Thread(target=run_ensure_active)
    ensure_thread.start()
    assert discovery.started.wait(timeout=0.2)

    sync_completed = Event()

    # When: Recording becomes active before the probe has released
    def run_sync_recording_active() -> None:
        publisher.sync_recording_active(True)
        sync_completed.set()

    sync_thread = Thread(target=run_sync_recording_active)
    sync_thread.start()
    sync_thread.join(timeout=0.3)

    discovery.release.set()
    ensure_thread.join(timeout=0.3)

    # Then: Recording-priority preemption finishes without spawning preview ffmpeg
    assert sync_completed.is_set()
    assert not sync_thread.is_alive()
    assert not ensure_thread.is_alive()
    result = start_result["result"]
    assert isinstance(result, LivePublisherStartRefusal)
    assert result.reason == LivePublisherRefusalReason.RECORDING_PRIORITY
    assert calls == []
    assert publisher.status() == LivePublisherStatus(state=LivePublisherState.IDLE, viewer_count=0)


def test_recording_priority_cancels_stale_probe_when_recording_ends_before_release(
    tmp_path: Path,
) -> None:
    """Recording preemption should fence a slow probe even if recording ends before it returns."""

    # Given: A publisher whose codec probe is still in flight
    class _ThreadClock:
        def now(self) -> float:
            return time.monotonic()

        def sleep(self, seconds: float) -> None:
            time.sleep(seconds)

    discovery = SlowDiscovery(_probe_stream(video_codec="h264", audio_codec="aac"))
    publisher = _make_publisher(tmp_path, clock=_ThreadClock(), discovery=discovery)
    calls: list[dict[str, object]] = []
    start_result: dict[str, LivePublisherStatus | LivePublisherStartRefusal] = {}

    def run_ensure_active() -> None:
        with patch(
            "homesec.sources.rtsp.live_publisher.subprocess.Popen",
            side_effect=_fake_popen_factory(calls),
        ):
            start_result["result"] = publisher.ensure_active()

    ensure_thread = Thread(target=run_ensure_active)
    ensure_thread.start()
    assert discovery.started.wait(timeout=0.2)

    # When: Recording briefly preempts preview before the slow probe returns
    publisher.sync_recording_active(True)
    publisher.sync_recording_active(False)
    discovery.release.set()
    ensure_thread.join(timeout=0.3)

    # Then: The stale activation stays cancelled and does not restart preview
    assert not ensure_thread.is_alive()
    result = start_result["result"]
    assert isinstance(result, LivePublisherStartRefusal)
    assert result.reason == LivePublisherRefusalReason.RECORDING_PRIORITY
    assert calls == []
    assert publisher.status() == LivePublisherStatus(state=LivePublisherState.IDLE, viewer_count=0)


def test_recording_priority_cancels_waiting_activation_after_recording_window(
    tmp_path: Path,
) -> None:
    """Recording preemption should fence queued activations after recording has ended."""

    # Given: One preview activation probing slowly while a second caller waits behind it
    class _ThreadClock:
        def __init__(self) -> None:
            self.wait_started = Event()
            self.release_wait = Event()

        def now(self) -> float:
            return time.monotonic()

        def sleep(self, seconds: float) -> None:
            _ = seconds
            self.wait_started.set()
            self.release_wait.wait(timeout=1.0)

    clock = _ThreadClock()
    discovery = SlowDiscovery(_probe_stream(video_codec="h264", audio_codec="aac"))
    publisher = _make_publisher(tmp_path, clock=clock, discovery=discovery)
    calls: list[dict[str, object]] = []
    start_results: dict[str, LivePublisherStatus | LivePublisherStartRefusal] = {}

    def run_ensure_active(name: str) -> None:
        start_results[name] = publisher.ensure_active()

    with patch(
        "homesec.sources.rtsp.live_publisher.subprocess.Popen",
        side_effect=_fake_popen_factory(calls),
    ):
        first_thread = Thread(target=run_ensure_active, args=("first",))
        second_thread = Thread(target=run_ensure_active, args=("second",))
        first_thread.start()
        assert discovery.started.wait(timeout=0.2)
        second_thread.start()
        assert clock.wait_started.wait(timeout=0.2)

        # When: Recording briefly preempts preview before the waiting caller can wake
        publisher.sync_recording_active(True)
        publisher.sync_recording_active(False)
        discovery.release.set()
        first_thread.join(timeout=1.0)
        clock.release_wait.set()
        second_thread.join(timeout=1.0)

    # Then: Both stale activation attempts resolve as recording-priority refusals
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    first_result = start_results["first"]
    second_result = start_results["second"]
    assert isinstance(first_result, LivePublisherStartRefusal)
    assert first_result.reason == LivePublisherRefusalReason.RECORDING_PRIORITY
    assert isinstance(second_result, LivePublisherStartRefusal)
    assert second_result.reason == LivePublisherRefusalReason.RECORDING_PRIORITY
    assert calls == []
    assert publisher.status() == LivePublisherStatus(state=LivePublisherState.IDLE, viewer_count=0)
