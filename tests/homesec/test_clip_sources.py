"""Tests for clip source implementations."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from homesec.models.clip import Clip
from homesec.sources import FtpSource, FtpSourceConfig, LocalFolderSource, LocalFolderSourceConfig


async def _wait_for(
    condition: Callable[[], bool], timeout_s: float = 1.0, interval_s: float = 0.05
) -> None:
    start = time.monotonic()
    while True:
        if condition():
            return
        if time.monotonic() - start > timeout_s:
            raise AssertionError("Condition not met before timeout")
        await asyncio.sleep(interval_s)


class TestLocalFolderSource:
    """Test LocalFolderSource implementation."""

    def _config(self, path: Path, poll_interval: float = 1.0) -> LocalFolderSourceConfig:
        return LocalFolderSourceConfig(
            watch_dir=str(path),
            poll_interval=poll_interval,
            stability_threshold_s=0.01,  # Very low for tests
        )

    @pytest.mark.asyncio
    async def test_callback_registration(self, tmp_path: Path) -> None:
        """Test callback can be registered."""
        # Given a LocalFolderSource
        source = LocalFolderSource(self._config(tmp_path), camera_name="test")

        called = []

        def callback(clip: Clip) -> None:
            called.append(clip)

        # When registering a callback
        source.register_callback(callback)
        # Then the callback is stored
        assert source._callback is callback

    @pytest.mark.asyncio
    async def test_health_check_before_start(self, tmp_path: Path) -> None:
        """Test health check returns True when dir exists."""
        # Given a LocalFolderSource with existing dir
        source = LocalFolderSource(self._config(tmp_path), camera_name="test")
        # When checking health
        healthy = source.is_healthy()
        # Then it is healthy
        assert healthy

    @pytest.mark.asyncio
    async def test_health_check_missing_dir(self, tmp_path: Path) -> None:
        """Test health check returns False when dir is missing."""
        # Given a LocalFolderSource with missing dir
        watch_dir = tmp_path / "nonexistent"
        source = LocalFolderSource(self._config(watch_dir), camera_name="test")

        # When the dir is removed
        watch_dir.rmdir()

        # Then health is False
        assert not source.is_healthy()

    @pytest.mark.asyncio
    async def test_heartbeat_updates(self, tmp_path: Path) -> None:
        """Test heartbeat timestamp updates during polling."""
        # Given a LocalFolderSource
        source = LocalFolderSource(self._config(tmp_path, poll_interval=0.1), camera_name="test")

        initial_heartbeat = source.last_heartbeat()

        # When the source runs briefly
        await source.start()
        await _wait_for(lambda: source.last_heartbeat() > initial_heartbeat, timeout_s=1.0)

        # Then heartbeat is updated
        assert source.last_heartbeat() > initial_heartbeat

        # Cleanup
        await source.shutdown()

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, tmp_path: Path) -> None:
        """Test source can be started and stopped."""
        # Given a LocalFolderSource
        source = LocalFolderSource(self._config(tmp_path, poll_interval=0.1), camera_name="test")

        # When first created
        initial_thread = source._task

        # Then it starts with no thread
        assert initial_thread is None

        # When starting
        await source.start()

        # Then the thread is alive
        assert source._task is not None
        assert not source._task.done()
        assert source.is_healthy()

        # When stopping
        await source.shutdown()

        # Then the thread is cleared
        assert source._task is None

    @pytest.mark.asyncio
    async def test_detects_new_clips(self, tmp_path: Path) -> None:
        """Test source detects new .mp4 files and triggers callback."""
        # Given a LocalFolderSource with a registered callback
        source = LocalFolderSource(self._config(tmp_path, poll_interval=0.1), camera_name="test")

        detected_clips: list[Clip] = []

        def callback(clip: Clip) -> None:
            detected_clips.append(clip)

        source.register_callback(callback)
        await source.start()

        # When a new .mp4 file is created
        clip_file = tmp_path / "test_clip.mp4"
        clip_file.write_bytes(b"fake video data")

        await _wait_for(lambda: len(detected_clips) == 1, timeout_s=2.0)

        # Then the clip is detected
        assert len(detected_clips) == 1
        assert detected_clips[0].clip_id == "test_clip"
        assert detected_clips[0].camera_name == "test"
        assert detected_clips[0].local_path == clip_file
        assert detected_clips[0].source_type == "local_folder"

        # Cleanup
        await source.shutdown()

    @pytest.mark.asyncio
    async def test_ignores_non_video_files(self, tmp_path: Path) -> None:
        """Test source ignores non-.mp4 files."""
        # Given a LocalFolderSource
        source = LocalFolderSource(self._config(tmp_path, poll_interval=0.1), camera_name="test")

        detected_clips: list[Clip] = []
        source.register_callback(lambda c: detected_clips.append(c))
        await source.start()
        initial_heartbeat = source.last_heartbeat()

        # When non-video files are created
        (tmp_path / "test.txt").write_text("not a video")
        (tmp_path / "test.jpg").write_bytes(b"fake image")

        # Then no clips are detected
        await _wait_for(lambda: source.last_heartbeat() > initial_heartbeat, timeout_s=1.0)
        assert len(detected_clips) == 0

        await source.shutdown()

    @pytest.mark.asyncio
    async def test_does_not_detect_same_file_twice(self, tmp_path: Path) -> None:
        """Test source does not trigger callback for same file twice."""
        # Given a LocalFolderSource
        source = LocalFolderSource(self._config(tmp_path, poll_interval=0.1), camera_name="test")

        detected_clips: list[Clip] = []
        source.register_callback(lambda c: detected_clips.append(c))

        # When a file exists before start
        clip_file = tmp_path / "test_clip.mp4"
        clip_file.write_bytes(b"fake video data")

        await source.start()

        await _wait_for(lambda: len(detected_clips) == 1, timeout_s=2.0)

        # Then it is only detected once
        assert len(detected_clips) == 1

        await source.shutdown()

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_crash(self, tmp_path: Path) -> None:
        """Test source continues running if callback raises exception."""
        # Given a LocalFolderSource with a failing callback
        source = LocalFolderSource(self._config(tmp_path, poll_interval=0.1), camera_name="test")

        call_count = []

        def bad_callback(clip: Clip) -> None:
            call_count.append(clip)
            raise ValueError("Simulated error")

        source.register_callback(bad_callback)
        await source.start()

        # When multiple clips are created
        (tmp_path / "clip1.mp4").write_bytes(b"video")
        await _wait_for(lambda: len(call_count) >= 1, timeout_s=2.0)

        (tmp_path / "clip2.mp4").write_bytes(b"video")
        await _wait_for(lambda: len(call_count) >= 2, timeout_s=2.0)

        # Then both clips are detected despite the callback error
        assert len(call_count) == 2

        # Then source remains healthy
        assert source.is_healthy()

        await source.shutdown()

    @pytest.mark.asyncio
    async def test_stability_threshold_delays_detection(self, tmp_path: Path) -> None:
        """Test files are ignored until they stabilize."""
        # Given a LocalFolderSource with a stability threshold
        config = LocalFolderSourceConfig(
            watch_dir=str(tmp_path),
            poll_interval=0.05,
            stability_threshold_s=0.3,
        )
        source = LocalFolderSource(config, camera_name="test")
        detected: list[Clip] = []
        source.register_callback(lambda clip: detected.append(clip))
        await source.start()
        initial_heartbeat = source.last_heartbeat()

        # When a new clip appears and is too new
        clip_file = tmp_path / "unstable.mp4"
        clip_file.write_bytes(b"video")
        await _wait_for(lambda: source.last_heartbeat() > initial_heartbeat, timeout_s=1.0)

        # Then it is not detected yet
        assert detected == []

        # When the file becomes old enough
        await _wait_for(lambda: len(detected) == 1, timeout_s=1.0)

        # Then it is detected
        assert detected[0].clip_id == "unstable"
        await source.shutdown()


class TestFtpSource:
    """Test FtpSource behavior."""

    @pytest.mark.asyncio
    async def test_rejects_non_matching_extension(self, tmp_path: Path) -> None:
        """Rejects unsupported file extensions and deletes when configured."""
        # Given a FtpSource that deletes non-matching uploads
        config = FtpSourceConfig(
            root_dir=str(tmp_path),
            allowed_extensions=[".mp4"],
            delete_non_matching=True,
        )
        source = FtpSource(config, camera_name="ftp_cam")
        emitted: list[Clip] = []
        source.register_callback(lambda clip: emitted.append(clip))

        # When a non-mp4 file is received
        bad_file = tmp_path / "upload.txt"
        bad_file.write_text("nope")
        source._handle_file_received(bad_file)

        # Then no clip is emitted and file is deleted
        assert emitted == []
        assert not bad_file.exists()

    @pytest.mark.asyncio
    async def test_accepts_allowed_extension(self, tmp_path: Path) -> None:
        """Accepts allowed extensions and emits clips."""
        # Given a FtpSource that accepts .mp4
        config = FtpSourceConfig(root_dir=str(tmp_path), allowed_extensions=[".mp4"])
        source = FtpSource(config, camera_name="ftp_cam")
        emitted: list[Clip] = []
        source.register_callback(lambda clip: emitted.append(clip))

        # When an allowed file is received
        clip_path = tmp_path / "upload.mp4"
        clip_path.write_bytes(b"video")
        source._handle_file_received(clip_path)

        # Then a clip is emitted
        assert len(emitted) == 1
        assert emitted[0].clip_id == "upload"
        assert emitted[0].source_type == "ftp"

    @pytest.mark.asyncio
    async def test_incomplete_upload_deletes_when_enabled(self, tmp_path: Path) -> None:
        """Deletes incomplete uploads when configured."""
        # Given a FtpSource with delete_incomplete enabled
        config = FtpSourceConfig(root_dir=str(tmp_path), delete_incomplete=True)
        source = FtpSource(config, camera_name="ftp_cam")

        # When an incomplete upload is received
        incomplete = tmp_path / "partial.mp4"
        incomplete.write_bytes(b"video")
        source._handle_incomplete_file(incomplete)

        # Then the file is deleted
        assert not incomplete.exists()
