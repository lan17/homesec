"""End-to-end integration tests for HomeSec pipeline."""

from __future__ import annotations

import asyncio
import ftplib
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

import pytest

from homesec.config import load_config_from_dict
from homesec.models.config import DefaultAlertPolicySettings
from homesec.models.filter import FilterResult
from homesec.pipeline import ClipPipeline
from homesec.plugins.alert_policies.default import DefaultAlertPolicy
from homesec.repository import ClipRepository
from homesec.sources import (
    FtpSource,
    FtpSourceConfig,
    LocalFolderSource,
    LocalFolderSourceConfig,
)
from tests.homesec.mocks import (
    MockEventStore,
    MockFilter,
    MockNotifier,
    MockStateStore,
    MockStorage,
    MockVLM,
)

if TYPE_CHECKING:
    from homesec.models.config import Config


async def _wait_for(
    predicate: Callable[[], bool], timeout_s: float = 5.0, interval_s: float = 0.1
) -> None:
    start = asyncio.get_running_loop().time()
    while True:
        if predicate():
            return
        if asyncio.get_running_loop().time() - start > timeout_s:
            raise AssertionError("Condition not met before timeout")
        await asyncio.sleep(interval_s)


@pytest.fixture
def integration_config() -> Config:
    """Create config for integration tests."""
    return load_config_from_dict(
        {
            "version": 1,
            "cameras": [
                {
                    "name": "test_camera",
                    "source": {
                        "type": "local_folder",
                        "config": {
                            "watch_dir": "recordings",
                            "poll_interval": 0.1,
                            "stability_threshold_s": 0.01,
                        },
                    },
                }
            ],
            "storage": {
                "backend": "dropbox",
                "dropbox": {
                    "root": "/homecam",
                },
            },
            "state_store": {
                "dsn": "postgresql://user:pass@localhost/db",
            },
            "notifiers": [
                {
                    "backend": "mqtt",
                    "config": {
                        "host": "localhost",
                        "port": 1883,
                    },
                }
            ],
            "filter": {
                "plugin": "yolo",
                "max_workers": 1,
                "config": {
                    "model_path": "yolov8n.pt",
                },
            },
            "vlm": {
                "backend": "openai",
                "trigger_classes": ["person"],
                "max_workers": 1,
                "llm": {
                    "api_key_env": "OPENAI_API_KEY",
                    "model": "gpt-4o",
                },
            },
            "alert_policy": {
                "backend": "default",
                "config": {
                    "min_risk_level": "low",
                    "notify_on_motion": False,
                },
            },
        }
    )


def make_alert_policy(config: Config) -> DefaultAlertPolicy:
    settings = DefaultAlertPolicySettings.model_validate(config.alert_policy.config)
    settings = DefaultAlertPolicySettings.model_validate(config.alert_policy.config)
    settings.overrides = config.per_camera_alert
    settings.trigger_classes = set(config.vlm.trigger_classes)
    return DefaultAlertPolicy(settings)


class TestFullPipelineIntegration:
    """Test full pipeline flow with LocalFolderSource."""

    @pytest.mark.asyncio
    async def test_clip_flows_through_pipeline(
        self, integration_config: Config, tmp_path: Path
    ) -> None:
        """Test that a clip dropped into folder flows through entire pipeline."""
        # Given a pipeline wired with mocks and a LocalFolderSource
        filter_result = FilterResult(
            detected_classes=["person"],
            confidence=0.9,
            model="mock",
            sampled_frames=30,
        )
        storage = MockStorage()
        state_store = MockStateStore()
        event_store = MockEventStore()
        repository = ClipRepository(state_store, event_store, retry=integration_config.retry)
        filter_plugin = MockFilter(result=filter_result)
        vlm_plugin = MockVLM()
        notifier = MockNotifier()

        # Create pipeline
        pipeline = ClipPipeline(
            config=integration_config,
            storage=storage,
            repository=repository,
            filter_plugin=filter_plugin,
            vlm_plugin=vlm_plugin,
            notifier=notifier,
            alert_policy=make_alert_policy(integration_config),
        )
        pipeline.set_event_loop(asyncio.get_running_loop())

        source = LocalFolderSource(
            LocalFolderSourceConfig(
                watch_dir=str(tmp_path),
                poll_interval=0.1,
                stability_threshold_s=0.01,
            ),
            camera_name="test_camera",
        )
        source.register_callback(pipeline.on_new_clip)
        await source.start()

        try:
            # When a clip file is dropped in the watch folder
            clip_file = tmp_path / "test_clip.mp4"
            clip_file.write_bytes(b"fake video content")

            await _wait_for(lambda: len(notifier.sent_alerts) == 1, timeout_s=5.0)

            # Then notification is sent and state is stored
            assert len(notifier.sent_alerts) == 1
            alert = notifier.sent_alerts[0]
            assert alert.camera_name == "test_camera"
            assert alert.clip_id == "test_clip"

            state = await state_store.get("test_clip")
            assert state is not None
            assert state.status == "done"
            assert state.filter_result is not None
            assert state.analysis_result is not None

        finally:
            await source.shutdown()
            await pipeline.shutdown()

    @pytest.mark.asyncio
    async def test_multiple_clips_processed_concurrently(
        self, integration_config: Config, tmp_path: Path
    ) -> None:
        """Test that multiple clips can be processed."""
        # Given a pipeline wired with delayed mocks
        filter_result = FilterResult(
            detected_classes=["person"],
            confidence=0.9,
            model="mock",
            sampled_frames=30,
        )
        storage = MockStorage(delay_s=0.05)
        state_store = MockStateStore()
        event_store = MockEventStore()
        repository = ClipRepository(state_store, event_store, retry=integration_config.retry)
        filter_plugin = MockFilter(result=filter_result, delay_s=0.05)
        vlm_plugin = MockVLM(delay_s=0.05)
        notifier = MockNotifier()

        pipeline = ClipPipeline(
            config=integration_config,
            storage=storage,
            repository=repository,
            filter_plugin=filter_plugin,
            vlm_plugin=vlm_plugin,
            notifier=notifier,
            alert_policy=make_alert_policy(integration_config),
        )
        pipeline.set_event_loop(asyncio.get_running_loop())

        source = LocalFolderSource(
            LocalFolderSourceConfig(
                watch_dir=str(tmp_path),
                poll_interval=0.05,
                stability_threshold_s=0.01,
            ),
            camera_name="test_camera",
        )
        source.register_callback(pipeline.on_new_clip)
        await source.start()

        try:
            # When multiple clips are dropped
            for i in range(3):
                clip_file = tmp_path / f"clip_{i}.mp4"
                clip_file.write_bytes(f"video content {i}".encode())

            await _wait_for(lambda: len(notifier.sent_alerts) >= 3, timeout_s=10.0)

            # Then all clips are processed
            assert len(notifier.sent_alerts) == 3
            clip_ids = {a.clip_id for a in notifier.sent_alerts}
            assert clip_ids == {"clip_0", "clip_1", "clip_2"}

        finally:
            await source.shutdown()
            await pipeline.shutdown()


class TestFtpSourceIntegration:
    """Test FTP source behavior with real uploads."""

    @pytest.mark.asyncio
    async def test_ftp_upload_emits_clip(self, tmp_path: Path) -> None:
        """FTP upload should trigger clip callback."""
        # Given an FTP source and a callback
        root_dir = tmp_path / "ftp_root"
        config = FtpSourceConfig(
            host="127.0.0.1",
            port=0,
            root_dir=str(root_dir),
        )
        source = FtpSource(config, camera_name="ftp_cam")

        clips = []
        ready = Event()

        def callback(clip) -> None:
            clips.append(clip)
            ready.set()

        source.register_callback(callback)
        await source.start()

        ftp = ftplib.FTP()
        try:
            server = source._server
            assert server is not None
            port = server.socket.getsockname()[1]

            connected = False
            for _ in range(20):
                try:
                    ftp.connect("127.0.0.1", port, timeout=2.0)
                    connected = True
                    break
                except OSError:
                    time.sleep(0.1)

            assert connected, "FTP server did not start"
            ftp.login()

            # When a file is uploaded
            local_file = tmp_path / "upload.mp4"
            local_file.write_bytes(b"fake video data")
            with local_file.open("rb") as handle:
                ftp.storbinary("STOR upload.mp4", handle)
            ftp.quit()

            # Then a clip is emitted
            assert ready.wait(5.0)
            assert len(clips) == 1
            clip = clips[0]
            assert clip.camera_name == "ftp_cam"
            assert clip.source_type == "ftp"
            assert clip.local_path.exists()
        finally:
            try:
                ftp.close()
            except Exception:
                pass
            await source.shutdown()
