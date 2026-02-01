"""Tests for ClipPipeline core."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from homesec.errors import NotifyError
from homesec.models.clip import Clip, ClipStateData
from homesec.models.config import (
    AlertPolicyConfig,
    CameraConfig,
    CameraSourceConfig,
    Config,
    NotifierConfig,
    RetryConfig,
    StateStoreConfig,
    StorageConfig,
)
from homesec.models.enums import RiskLevel
from homesec.models.filter import FilterConfig, FilterOverrides, FilterResult
from homesec.models.vlm import AnalysisResult, VLMConfig
from homesec.pipeline import ClipPipeline
from homesec.plugins.alert_policies.default import DefaultAlertPolicy, DefaultAlertPolicySettings
from homesec.plugins.analyzers.openai import OpenAIConfig
from homesec.plugins.filters.yolo import YoloFilterConfig
from homesec.plugins.storage.dropbox import DropboxStorageConfig
from homesec.repository import ClipRepository
from tests.homesec.mocks import (
    MockEventStore,
    MockFilter,
    MockNotifier,
    MockStateStore,
    MockStorage,
    MockVLM,
)


@dataclass
class PipelineMocks:
    storage: MockStorage
    state_store: MockStateStore
    event_store: MockEventStore
    filter: MockFilter
    vlm: MockVLM
    notifier: MockNotifier


def make_repository(config: Config, mocks: PipelineMocks) -> ClipRepository:
    """Create a ClipRepository with test retry config."""
    return ClipRepository(mocks.state_store, mocks.event_store, retry=config.retry)


def make_alert_policy(config: Config) -> DefaultAlertPolicy:
    """Build the default alert policy from config."""
    settings = DefaultAlertPolicySettings.model_validate(config.alert_policy.config)
    # Inject runtime fields as the registry would
    settings.trigger_classes = list(config.vlm.trigger_classes)
    return DefaultAlertPolicy(settings)


@pytest.fixture
def base_config() -> Config:
    """Create a base Config for tests."""
    cameras = [
        CameraConfig(
            name="front_door",
            source=CameraSourceConfig(
                backend="local_folder",
                config={
                    "watch_dir": "recordings",
                    "poll_interval": 1.0,
                },
            ),
        )
    ]
    return Config(
        cameras=cameras,
        storage=StorageConfig(
            backend="dropbox",
            config=DropboxStorageConfig(root="/homecam"),
        ),
        state_store=StateStoreConfig(dsn="postgresql://user:pass@localhost/db"),
        notifiers=[
            NotifierConfig(
                backend="mqtt",
                config={"host": "localhost"},
            )
        ],
        filter=FilterConfig(
            backend="yolo",
            config=YoloFilterConfig(model_path="yolov8n.pt"),
        ),
        vlm=VLMConfig(
            backend="openai",
            trigger_classes=["person", "car"],
            config=OpenAIConfig(
                api_key_env="OPENAI_API_KEY",
                model="gpt-4o",
            ),
        ),
        alert_policy=AlertPolicyConfig(
            backend="default",
            config={
                "min_risk_level": "low",
                "notify_on_motion": False,
            },
        ),
        retry=RetryConfig(max_attempts=1, backoff_s=0.0),
    )


@pytest.fixture
def sample_clip(tmp_path: Path) -> Clip:
    """Create a sample clip for testing."""
    from datetime import datetime, timedelta

    video_path = tmp_path / "test_clip.mp4"
    video_path.write_bytes(b"fake video content")
    start_ts = datetime.now()
    return Clip(
        clip_id="test-clip-001",
        camera_name="front_door",
        local_path=video_path,
        start_ts=start_ts,
        end_ts=start_ts + timedelta(seconds=10),
        duration_s=10.0,
        source_backend="mock",
    )


@pytest.fixture
def mocks() -> PipelineMocks:
    """Create all mock dependencies."""
    person_result = FilterResult(
        detected_classes=["person"],
        confidence=0.9,
        model="mock",
        sampled_frames=30,
    )
    return PipelineMocks(
        storage=MockStorage(),
        state_store=MockStateStore(),
        event_store=MockEventStore(),
        filter=MockFilter(result=person_result),
        vlm=MockVLM(),
        notifier=MockNotifier(),
    )


class TestClipPipelineHappyPath:
    """Test successful processing paths."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_person_detection(
        self, base_config: Config, sample_clip: Clip, mocks: PipelineMocks
    ) -> None:
        """Full pipeline: person detected -> VLM runs -> notification sent."""
        # Given a pipeline with all mock dependencies
        pipeline = ClipPipeline(
            config=base_config,
            storage=mocks.storage,
            repository=make_repository(base_config, mocks),
            filter_plugin=mocks.filter,
            vlm_plugin=mocks.vlm,
            notifier=mocks.notifier,
            alert_policy=make_alert_policy(base_config),
        )

        # When a new clip is processed
        pipeline.on_new_clip(sample_clip)
        await pipeline.shutdown()

        # Then state is stored and notification is sent
        state_store: MockStateStore = mocks.state_store
        state = await state_store.get(sample_clip.clip_id)
        assert state is not None
        assert state.status == "done"
        assert state.filter_result is not None
        assert state.filter_result.detected_classes == ["person"]
        assert state.analysis_result is not None
        assert state.storage_uri is not None
        assert state.view_url is not None

        notifier: MockNotifier = mocks.notifier
        assert len(notifier.sent_alerts) == 1
        alert = notifier.sent_alerts[0]
        assert alert.clip_id == sample_clip.clip_id
        assert alert.camera_name == "front_door"
        assert not alert.upload_failed

    @pytest.mark.asyncio
    async def test_vlm_skipped_when_no_trigger_classes(
        self, base_config: Config, sample_clip: Clip, mocks: PipelineMocks
    ) -> None:
        """VLM should be skipped if filter detects no trigger classes."""
        # Given a filter result with no trigger classes
        dog_result = FilterResult(
            detected_classes=["dog"],
            confidence=0.9,
            model="mock",
            sampled_frames=30,
        )
        filter_mock = MockFilter(result=dog_result)
        mocks.filter = filter_mock

        pipeline = ClipPipeline(
            config=base_config,
            storage=mocks.storage,
            repository=make_repository(base_config, mocks),
            filter_plugin=filter_mock,
            vlm_plugin=mocks.vlm,
            notifier=mocks.notifier,
            alert_policy=make_alert_policy(base_config),
        )

        # When a clip is processed
        pipeline.on_new_clip(sample_clip)
        await pipeline.shutdown()

        # Then VLM is skipped and no analysis is stored
        state_store: MockStateStore = mocks.state_store
        state = await state_store.get(sample_clip.clip_id)
        assert state is not None
        assert state.analysis_result is None

    @pytest.mark.asyncio
    async def test_run_mode_always_runs_vlm_regardless(
        self, base_config: Config, sample_clip: Clip, mocks: PipelineMocks
    ) -> None:
        """When run_mode=always, VLM runs even without trigger classes."""
        # Given run_mode=always and a non-trigger filter result
        base_config = Config(
            cameras=base_config.cameras,
            storage=base_config.storage,
            state_store=base_config.state_store,
            notifiers=base_config.notifiers,
            filter=base_config.filter,
            vlm=base_config.vlm.model_copy(update={"run_mode": "always"}),
            alert_policy=AlertPolicyConfig(
                backend="default",
                config={
                    "min_risk_level": "low",
                    "notify_on_motion": False,
                },
            ),
        )
        motion_result = FilterResult(
            detected_classes=["motion_only"],
            confidence=0.5,
            model="mock",
            sampled_frames=30,
        )
        filter_mock = MockFilter(result=motion_result)
        mocks.filter = filter_mock

        pipeline = ClipPipeline(
            config=base_config,
            storage=mocks.storage,
            repository=make_repository(base_config, mocks),
            filter_plugin=filter_mock,
            vlm_plugin=mocks.vlm,
            notifier=mocks.notifier,
            alert_policy=make_alert_policy(base_config),
        )

        # When a clip is processed
        pipeline.on_new_clip(sample_clip)
        await pipeline.shutdown()

        # Then VLM runs and analysis is present
        state_store: MockStateStore = mocks.state_store
        state = await state_store.get(sample_clip.clip_id)
        assert state is not None
        assert state.analysis_result is not None


class TestClipPipelineErrorHandling:
    """Test error handling and partial failures."""

    @pytest.mark.asyncio
    async def test_upload_failure_does_not_block_processing(
        self, base_config: Config, sample_clip: Clip, mocks: PipelineMocks
    ) -> None:
        """Upload failure should not block filter/VLM/notify."""
        # Given storage fails on upload
        storage_mock = MockStorage(simulate_failure=True)
        mocks.storage = storage_mock

        pipeline = ClipPipeline(
            config=base_config,
            storage=storage_mock,
            repository=make_repository(base_config, mocks),
            filter_plugin=mocks.filter,
            vlm_plugin=mocks.vlm,
            notifier=mocks.notifier,
            alert_policy=make_alert_policy(base_config),
        )

        # When a clip is processed
        pipeline.on_new_clip(sample_clip)
        await pipeline.shutdown()

        # Then processing completes and notify marks upload_failed
        state_store: MockStateStore = mocks.state_store
        state = await state_store.get(sample_clip.clip_id)
        assert state is not None
        assert state.status == "done"
        assert state.storage_uri is None
        assert state.view_url is None

        notifier: MockNotifier = mocks.notifier
        assert len(notifier.sent_alerts) == 1
        assert notifier.sent_alerts[0].upload_failed

    @pytest.mark.asyncio
    async def test_state_store_failure_does_not_abort_processing(
        self, base_config: Config, sample_clip: Clip, mocks: PipelineMocks
    ) -> None:
        """State store failures should not abort clip processing."""

        class FailingStateStore(MockStateStore):
            async def upsert(self, clip_id: str, data: ClipStateData) -> None:
                raise RuntimeError("Simulated state store upsert failure")

        # Given a pipeline with a failing state store
        pipeline = ClipPipeline(
            config=base_config,
            storage=mocks.storage,
            repository=ClipRepository(
                FailingStateStore(), mocks.event_store, retry=base_config.retry
            ),
            filter_plugin=mocks.filter,
            vlm_plugin=mocks.vlm,
            notifier=mocks.notifier,
            alert_policy=make_alert_policy(base_config),
        )

        # When a clip is processed
        pipeline.on_new_clip(sample_clip)
        await pipeline.shutdown()

        # Then notifications still send
        notifier: MockNotifier = mocks.notifier
        assert len(notifier.sent_alerts) == 1

    @pytest.mark.asyncio
    async def test_filter_failure_aborts_processing(
        self, base_config: Config, sample_clip: Clip, mocks: PipelineMocks
    ) -> None:
        """Filter failure should abort clip processing entirely."""
        # Given a filter that fails
        filter_mock = MockFilter(simulate_failure=True)
        mocks.filter = filter_mock

        pipeline = ClipPipeline(
            config=base_config,
            storage=mocks.storage,
            repository=make_repository(base_config, mocks),
            filter_plugin=filter_mock,
            vlm_plugin=mocks.vlm,
            notifier=mocks.notifier,
            alert_policy=make_alert_policy(base_config),
        )

        # When a clip is processed
        pipeline.on_new_clip(sample_clip)
        await pipeline.shutdown()

        # Then processing aborts and no notification is sent
        state_store: MockStateStore = mocks.state_store
        state = await state_store.get(sample_clip.clip_id)
        assert state is not None
        assert state.status == "error"

        notifier: MockNotifier = mocks.notifier
        assert len(notifier.sent_alerts) == 0

    @pytest.mark.asyncio
    async def test_vlm_failure_continues_to_notify(
        self, base_config: Config, sample_clip: Clip, mocks: PipelineMocks
    ) -> None:
        """VLM failure should not block notification when notify_on_motion=True."""
        # Given notify_on_motion is enabled and VLM fails
        config = Config(
            cameras=base_config.cameras,
            storage=base_config.storage,
            state_store=base_config.state_store,
            notifiers=base_config.notifiers,
            filter=base_config.filter,
            vlm=base_config.vlm,
            alert_policy=AlertPolicyConfig(
                backend="default",
                config={
                    "min_risk_level": "low",
                    "notify_on_motion": True,
                },  # Notify even without VLM
            ),
        )

        vlm_mock = MockVLM(simulate_failure=True)
        mocks.vlm = vlm_mock

        pipeline = ClipPipeline(
            config=config,
            storage=mocks.storage,
            repository=make_repository(config, mocks),
            filter_plugin=mocks.filter,
            vlm_plugin=vlm_mock,
            notifier=mocks.notifier,
            alert_policy=make_alert_policy(config),
        )

        # When a clip is processed
        pipeline.on_new_clip(sample_clip)
        await pipeline.shutdown()

        # Then processing completes and notification is sent without analysis
        state_store: MockStateStore = mocks.state_store
        state = await state_store.get(sample_clip.clip_id)
        assert state is not None
        assert state.status == "done"

        notifier: MockNotifier = mocks.notifier
        assert len(notifier.sent_alerts) == 1
        alert = notifier.sent_alerts[0]
        assert alert.risk_level is None  # No VLM analysis
        assert alert.activity_type is None
        assert alert.vlm_failed is True
        assert alert.notify_reason == "notify_on_motion=true"

    @pytest.mark.asyncio
    async def test_vlm_failure_falls_back_to_filter_trigger(
        self, base_config: Config, sample_clip: Clip, mocks: PipelineMocks
    ) -> None:
        """VLM failure should still notify when filter hits trigger classes."""
        # Given notify_on_motion is disabled and VLM fails
        config = Config(
            cameras=base_config.cameras,
            storage=base_config.storage,
            state_store=base_config.state_store,
            notifiers=base_config.notifiers,
            filter=base_config.filter,
            vlm=base_config.vlm,
            alert_policy=AlertPolicyConfig(
                backend="default",
                config={
                    "min_risk_level": "low",
                    "notify_on_motion": False,
                },
            ),
        )
        vlm_mock = MockVLM(simulate_failure=True)
        mocks.vlm = vlm_mock

        pipeline = ClipPipeline(
            config=config,
            storage=mocks.storage,
            repository=make_repository(config, mocks),
            filter_plugin=mocks.filter,
            vlm_plugin=vlm_mock,
            notifier=mocks.notifier,
            alert_policy=make_alert_policy(config),
        )

        # When a clip is processed
        pipeline.on_new_clip(sample_clip)
        await pipeline.shutdown()

        # Then the alert falls back to filter triggers
        notifier: MockNotifier = mocks.notifier
        assert len(notifier.sent_alerts) == 1
        alert = notifier.sent_alerts[0]
        assert alert.vlm_failed is True
        assert alert.notify_reason == "filter_detected_trigger_vlm_failed"

    @pytest.mark.asyncio
    async def test_notify_failure_still_marks_done(
        self, base_config: Config, sample_clip: Clip, mocks: PipelineMocks
    ) -> None:
        """Notify failure should still mark clip as done."""
        # Given a notifier that fails
        notifier_mock = MockNotifier(simulate_failure=True)
        mocks.notifier = notifier_mock

        pipeline = ClipPipeline(
            config=base_config,
            storage=mocks.storage,
            repository=make_repository(base_config, mocks),
            filter_plugin=mocks.filter,
            vlm_plugin=mocks.vlm,
            notifier=notifier_mock,
            alert_policy=make_alert_policy(base_config),
        )

        # When a clip is processed
        pipeline.on_new_clip(sample_clip)
        await pipeline.shutdown()

        # Then processing completes and notify stage is error
        state_store: MockStateStore = mocks.state_store
        state = await state_store.get(sample_clip.clip_id)
        assert state is not None
        assert state.status == "done"


class TestClipPipelineRetries:
    """Test retry behavior for pipeline stages."""

    @pytest.mark.asyncio
    async def test_stage_retries_succeed(
        self, base_config: Config, sample_clip: Clip, mocks: PipelineMocks
    ) -> None:
        """Retry config should allow transient failures to recover."""
        # Given retry config with two attempts and no backoff
        base_config.retry = RetryConfig(max_attempts=2, backoff_s=0.0)

        class FlakyStorage(MockStorage):
            def __init__(self) -> None:
                super().__init__(simulate_failure=False)
                self.failures_remaining = 1

            async def put_file(self, local_path: Path, dest_path: str):
                if self.failures_remaining > 0:
                    self.failures_remaining -= 1
                    self.put_count += 1
                    raise RuntimeError("Simulated storage failure")
                return await super().put_file(local_path, dest_path)

        class FlakyFilter(MockFilter):
            def __init__(self, result: FilterResult) -> None:
                super().__init__(simulate_failure=False, result=result)
                self.failures_remaining = 1

            async def detect(
                self, video_path: Path, overrides: FilterOverrides | None = None
            ) -> FilterResult:
                self.detect_count += 1
                if self.failures_remaining > 0:
                    self.failures_remaining -= 1
                    raise RuntimeError("Simulated filter failure")
                return self.result

        class FlakyVLM(MockVLM):
            def __init__(self, result: AnalysisResult) -> None:
                super().__init__(simulate_failure=False, result=result)
                self.failures_remaining = 1

            async def analyze(
                self, video_path: Path, filter_result: FilterResult, config: VLMConfig
            ) -> AnalysisResult:
                self.analyze_count += 1
                if self.failures_remaining > 0:
                    self.failures_remaining -= 1
                    raise RuntimeError("Simulated VLM failure")
                return self.result

        class FlakyNotifier(MockNotifier):
            def __init__(self) -> None:
                super().__init__(simulate_failure=False)
                self.failures_remaining = 1
                self.send_calls = 0

            async def send(self, alert) -> None:
                self.send_calls += 1
                if self.failures_remaining > 0:
                    self.failures_remaining -= 1
                    raise NotifyError(
                        clip_id=alert.clip_id,
                        notifier_name="flaky_notifier",
                        cause=RuntimeError("Simulated notifier failure"),
                    )
                await super().send(alert)

        # Given a pipeline with flaky dependencies
        person_result = FilterResult(
            detected_classes=["person"],
            confidence=0.9,
            model="mock",
            sampled_frames=30,
        )
        analysis_result = AnalysisResult(
            risk_level="low",
            activity_type="person_passing",
            summary="Person walked through frame",
        )
        storage = FlakyStorage()
        filter_plugin = FlakyFilter(result=person_result)
        vlm_plugin = FlakyVLM(result=analysis_result)
        notifier = FlakyNotifier()

        pipeline = ClipPipeline(
            config=base_config,
            storage=storage,
            repository=make_repository(base_config, mocks),
            filter_plugin=filter_plugin,
            vlm_plugin=vlm_plugin,
            notifier=notifier,
            alert_policy=make_alert_policy(base_config),
        )

        # When a clip is processed
        pipeline.on_new_clip(sample_clip)
        await pipeline.shutdown()

        # Then retries succeed and attempts are recorded
        state_store: MockStateStore = mocks.state_store
        state = await state_store.get(sample_clip.clip_id)
        assert state is not None
        assert state.status == "done"
        assert storage.put_count == 2
        assert filter_plugin.detect_count == 2
        assert vlm_plugin.analyze_count == 2
        assert notifier.send_calls == 2
        assert len(notifier.sent_alerts) == 1

    @pytest.mark.asyncio
    async def test_state_store_upsert_retries(
        self, base_config: Config, sample_clip: Clip, mocks: PipelineMocks
    ) -> None:
        """State store upsert should retry on transient failures."""
        # Given retry config with three attempts
        base_config.retry = RetryConfig(max_attempts=3, backoff_s=0.0)

        class RetryableStateStoreError(RuntimeError):
            sqlstate = "40001"

        class FlakyStateStore(MockStateStore):
            def __init__(self) -> None:
                super().__init__(simulate_failure=False)
                self.failures_remaining = 2

            async def upsert(self, clip_id: str, data: ClipStateData) -> None:
                self.upsert_count += 1
                if self.failures_remaining > 0:
                    self.failures_remaining -= 1
                    raise RetryableStateStoreError("Simulated state store failure")
                self.states[clip_id] = data

        state_store = FlakyStateStore()
        repository = ClipRepository(state_store, mocks.event_store, retry=base_config.retry)

        # When persisting state
        state = await repository.initialize_clip(sample_clip)

        # Then upsert is retried and succeeds
        assert state_store.upsert_count == 3
        assert state_store.states[sample_clip.clip_id] is state


class TestClipPipelineAlertPolicy:
    """Test alert policy integration."""

    @pytest.mark.asyncio
    async def test_no_notification_when_below_risk_threshold(
        self, base_config: Config, sample_clip: Clip, mocks: PipelineMocks
    ) -> None:
        """No notification when VLM risk level is below threshold."""
        # Given a high risk threshold
        base_config = Config(
            cameras=base_config.cameras,
            storage=base_config.storage,
            state_store=base_config.state_store,
            notifiers=base_config.notifiers,
            filter=base_config.filter,
            vlm=base_config.vlm,
            alert_policy=AlertPolicyConfig(
                backend="default",
                config={
                    "min_risk_level": "high",  # Only notify on high risk
                    "notify_on_motion": False,
                },
            ),
        )

        # Given VLM returns a medium risk result
        medium_result = AnalysisResult(
            risk_level="medium",
            activity_type="person_walking",
            summary="Person walking past.",
        )
        vlm_mock = MockVLM(result=medium_result)
        mocks.vlm = vlm_mock

        pipeline = ClipPipeline(
            config=base_config,
            storage=mocks.storage,
            repository=make_repository(base_config, mocks),
            filter_plugin=mocks.filter,
            vlm_plugin=vlm_mock,
            notifier=mocks.notifier,
            alert_policy=make_alert_policy(base_config),
        )

        # When a clip is processed
        pipeline.on_new_clip(sample_clip)
        await pipeline.shutdown()

        # Then no notification is sent
        notifier: MockNotifier = mocks.notifier
        assert len(notifier.sent_alerts) == 0

        # Then notify stage is skipped in state
        state_store: MockStateStore = mocks.state_store
        state = await state_store.get(sample_clip.clip_id)
        assert state is not None

    @pytest.mark.asyncio
    async def test_notification_sent_when_above_risk_threshold(
        self, base_config: Config, sample_clip: Clip, mocks: PipelineMocks
    ) -> None:
        """Notification sent when VLM risk level meets threshold."""
        # Given VLM returns a high risk result
        high_result = AnalysisResult(
            risk_level="high",
            activity_type="person_loitering",
            summary="Person loitering suspiciously.",
        )
        vlm_mock = MockVLM(result=high_result)
        mocks.vlm = vlm_mock

        pipeline = ClipPipeline(
            config=base_config,
            storage=mocks.storage,
            repository=make_repository(base_config, mocks),
            filter_plugin=mocks.filter,
            vlm_plugin=vlm_mock,
            notifier=mocks.notifier,
            alert_policy=make_alert_policy(base_config),
        )

        # When a clip is processed
        pipeline.on_new_clip(sample_clip)
        await pipeline.shutdown()

        # Then notification is sent with high risk
        notifier: MockNotifier = mocks.notifier
        assert len(notifier.sent_alerts) == 1
        alert = notifier.sent_alerts[0]
        assert alert.risk_level == RiskLevel.HIGH
        assert alert.activity_type == "person_loitering"


class TestClipPipelineAlertOverrides:
    """Test per-camera alert policy overrides."""

    @pytest.mark.asyncio
    async def test_per_camera_notify_on_motion_sends_alert_without_vlm(
        self, base_config: Config, sample_clip: Clip, mocks: PipelineMocks
    ) -> None:
        """Per-camera notify_on_motion should send alert even without VLM."""
        # Given per-camera override enables notify_on_motion
        base_config = Config(
            cameras=base_config.cameras,
            storage=base_config.storage,
            state_store=base_config.state_store,
            notifiers=base_config.notifiers,
            filter=base_config.filter,
            vlm=base_config.vlm,
            alert_policy=AlertPolicyConfig(
                backend="default",
                config={
                    "min_risk_level": "low",
                    "notify_on_motion": False,
                    "overrides": {"front_door": {"notify_on_motion": True}},
                },
            ),
        )

        # Given filter detects class not in trigger_classes
        motion_result = FilterResult(
            detected_classes=["dog"],
            confidence=0.9,
            model="mock",
            sampled_frames=30,
        )
        filter_mock = MockFilter(result=motion_result)
        mocks.filter = filter_mock

        pipeline = ClipPipeline(
            config=base_config,
            storage=mocks.storage,
            repository=make_repository(base_config, mocks),
            filter_plugin=filter_mock,
            vlm_plugin=mocks.vlm,
            notifier=mocks.notifier,
            alert_policy=make_alert_policy(base_config),
        )

        # When a clip is processed
        pipeline.on_new_clip(sample_clip)
        await pipeline.shutdown()

        # Then alert is sent even though VLM did not run
        notifier: MockNotifier = mocks.notifier
        assert len(notifier.sent_alerts) == 1
        assert notifier.sent_alerts[0].notify_reason == "notify_on_motion=true"


class TestClipPipelineConcurrency:
    """Test concurrent clip processing."""

    @pytest.mark.asyncio
    async def test_multiple_clips_processed_concurrently(
        self, base_config: Config, tmp_path: Path, mocks: PipelineMocks
    ) -> None:
        """Multiple clips can be processed concurrently."""
        from datetime import datetime, timedelta

        # Given multiple clips and delayed mocks
        clips = []
        for i in range(3):
            video_path = tmp_path / f"test_clip_{i}.mp4"
            video_path.write_bytes(b"fake video content")
            start_ts = datetime.now()
            clips.append(
                Clip(
                    clip_id=f"test-clip-{i:03d}",
                    camera_name="front_door",
                    local_path=video_path,
                    start_ts=start_ts,
                    end_ts=start_ts + timedelta(seconds=10),
                    duration_s=10.0,
                    source_backend="mock",
                )
            )

        person_result = FilterResult(
            detected_classes=["person"],
            confidence=0.9,
            model="mock",
            sampled_frames=30,
        )
        storage_mock = MockStorage(delay_s=0.05)
        filter_mock = MockFilter(result=person_result, delay_s=0.05)
        vlm_mock = MockVLM(delay_s=0.05)
        mocks.storage = storage_mock
        mocks.filter = filter_mock
        mocks.vlm = vlm_mock

        pipeline = ClipPipeline(
            config=base_config,
            storage=storage_mock,
            repository=make_repository(base_config, mocks),
            filter_plugin=filter_mock,
            vlm_plugin=vlm_mock,
            notifier=mocks.notifier,
            alert_policy=make_alert_policy(base_config),
        )

        # When all clips are submitted
        for clip in clips:
            pipeline.on_new_clip(clip)

        await pipeline.shutdown()

        # Then all clips are processed and notifications sent
        state_store: MockStateStore = mocks.state_store
        for clip in clips:
            state = await state_store.get(clip.clip_id)
            assert state is not None
            assert state.status == "done"

        # Verify all notifications sent
        notifier: MockNotifier = mocks.notifier
        assert len(notifier.sent_alerts) == 3

    @pytest.mark.asyncio
    async def test_vlm_overlaps_upload(self, base_config: Config, sample_clip: Clip) -> None:
        """VLM should start while upload is still in progress."""
        # Given an upload that blocks and a VLM that signals start
        upload_started = asyncio.Event()
        release_upload = asyncio.Event()
        vlm_started = asyncio.Event()

        class BlockingStorage(MockStorage):
            async def put_file(self, local_path: Path, dest_path: str):
                upload_started.set()
                await release_upload.wait()
                return await super().put_file(local_path, dest_path)

        class SignalingVLM(MockVLM):
            async def analyze(
                self, video_path: Path, filter_result: FilterResult, config: VLMConfig
            ) -> AnalysisResult:
                vlm_started.set()
                return await super().analyze(video_path, filter_result, config)

        person_result = FilterResult(
            detected_classes=["person"],
            confidence=0.9,
            model="mock",
            sampled_frames=30,
        )
        pipeline = ClipPipeline(
            config=base_config,
            storage=BlockingStorage(),
            repository=ClipRepository(MockStateStore(), MockEventStore(), retry=base_config.retry),
            filter_plugin=MockFilter(result=person_result),
            vlm_plugin=SignalingVLM(),
            notifier=MockNotifier(),
            alert_policy=make_alert_policy(base_config),
        )

        # When a clip is processed
        pipeline.on_new_clip(sample_clip)

        try:
            # Then VLM starts before upload completes
            await asyncio.wait_for(upload_started.wait(), 1.0)
            await asyncio.wait_for(vlm_started.wait(), 1.0)
        finally:
            release_upload.set()
            await pipeline.shutdown()


class TestClipPipelineShutdown:
    """Test graceful shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_in_flight_clips(
        self, base_config: Config, sample_clip: Clip, mocks: PipelineMocks
    ) -> None:
        """Shutdown waits for in-flight clips to complete."""
        # Given a slow filter to keep work in flight
        person_result = FilterResult(
            detected_classes=["person"],
            confidence=0.9,
            model="mock",
            sampled_frames=30,
        )
        filter_mock = MockFilter(result=person_result, delay_s=0.2)
        mocks.filter = filter_mock

        pipeline = ClipPipeline(
            config=base_config,
            storage=mocks.storage,
            repository=make_repository(base_config, mocks),
            filter_plugin=filter_mock,
            vlm_plugin=mocks.vlm,
            notifier=mocks.notifier,
            alert_policy=make_alert_policy(base_config),
        )

        # When shutdown is called immediately after submitting a clip
        pipeline.on_new_clip(sample_clip)
        await pipeline.shutdown(timeout=5.0)

        # Then the clip finishes processing
        state_store: MockStateStore = mocks.state_store
        state = await state_store.get(sample_clip.clip_id)
        assert state is not None
        assert state.status == "done"


class TestClipPipelineSemaphores:
    """Test concurrency limits."""

    @pytest.mark.asyncio
    async def test_global_concurrency_limit(
        self, base_config: Config, tmp_path: Path, mocks: PipelineMocks
    ) -> None:
        """Verify global clip limit is enforced."""
        from datetime import datetime

        # Given a global clip limit of 1
        base_config.concurrency.max_clips_in_flight = 1

        # Given a filter that blocks until released
        block_event = asyncio.Event()
        start_event = asyncio.Event()

        class BlockingMockFilter(MockFilter):
            async def detect(self, video_path: Path, overrides: FilterOverrides | None = None):
                start_event.set()
                await block_event.wait()
                return await super().detect(video_path, overrides=overrides)

        blocking_filter = BlockingMockFilter(result=mocks.filter.result)
        mocks.filter = blocking_filter

        pipeline = ClipPipeline(
            config=base_config,
            storage=mocks.storage,
            repository=make_repository(base_config, mocks),
            filter_plugin=blocking_filter,
            vlm_plugin=mocks.vlm,
            notifier=mocks.notifier,
            alert_policy=make_alert_policy(base_config),
        )

        # Given two clips
        clip1 = Clip(
            clip_id="c1",
            camera_name="cam",
            local_path=tmp_path / "c1",
            start_ts=datetime.now(),
            end_ts=datetime.now(),
            duration_s=1,
            source_backend="test",
        )
        clip2 = Clip(
            clip_id="c2",
            camera_name="cam",
            local_path=tmp_path / "c2",
            start_ts=datetime.now(),
            end_ts=datetime.now(),
            duration_s=1,
            source_backend="test",
        )

        # When clip 1 starts processing
        pipeline.on_new_clip(clip1)

        await asyncio.wait_for(start_event.wait(), timeout=1.0)
        start_event.clear()

        # When clip 2 is submitted while clip 1 is running
        pipeline.on_new_clip(clip2)

        # Then clip 2 does not start filtering before clip 1 completes
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(start_event.wait(), timeout=0.2)

        # When clip 1 is released
        block_event.set()

        # Then clip 2 begins
        await asyncio.wait_for(start_event.wait(), timeout=1.0)

        await pipeline.shutdown()

    @pytest.mark.asyncio
    async def test_stage_concurrency_limit(
        self, base_config: Config, tmp_path: Path, mocks: PipelineMocks
    ) -> None:
        """Verify stage-specific limit is enforced."""
        from datetime import datetime

        # Given an upload worker limit of 1
        base_config.concurrency.max_clips_in_flight = 5
        base_config.concurrency.upload_workers = 1

        # Given storage that blocks uploads
        block_event = asyncio.Event()
        start_event = asyncio.Event()

        class BlockingMockStorage(MockStorage):
            async def put_file(self, *args, **kwargs):
                start_event.set()
                await block_event.wait()
                return await super().put_file(*args, **kwargs)

        blocking_storage = BlockingMockStorage()

        pipeline = ClipPipeline(
            config=base_config,
            storage=blocking_storage,
            repository=make_repository(base_config, mocks),
            filter_plugin=mocks.filter,
            vlm_plugin=mocks.vlm,
            notifier=mocks.notifier,
            alert_policy=make_alert_policy(base_config),
        )

        # Given two clips
        clip1 = Clip(
            clip_id="c1",
            camera_name="cam",
            local_path=tmp_path / "c1",
            start_ts=datetime.now(),
            end_ts=datetime.now(),
            duration_s=1,
            source_backend="test",
        )
        clip2 = Clip(
            clip_id="c2",
            camera_name="cam",
            local_path=tmp_path / "c2",
            start_ts=datetime.now(),
            end_ts=datetime.now(),
            duration_s=1,
            source_backend="test",
        )

        # When clip 1 starts uploading
        pipeline.on_new_clip(clip1)
        await asyncio.wait_for(start_event.wait(), timeout=1.0)
        start_event.clear()

        # When clip 2 is submitted
        pipeline.on_new_clip(clip2)

        # Then clip 2 upload is blocked
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(start_event.wait(), timeout=0.2)

        # When clip 1 upload is released
        block_event.set()
        await asyncio.wait_for(start_event.wait(), timeout=1.0)

        await pipeline.shutdown()
