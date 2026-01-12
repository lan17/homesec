"""Tests for OpenAIVLM analyzer plugin."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import cv2
import numpy as np
import pytest
from PIL import Image

from homesec.models.filter import FilterResult
from homesec.models.vlm import OpenAILLMConfig, VLMConfig
from homesec.plugins.analyzers.openai import OpenAIVLM


def _make_config(**overrides: Any) -> VLMConfig:
    """Create a VLMConfig with defaults for testing."""
    defaults: dict[str, Any] = {
        "backend": "openai",
        "trigger_classes": ["person"],
        "llm": OpenAILLMConfig(
            api_key_env="OPENAI_API_KEY",
            model="gpt-4o",
            request_timeout=30.0,
        ),
    }
    defaults.update(overrides)
    return VLMConfig(**defaults)


def _make_filter_result(**overrides: Any) -> FilterResult:
    """Create a FilterResult with defaults for testing."""
    defaults: dict[str, Any] = {
        "detected_classes": ["person"],
        "confidence": 0.9,
        "model": "yolo",
        "sampled_frames": 10,
    }
    defaults.update(overrides)
    return FilterResult(**defaults)


def _create_test_video(path: Path, frames: int = 5, size: tuple[int, int] = (64, 64)) -> None:
    """Create a minimal test video file.

    Creates frames with different colors so we can verify extraction works.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, size)

    for i in range(frames):
        # Create frames with different content (gradient based on frame number)
        color = (i * 50) % 256
        frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        out.write(frame)

    out.release()


def _analysis_response(
    risk_level: str = "low",
    activity: str = "passerby",
    description: str = "Person walks by.",
) -> dict[str, Any]:
    """Create a valid OpenAI API response with SequenceAnalysis."""
    analysis = {
        "sequence_description": description,
        "max_risk_level": risk_level,
        "primary_activity": activity,
        "observations": ["person enters frame", "person leaves frame"],
        "entities_timeline": [],
        "requires_review": False,
        "frame_count": 2,
        "video_start_time": "00:00:00.00",
        "video_end_time": "00:00:01.00",
    }
    return {
        "choices": [{"message": {"content": json.dumps(analysis)}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


def _mock_http_response(status: int, json_data: dict[str, Any]) -> AsyncMock:
    """Create a mock aiohttp response."""
    response = AsyncMock()
    response.status = status
    response.json = AsyncMock(return_value=json_data)
    response.text = AsyncMock(return_value=json.dumps(json_data))
    return response


class TestOpenAIVLMAnalyze:
    """Tests for the analyze method - mocking at HTTP boundary."""

    @pytest.mark.asyncio
    async def test_analyze_extracts_frames_and_calls_api(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Analyze extracts frames from real video and sends correct API request."""
        # Given: A real test video and an analyzer with mocked HTTP session
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        video_path = tmp_path / "clip.mp4"
        _create_test_video(video_path, frames=5)

        analyzer = OpenAIVLM(_make_config())

        captured_request: dict[str, Any] = {}

        # Create async context manager for the response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=_analysis_response())

        async_cm = AsyncMock()
        async_cm.__aenter__ = AsyncMock(return_value=mock_response)
        async_cm.__aexit__ = AsyncMock(return_value=None)

        def capture_post(url: str, json: dict[str, Any], headers: dict[str, str]) -> Any:
            captured_request["url"] = url
            captured_request["json"] = json
            captured_request["headers"] = headers
            return async_cm

        mock_session = MagicMock()
        mock_session.post = capture_post
        mock_session.close = AsyncMock()
        mock_session.closed = False
        analyzer._session = mock_session

        # When: Analyzing the video
        result = await analyzer.analyze(video_path, _make_filter_result(), _make_config())

        # Then: API was called with correct structure
        assert captured_request["url"] == "https://api.openai.com/v1/chat/completions"
        assert captured_request["headers"]["Authorization"] == "Bearer test-api-key"

        payload = captured_request["json"]
        assert payload["model"] == "gpt-4o"
        assert "response_format" in payload

        # Verify frames were extracted and encoded
        messages = payload["messages"]
        assert len(messages) == 2  # system + user
        user_content = messages[1]["content"]

        # Count image_url entries (should have frames)
        image_entries = [c for c in user_content if c.get("type") == "image_url"]
        assert len(image_entries) > 0

        # Verify images are valid base64 JPEGs
        for entry in image_entries:
            url = entry["image_url"]["url"]
            assert url.startswith("data:image/jpeg;base64,")
            b64_data = url.split(",")[1]
            decoded = base64.b64decode(b64_data)
            # Verify it's a valid JPEG
            img = Image.open(io.BytesIO(decoded))
            assert img.format == "JPEG"

        # Verify result is correct
        assert result.risk_level == "low"
        assert result.activity_type == "passerby"
        assert result.prompt_tokens == 100

        await analyzer.shutdown()

    @pytest.mark.asyncio
    async def test_analyze_handles_api_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Analyze raises RuntimeError with status code on API failure."""
        # Given: A video and analyzer with API returning 401
        monkeypatch.setenv("OPENAI_API_KEY", "bad-key")
        video_path = tmp_path / "clip.mp4"
        _create_test_video(video_path, frames=3)

        analyzer = OpenAIVLM(_make_config())

        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value='{"error": "Invalid API key"}')

        async_cm = AsyncMock()
        async_cm.__aenter__ = AsyncMock(return_value=mock_response)
        async_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=async_cm)
        mock_session.close = AsyncMock()
        mock_session.closed = False
        analyzer._session = mock_session

        # When/Then: Analyze raises with status code
        with pytest.raises(RuntimeError) as exc_info:
            await analyzer.analyze(video_path, _make_filter_result(), _make_config())

        assert "401" in str(exc_info.value)
        await analyzer.shutdown()

    @pytest.mark.asyncio
    async def test_analyze_handles_malformed_json_response(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Analyze raises JSONDecodeError when API returns invalid JSON content."""
        # Given: A video and analyzer with API returning invalid JSON in content
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        video_path = tmp_path / "clip.mp4"
        _create_test_video(video_path, frames=3)

        analyzer = OpenAIVLM(_make_config())

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"choices": [{"message": {"content": "not valid json {"}}], "usage": {}}
        )

        async_cm = AsyncMock()
        async_cm.__aenter__ = AsyncMock(return_value=mock_response)
        async_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=async_cm)
        mock_session.close = AsyncMock()
        mock_session.closed = False
        analyzer._session = mock_session

        # When/Then: Analyze raises JSONDecodeError
        with pytest.raises(json.JSONDecodeError):
            await analyzer.analyze(video_path, _make_filter_result(), _make_config())

        await analyzer.shutdown()

    @pytest.mark.asyncio
    async def test_analyze_handles_schema_mismatch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Analyze raises ValueError when response doesn't match schema."""
        # Given: A video and analyzer with API returning wrong schema
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        video_path = tmp_path / "clip.mp4"
        _create_test_video(video_path, frames=3)

        analyzer = OpenAIVLM(_make_config())

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"choices": [{"message": {"content": '{"wrong": "schema"}'}}], "usage": {}}
        )

        async_cm = AsyncMock()
        async_cm.__aenter__ = AsyncMock(return_value=mock_response)
        async_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=async_cm)
        mock_session.close = AsyncMock()
        mock_session.closed = False
        analyzer._session = mock_session

        # When/Then: Analyze raises ValueError
        with pytest.raises(ValueError, match="does not match SequenceAnalysis"):
            await analyzer.analyze(video_path, _make_filter_result(), _make_config())

        await analyzer.shutdown()

    @pytest.mark.asyncio
    async def test_analyze_raises_on_empty_video(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Analyze raises ValueError when no frames can be extracted."""
        # Given: An empty/corrupted video file
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        video_path = tmp_path / "empty.mp4"
        video_path.write_bytes(b"not a real video")

        analyzer = OpenAIVLM(_make_config())

        # When/Then: Analyze raises ValueError
        with pytest.raises(ValueError, match="No frames extracted"):
            await analyzer.analyze(video_path, _make_filter_result(), _make_config())

        await analyzer.shutdown()


class TestOpenAIVLMFrameExtraction:
    """Tests for frame extraction - using real video files."""

    def test_extract_frames_from_real_video(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Extracts frames from a real video file."""
        # Given: A test video with 10 frames
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        video_path = tmp_path / "clip.mp4"
        _create_test_video(video_path, frames=10, size=(100, 100))

        analyzer = OpenAIVLM(_make_config())

        # When: Extracting frames
        frames = analyzer._extract_frames(video_path, max_frames=5, max_size=512, quality=85)

        # Then: Correct number of frames extracted
        assert len(frames) == 5

        # Verify each frame is valid base64 JPEG with timestamp
        for b64_data, timestamp in frames:
            decoded = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(decoded))
            assert img.format == "JPEG"
            # Timestamp format: HH:MM:SS.cc
            assert len(timestamp.split(":")) == 3

    def test_extract_frames_samples_evenly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Samples frames evenly across the video."""
        # Given: A video with 100 frames
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        video_path = tmp_path / "long.mp4"
        _create_test_video(video_path, frames=100, size=(64, 64))

        analyzer = OpenAIVLM(_make_config())

        # When: Extracting 10 frames
        frames = analyzer._extract_frames(video_path, max_frames=10, max_size=512, quality=85)

        # Then: 10 frames extracted, spread across video
        assert len(frames) == 10

    def test_extract_frames_respects_max_size(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Resizes large frames to max_size."""
        # Given: A video with large frames
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        video_path = tmp_path / "large.mp4"
        _create_test_video(video_path, frames=3, size=(1920, 1080))

        analyzer = OpenAIVLM(_make_config())

        # When: Extracting with max_size=512
        frames = analyzer._extract_frames(video_path, max_frames=3, max_size=512, quality=85)

        # Then: Frames are resized
        for b64_data, _ in frames:
            decoded = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(decoded))
            assert max(img.size) <= 512


class TestOpenAIVLMImageProcessing:
    """Tests for image processing helper methods."""

    def test_resize_image_landscape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resizes landscape image preserving aspect ratio."""
        # Given: An analyzer and a wide image
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        analyzer = OpenAIVLM(_make_config())
        img = Image.new("RGB", (2000, 1000))  # 2:1 aspect ratio

        # When: Resizing to max 512
        result = analyzer._resize_image(img, 512)

        # Then: Width is max_size, height preserves ratio
        assert result.size[0] == 512
        assert result.size[1] == 256  # 512 * 0.5

    def test_resize_image_portrait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resizes portrait image preserving aspect ratio."""
        # Given: An analyzer and a tall image
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        analyzer = OpenAIVLM(_make_config())
        img = Image.new("RGB", (1000, 2000))  # 1:2 aspect ratio

        # When: Resizing to max 512
        result = analyzer._resize_image(img, 512)

        # Then: Height is max_size, width preserves ratio
        assert result.size[0] == 256  # 512 * 0.5
        assert result.size[1] == 512

    def test_resize_image_already_small(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Doesn't resize image already within limits."""
        # Given: An analyzer and a small image
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        analyzer = OpenAIVLM(_make_config())
        img = Image.new("RGB", (256, 256))

        # When: Resizing to max 512
        result = analyzer._resize_image(img, 512)

        # Then: Image is unchanged
        assert result.size == (256, 256)

    def test_format_timestamp_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Formats 0ms timestamp correctly."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        analyzer = OpenAIVLM(_make_config())

        assert analyzer._format_timestamp(0.0) == "00:00:00.00"

    def test_format_timestamp_seconds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Formats timestamp with seconds correctly."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        analyzer = OpenAIVLM(_make_config())

        assert analyzer._format_timestamp(5500.0) == "00:00:05.50"

    def test_format_timestamp_minutes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Formats timestamp with minutes correctly."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        analyzer = OpenAIVLM(_make_config())

        assert analyzer._format_timestamp(90000.0) == "00:01:30.00"

    def test_format_timestamp_hours(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Formats timestamp with hours correctly."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        analyzer = OpenAIVLM(_make_config())

        assert analyzer._format_timestamp(3723000.0) == "01:02:03.00"


class TestOpenAIVLMConfiguration:
    """Tests for configuration validation."""

    def test_raises_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises ValueError when API key env var is not set."""
        # Given: No API key in environment
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # When/Then: Creating analyzer raises ValueError
        with pytest.raises(ValueError, match="API key not found"):
            OpenAIVLM(_make_config())

    def test_raises_with_wrong_llm_config_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises ValueError when llm is not OpenAILLMConfig."""
        # Given: A config with wrong llm type
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        from pydantic import BaseModel

        class WrongLLMConfig(BaseModel):
            api_key_env: str = "OPENAI_API_KEY"

        config = _make_config()
        wrong_config = config.model_copy(update={"llm": WrongLLMConfig()})

        # When/Then: Creating analyzer raises ValueError
        with pytest.raises(ValueError, match="requires llm=OpenAILLMConfig"):
            OpenAIVLM(wrong_config)


class TestOpenAIVLMShutdown:
    """Tests for shutdown behavior."""

    @pytest.mark.asyncio
    async def test_analyze_fails_after_shutdown(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Analyze raises RuntimeError after shutdown."""
        # Given: An analyzer that has been shut down
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        video_path = tmp_path / "clip.mp4"
        _create_test_video(video_path, frames=3)

        analyzer = OpenAIVLM(_make_config())
        await analyzer.shutdown()

        # When/Then: Analyze raises RuntimeError
        with pytest.raises(RuntimeError, match="shut down"):
            await analyzer.analyze(video_path, _make_filter_result(), _make_config())

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Shutdown can be called multiple times safely."""
        # Given: An analyzer
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        analyzer = OpenAIVLM(_make_config())

        # When: Calling shutdown multiple times
        await analyzer.shutdown()
        await analyzer.shutdown()
        await analyzer.shutdown()

        # Then: No exception is raised

    @pytest.mark.asyncio
    async def test_shutdown_closes_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Shutdown closes the HTTP session."""
        # Given: An analyzer with an active session
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        analyzer = OpenAIVLM(_make_config())
        session = await analyzer._ensure_session()
        assert not session.closed

        # When: Shutting down
        await analyzer.shutdown()

        # Then: Session is closed
        assert session.closed
