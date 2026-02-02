"""Tests for OpenAIVLM analyzer plugin."""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import cv2
import numpy as np
import pytest
from PIL import Image

from homesec.models.enums import RiskLevel
from homesec.models.filter import FilterResult
from homesec.models.vlm import VLMConfig, VLMPreprocessConfig
from homesec.plugins.analyzers.openai import OpenAIConfig, OpenAIVLM


def _make_config(**overrides: Any) -> VLMConfig:
    """Create a VLMConfig with defaults for testing."""
    defaults: dict[str, Any] = {
        "backend": "openai",
        "trigger_classes": ["person"],
        "config": OpenAIConfig(
            api_key_env="OPENAI_API_KEY",
            model="gpt-4o",
            request_timeout=30.0,
        ),
    }
    defaults.update(overrides)
    return VLMConfig(**defaults)


def _make_vlm(config: VLMConfig | None = None) -> OpenAIVLM:
    """Create an OpenAIVLM."""
    if config is None:
        config = _make_config()
    llm_config = config.config
    if not isinstance(llm_config, OpenAIConfig):
        raise TypeError("Expected OpenAIConfig")

    return OpenAIVLM(llm_config)


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
    if not out.isOpened():
        pytest.skip("OpenCV VideoWriter (mp4v) not available")

    for i in range(frames):
        # Create frames with different content (gradient based on frame number)
        color = (i * 50) % 256
        frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        out.write(frame)

    out.release()
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("OpenCV failed to write test video")


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


def _make_async_cm(response: AsyncMock) -> AsyncMock:
    async_cm = AsyncMock()
    async_cm.__aenter__ = AsyncMock(return_value=response)
    async_cm.__aexit__ = AsyncMock(return_value=None)
    return async_cm


def _patch_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    async_cm: AsyncMock,
    capture: dict[str, Any] | None = None,
) -> MagicMock:
    session = MagicMock()

    def capture_post(url: str, json: dict[str, Any], headers: dict[str, str]) -> AsyncMock:
        if capture is not None:
            capture["url"] = url
            capture["json"] = json
            capture["headers"] = headers
        return async_cm

    session.post = capture_post if capture is not None else MagicMock(return_value=async_cm)

    async def _close() -> None:
        session.closed = True

    session.close = AsyncMock(side_effect=_close)
    session.closed = False

    monkeypatch.setattr(
        "homesec.plugins.analyzers.openai.aiohttp.ClientSession",
        lambda **_kw: session,
    )
    return session


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

        analyzer = _make_vlm()
        captured_request: dict[str, Any] = {}

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=_analysis_response())

        async_cm = _make_async_cm(mock_response)
        _patch_session(monkeypatch, async_cm=async_cm, capture=captured_request)

        # When: Analyzing the video
        result = await analyzer.analyze(video_path, _make_filter_result(), _make_config())

        # Then: API was called with correct structure
        assert captured_request["url"] == "https://api.openai.com/v1/chat/completions"
        assert captured_request["headers"]["Authorization"] == "Bearer test-api-key"

        payload = captured_request["json"]
        assert payload["model"] == "gpt-4o"
        assert "response_format" in payload

        # Then: Frames were extracted and encoded
        messages = payload["messages"]
        assert len(messages) == 2  # system + user
        user_content = messages[1]["content"]

        image_entries = [c for c in user_content if c.get("type") == "image_url"]
        assert len(image_entries) > 0

        for entry in image_entries:
            url = entry["image_url"]["url"]
            assert url.startswith("data:image/jpeg;base64,")
            b64_data = url.split(",")[1]
            decoded = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(decoded))
            assert img.format == "JPEG"

        # Then: Timestamps are formatted consistently
        frame_texts = [
            c["text"]
            for c in user_content
            if c.get("type") == "text" and c["text"].startswith("Frame at ")
        ]
        assert frame_texts
        for text in frame_texts:
            timestamp = text.split("Frame at ")[1].split(" (", 1)[0]
            assert re.match(r"^\d{2}:\d{2}:\d{2}\.\d{2}$", timestamp)

        # Then: Result is parsed correctly
        assert result.risk_level == RiskLevel.LOW
        assert result.activity_type == "passerby"
        assert result.prompt_tokens == 100

        await analyzer.shutdown()

    @pytest.mark.asyncio
    async def test_analyze_respects_preprocessing_limits(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Analyze limits frame count and image size per preprocessing config."""
        # Given: A video larger than preprocessing limits
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        preprocess = VLMPreprocessConfig(max_frames=3, max_size=64, quality=85)
        config = _make_config(preprocessing=preprocess)
        video_path = tmp_path / "clip.mp4"
        _create_test_video(video_path, frames=12, size=(128, 128))

        analyzer = _make_vlm(config)
        captured_request: dict[str, Any] = {}

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=_analysis_response())

        async_cm = _make_async_cm(mock_response)
        _patch_session(monkeypatch, async_cm=async_cm, capture=captured_request)

        # When: Analyzing the video
        await analyzer.analyze(video_path, _make_filter_result(), config)

        # Then: Frame count is capped
        user_content = captured_request["json"]["messages"][1]["content"]
        image_entries = [c for c in user_content if c.get("type") == "image_url"]
        assert len(image_entries) == 3

        # Then: Images are resized to max_size
        for entry in image_entries:
            b64_data = entry["image_url"]["url"].split(",", 1)[1]
            decoded = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(decoded))
            assert max(img.size) <= 64

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

        analyzer = _make_vlm()

        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value='{"error": "Invalid API key"}')

        async_cm = _make_async_cm(mock_response)
        _patch_session(monkeypatch, async_cm=async_cm)

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

        analyzer = _make_vlm()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"choices": [{"message": {"content": "not valid json {"}}], "usage": {}}
        )

        async_cm = _make_async_cm(mock_response)
        _patch_session(monkeypatch, async_cm=async_cm)

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

        analyzer = _make_vlm()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"choices": [{"message": {"content": '{"wrong": "schema"}'}}], "usage": {}}
        )

        async_cm = _make_async_cm(mock_response)
        _patch_session(monkeypatch, async_cm=async_cm)

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

        analyzer = _make_vlm()

        # When/Then: Analyze raises ValueError
        with pytest.raises(ValueError, match="No frames extracted"):
            await analyzer.analyze(video_path, _make_filter_result(), _make_config())

        await analyzer.shutdown()


class TestOpenAIVLMConfiguration:
    """Tests for configuration validation."""

    def test_raises_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises ValueError when API key env var is not set."""
        # Given: No API key in environment
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # When/Then: Creating analyzer raises ValueError
        with pytest.raises(ValueError, match="API key not found"):
            _make_vlm()


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

        analyzer = _make_vlm()
        await analyzer.shutdown()

        # When/Then: Analyze raises RuntimeError
        with pytest.raises(RuntimeError, match="shut down"):
            await analyzer.analyze(video_path, _make_filter_result(), _make_config())

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Shutdown can be called multiple times safely."""
        # Given: An analyzer with an active session
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        video_path = tmp_path / "clip.mp4"
        _create_test_video(video_path, frames=3)
        analyzer = _make_vlm()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=_analysis_response())

        async_cm = _make_async_cm(mock_response)
        session = _patch_session(monkeypatch, async_cm=async_cm)

        await analyzer.analyze(video_path, _make_filter_result(), _make_config())

        # When: Calling shutdown multiple times
        await analyzer.shutdown()
        await analyzer.shutdown()
        await analyzer.shutdown()

        # Then: Session is closed (shutdown state observable)
        assert session.closed is True
