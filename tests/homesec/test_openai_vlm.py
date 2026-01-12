"""Tests for OpenAIVLM analyzer plugin."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from homesec.models.filter import FilterResult
from homesec.models.vlm import OpenAILLMConfig, VLMConfig
from homesec.plugins.analyzers.openai import OpenAIVLM


def _make_config() -> VLMConfig:
    return VLMConfig(
        backend="openai",
        trigger_classes=["person"],
        llm=OpenAILLMConfig(
            api_key_env="OPENAI_API_KEY",
            model="gpt-4o",
            request_timeout=1.0,
        ),
    )


def _analysis_payload() -> dict[str, object]:
    return {
        "sequence_description": "Person walks by.",
        "max_risk_level": "low",
        "primary_activity": "passerby",
        "observations": ["person enters frame", "person leaves frame"],
        "entities_timeline": [],
        "requires_review": False,
        "frame_count": 2,
        "video_start_time": "00:00:00.00",
        "video_end_time": "00:00:01.00",
    }


@pytest.mark.asyncio
async def test_analyze_builds_payload_and_parses_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Analyze should build payload and parse structured output."""
    # Given an OpenAIVLM with stubbed frame extraction and API call
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = _make_config()
    analyzer = OpenAIVLM(config)
    frames = [("ZmFrZQ==", "00:00:00.00"), ("ZmFrZQ==", "00:00:01.00")]
    captured_payload: dict[str, object] = {}

    async def _fake_extract_frames_async(
        *_args: object, **_kwargs: object
    ) -> list[tuple[str, str]]:
        return frames

    async def _fake_call_api(
        payload: dict[str, object], _headers: dict[str, str]
    ) -> dict[str, object]:
        captured_payload.update(payload)
        return {
            "choices": [{"message": {"content": json.dumps(_analysis_payload())}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(analyzer, "_extract_frames_async", _fake_extract_frames_async)
    monkeypatch.setattr(analyzer, "_call_api", _fake_call_api)

    filter_result = FilterResult(
        detected_classes=["person"],
        confidence=0.9,
        model="mock",
        sampled_frames=2,
    )

    # When analyze is called
    result = await analyzer.analyze(tmp_path / "clip.mp4", filter_result, config)

    # Then a structured AnalysisResult is returned
    assert result.risk_level == "low"
    assert result.activity_type == "passerby"
    assert result.summary == "Person walks by."
    assert result.prompt_tokens == 1
    assert result.completion_tokens == 1

    # Then the payload includes response_format and token param
    assert captured_payload["model"] == "gpt-4o"
    assert "response_format" in captured_payload
    assert analyzer.token_param in captured_payload


@pytest.mark.asyncio
async def test_analyze_raises_on_empty_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Analyze should raise when no frames are extracted."""
    # Given an OpenAIVLM that extracts no frames
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = _make_config()
    analyzer = OpenAIVLM(config)

    async def _fake_extract_frames_async(
        *_args: object, **_kwargs: object
    ) -> list[tuple[str, str]]:
        return []

    monkeypatch.setattr(analyzer, "_extract_frames_async", _fake_extract_frames_async)

    # When analyze is called
    with pytest.raises(ValueError):
        await analyzer.analyze(
            tmp_path / "clip.mp4",
            FilterResult(
                detected_classes=["person"],
                confidence=0.9,
                model="mock",
                sampled_frames=2,
            ),
            config,
        )


@pytest.mark.asyncio
async def test_analyze_raises_on_invalid_api_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Analyze should raise on malformed API responses."""
    # Given an OpenAIVLM with malformed API response
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = _make_config()
    analyzer = OpenAIVLM(config)

    async def _fake_extract_frames_async(
        *_args: object, **_kwargs: object
    ) -> list[tuple[str, str]]:
        return [("ZmFrZQ==", "00:00:00.00")]

    async def _fake_call_api(
        _payload: dict[str, object], _headers: dict[str, str]
    ) -> dict[str, object]:
        return {"choices": []}

    monkeypatch.setattr(analyzer, "_extract_frames_async", _fake_extract_frames_async)
    monkeypatch.setattr(analyzer, "_call_api", _fake_call_api)

    # When analyze is called
    with pytest.raises(TypeError):
        await analyzer.analyze(
            tmp_path / "clip.mp4",
            FilterResult(
                detected_classes=["person"],
                confidence=0.9,
                model="mock",
                sampled_frames=1,
            ),
            config,
        )


class TestOpenAIVLMImageProcessing:
    """Tests for image processing methods."""

    def test_format_timestamp_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Formats 0ms timestamp correctly."""
        # Given: An OpenAIVLM instance
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        analyzer = OpenAIVLM(_make_config())

        # When: Formatting 0ms timestamp
        result = analyzer._format_timestamp(0.0)

        # Then: Formatted as 00:00:00.00
        assert result == "00:00:00.00"

    def test_format_timestamp_seconds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Formats timestamp with seconds correctly."""
        # Given: An OpenAIVLM instance
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        analyzer = OpenAIVLM(_make_config())

        # When: Formatting 5500ms (5.5 seconds)
        result = analyzer._format_timestamp(5500.0)

        # Then: Formatted with seconds and centiseconds
        assert result == "00:00:05.50"

    def test_format_timestamp_minutes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Formats timestamp with minutes correctly."""
        # Given: An OpenAIVLM instance
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        analyzer = OpenAIVLM(_make_config())

        # When: Formatting 90000ms (1 minute 30 seconds)
        result = analyzer._format_timestamp(90000.0)

        # Then: Formatted with minutes
        assert result == "00:01:30.00"

    def test_format_timestamp_hours(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Formats timestamp with hours correctly."""
        # Given: An OpenAIVLM instance
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        analyzer = OpenAIVLM(_make_config())

        # When: Formatting 3723000ms (1 hour, 2 minutes, 3 seconds)
        result = analyzer._format_timestamp(3723000.0)

        # Then: Formatted with hours
        assert result == "01:02:03.00"

    def test_format_timestamp_negative_clamps_to_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Negative timestamps are clamped to zero."""
        # Given: An OpenAIVLM instance
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        analyzer = OpenAIVLM(_make_config())

        # When: Formatting negative timestamp
        result = analyzer._format_timestamp(-5000.0)

        # Then: Clamped to zero
        assert result == "00:00:00.00"

    def test_resize_image_landscape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resizes landscape image preserving aspect ratio."""
        # Given: An OpenAIVLM instance and a wide image
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        analyzer = OpenAIVLM(_make_config())

        from PIL import Image

        img = Image.new("RGB", (2000, 1000))  # 2:1 aspect ratio

        # When: Resizing to max 512
        result = analyzer._resize_image(img, 512)

        # Then: Width is max_size, height preserves ratio
        assert result.size[0] == 512
        assert result.size[1] == 256  # 512 * 0.5

    def test_resize_image_portrait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resizes portrait image preserving aspect ratio."""
        # Given: An OpenAIVLM instance and a tall image
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        analyzer = OpenAIVLM(_make_config())

        from PIL import Image

        img = Image.new("RGB", (1000, 2000))  # 1:2 aspect ratio

        # When: Resizing to max 512
        result = analyzer._resize_image(img, 512)

        # Then: Height is max_size, width preserves ratio
        assert result.size[0] == 256  # 512 * 0.5
        assert result.size[1] == 512

    def test_resize_image_already_small(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Doesn't resize image already within limits."""
        # Given: An OpenAIVLM instance and a small image
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        analyzer = OpenAIVLM(_make_config())

        from PIL import Image

        img = Image.new("RGB", (256, 256))

        # When: Resizing to max 512
        result = analyzer._resize_image(img, 512)

        # Then: Image is unchanged
        assert result.size == (256, 256)


class TestOpenAIVLMShutdown:
    """Tests for shutdown behavior."""

    @pytest.mark.asyncio
    async def test_analyze_fails_after_shutdown(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Analyze raises RuntimeError after shutdown."""
        # Given: An OpenAIVLM that has been shut down
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        analyzer = OpenAIVLM(_make_config())
        await analyzer.shutdown()

        # When/Then: Analyze raises RuntimeError
        with pytest.raises(RuntimeError, match="shut down"):
            await analyzer.analyze(
                tmp_path / "clip.mp4",
                FilterResult(
                    detected_classes=["person"],
                    confidence=0.9,
                    model="mock",
                    sampled_frames=1,
                ),
                _make_config(),
            )

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Shutdown can be called multiple times safely."""
        # Given: An OpenAIVLM instance
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
        # Given: An OpenAIVLM with an active session
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        analyzer = OpenAIVLM(_make_config())
        session = await analyzer._ensure_session()
        assert not session.closed

        # When: Shutting down
        await analyzer.shutdown()

        # Then: Session is closed
        assert session.closed


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

        # Create a valid config first, then override llm with wrong type
        config = _make_config()
        # Use model_copy to create a new instance with wrong llm type
        wrong_config = config.model_copy(update={"llm": WrongLLMConfig()})

        # When/Then: Creating analyzer raises ValueError
        with pytest.raises(ValueError, match="requires llm=OpenAILLMConfig"):
            OpenAIVLM(wrong_config)


class TestOpenAIVLMAPIErrors:
    """Tests for API error handling."""

    @pytest.mark.asyncio
    async def test_api_non_200_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """API errors raise RuntimeError with status and message."""
        # Given: An OpenAIVLM with mocked API returning 401
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        config = _make_config()
        analyzer = OpenAIVLM(config)

        async def _fake_extract_frames_async(
            *_args: object, **_kwargs: object
        ) -> list[tuple[str, str]]:
            return [("ZmFrZQ==", "00:00:00.00")]

        async def _fake_call_api(
            _payload: dict[str, object], _headers: dict[str, str]
        ) -> dict[str, object]:
            raise RuntimeError("OpenAI API error 401: Unauthorized")

        monkeypatch.setattr(analyzer, "_extract_frames_async", _fake_extract_frames_async)
        monkeypatch.setattr(analyzer, "_call_api", _fake_call_api)

        # When/Then: Analyze raises RuntimeError
        with pytest.raises(RuntimeError, match="401"):
            await analyzer.analyze(
                tmp_path / "clip.mp4",
                FilterResult(
                    detected_classes=["person"],
                    confidence=0.9,
                    model="mock",
                    sampled_frames=1,
                ),
                config,
            )

        await analyzer.shutdown()

    @pytest.mark.asyncio
    async def test_malformed_json_content_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Malformed JSON in response content raises JSONDecodeError."""
        # Given: An OpenAIVLM with API returning invalid JSON content
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        config = _make_config()
        analyzer = OpenAIVLM(config)

        async def _fake_extract_frames_async(
            *_args: object, **_kwargs: object
        ) -> list[tuple[str, str]]:
            return [("ZmFrZQ==", "00:00:00.00")]

        async def _fake_call_api(
            _payload: dict[str, object], _headers: dict[str, str]
        ) -> dict[str, object]:
            return {
                "choices": [{"message": {"content": "not valid json {"}}],
                "usage": {},
            }

        monkeypatch.setattr(analyzer, "_extract_frames_async", _fake_extract_frames_async)
        monkeypatch.setattr(analyzer, "_call_api", _fake_call_api)

        # When/Then: Analyze raises JSONDecodeError
        with pytest.raises(json.JSONDecodeError):
            await analyzer.analyze(
                tmp_path / "clip.mp4",
                FilterResult(
                    detected_classes=["person"],
                    confidence=0.9,
                    model="mock",
                    sampled_frames=1,
                ),
                config,
            )

        await analyzer.shutdown()

    @pytest.mark.asyncio
    async def test_schema_mismatch_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Response not matching SequenceAnalysis schema raises ValueError."""
        # Given: An OpenAIVLM with API returning wrong schema
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        config = _make_config()
        analyzer = OpenAIVLM(config)

        async def _fake_extract_frames_async(
            *_args: object, **_kwargs: object
        ) -> list[tuple[str, str]]:
            return [("ZmFrZQ==", "00:00:00.00")]

        async def _fake_call_api(
            _payload: dict[str, object], _headers: dict[str, str]
        ) -> dict[str, object]:
            # Valid JSON but wrong schema (missing required fields)
            return {
                "choices": [{"message": {"content": '{"wrong": "schema"}'}}],
                "usage": {},
            }

        monkeypatch.setattr(analyzer, "_extract_frames_async", _fake_extract_frames_async)
        monkeypatch.setattr(analyzer, "_call_api", _fake_call_api)

        # When/Then: Analyze raises ValueError
        with pytest.raises(ValueError, match="does not match SequenceAnalysis"):
            await analyzer.analyze(
                tmp_path / "clip.mp4",
                FilterResult(
                    detected_classes=["person"],
                    confidence=0.9,
                    model="mock",
                    sampled_frames=1,
                ),
                config,
            )

        await analyzer.shutdown()
