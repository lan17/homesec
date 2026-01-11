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

    async def _fake_extract_frames_async(*_args: object, **_kwargs: object) -> list[tuple[str, str]]:
        return frames

    async def _fake_call_api(payload: dict[str, object], _headers: dict[str, str]) -> dict[str, object]:
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

    async def _fake_extract_frames_async(*_args: object, **_kwargs: object) -> list[tuple[str, str]]:
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

    async def _fake_extract_frames_async(*_args: object, **_kwargs: object) -> list[tuple[str, str]]:
        return [("ZmFrZQ==", "00:00:00.00")]

    async def _fake_call_api(_payload: dict[str, object], _headers: dict[str, str]) -> dict[str, object]:
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
