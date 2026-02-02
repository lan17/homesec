"""OpenAI-compatible VLM analyzer plugin."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

aiohttp: Any
cv2: Any
Image: Any

try:
    import aiohttp as _aiohttp
    import cv2 as _cv2
    from PIL import Image as _Image
except Exception:
    aiohttp = None
    cv2 = None
    Image = None
else:
    aiohttp = _aiohttp
    cv2 = _cv2
    Image = _Image

from pydantic import BaseModel

from homesec.interfaces import VLMAnalyzer
from homesec.models.filter import FilterResult
from homesec.models.vlm import AnalysisResult, SequenceAnalysis, VLMConfig, VLMPreprocessConfig
from homesec.plugins.registry import PluginType, plugin

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """Analyze this residential security camera footage frame-by-frame and identify key security events.

CRITICAL INSTRUCTIONS:
1. Carefully examine EACH frame to identify when entities appear and disappear
2. Detect ALL entities: every person, vehicle, animal, package - even if they're far away or in the background
3. Record the FIRST timestamp where you see each person/vehicle
4. Record the LAST timestamp where you see each person/vehicle
5. Use ONLY the exact timestamps shown in frame labels - never guess or extrapolate

Focus on KEY EVENTS ONLY:
- Person approaching/departing property
- Doorbell ring, door interaction, window checking
- Suspicious behaviors: loitering, concealing face, multiple passes
- Package delivery or theft
- Vehicles stopping, driving past, or unusual patterns

Keep observations list concise (short bullet points of security-relevant actions)."""


class OpenAIConfig(BaseModel):
    """OpenAI-compatible LLM configuration."""

    model_config = {"extra": "forbid"}
    api_key_env: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    token_param: Literal["max_tokens", "max_completion_tokens"] = "max_completion_tokens"
    max_completion_tokens: int = 10_000
    max_tokens: int | None = None
    temperature: float | None = 0.0
    request_timeout: float = 60.0


def _ensure_openai_dependencies() -> None:
    """Fail fast with a clear error if OpenAI VLM dependencies are missing."""
    if aiohttp is None or cv2 is None or Image is None:
        raise RuntimeError(
            "Missing dependency for OpenAI VLM. "
            "Install with: uv pip install aiohttp opencv-python pillow"
        )


def _create_json_schema_format(
    schema_model: type[BaseModel], schema_name: str
) -> dict[str, object]:
    """Create OpenAI JSON schema format configuration."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema_model.model_json_schema(),
            "strict": True,
        },
    }


@plugin(plugin_type=PluginType.ANALYZER, name="openai")
class OpenAIVLM(VLMAnalyzer):
    """OpenAI-compatible VLM analyzer plugin.

    Uses aiohttp for async HTTP calls to OpenAI API.
    Supports structured output with Pydantic schemas.
    """

    config_cls = OpenAIConfig

    @classmethod
    def create(cls, config: OpenAIConfig) -> VLMAnalyzer:
        return cls(config)

    def __init__(self, llm_config: OpenAIConfig) -> None:
        """Initialize OpenAI VLM with validated LLM config.

        Args:
            llm_config: OpenAI-specific configuration (API key, model, etc.)
        """
        _ensure_openai_dependencies()
        self._config = llm_config

        # Get API key from env
        self._api_key_env = llm_config.api_key_env
        self.api_key = os.getenv(self._api_key_env)
        if not self.api_key:
            raise ValueError(f"API key not found in env: {self._api_key_env}")

        self.model = llm_config.model
        self.base_url = llm_config.base_url
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.temperature = llm_config.temperature
        self.token_param = llm_config.token_param
        self.max_tokens = self._resolve_token_limit(llm_config)
        self.request_timeout = float(llm_config.request_timeout)

        # Create HTTP session
        self._session: aiohttp.ClientSession | None = None
        self._shutdown_called = False

        logger.info(
            "OpenAIVLM initialized: model=%s, token_param=%s, temperature=%s",
            self.model,
            self.token_param,
            self.temperature if self.temperature is not None else "default",
        )

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Lazy-create aiohttp session with timeout."""
        if self._session is None:
            if aiohttp is None:
                raise RuntimeError("aiohttp dependency is required for OpenAI VLM")
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def analyze(
        self,
        video_path: Path,
        filter_result: FilterResult,
        config: VLMConfig,
    ) -> AnalysisResult:
        """Analyze video clip using OpenAI VLM.

        Extracts frames, encodes as base64, and calls OpenAI API
        with structured output schema.
        """
        if self._shutdown_called:
            raise RuntimeError("VLM has been shut down")

        start_time = asyncio.get_running_loop().time()

        # Extract frames
        frames = await self._extract_frames_async(video_path, config.preprocessing)

        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")

        messages = self._build_messages(frames, filter_result)
        payload = self._build_payload(messages)
        headers = self._build_headers()

        data = await self._call_api(payload, headers)
        usage = data.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        self._log_usage(usage, start_time, video_path)
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        prompt_token_count = prompt_tokens if isinstance(prompt_tokens, int) else None
        completion_token_count = completion_tokens if isinstance(completion_tokens, int) else None

        # Parse response
        content = self._extract_content(data)
        analysis = self._parse_sequence_analysis(content)
        return AnalysisResult(
            risk_level=analysis.max_risk_level,
            activity_type=analysis.primary_activity,
            summary=analysis.sequence_description,
            analysis=analysis,
            prompt_tokens=prompt_token_count,
            completion_tokens=completion_token_count,
        )

    async def _extract_frames_async(
        self, video_path: Path, preprocessing: VLMPreprocessConfig
    ) -> list[tuple[str, str]]:
        return await asyncio.to_thread(
            self._extract_frames,
            video_path,
            preprocessing.max_frames,
            preprocessing.max_size,
            preprocessing.quality,
        )

    def _build_messages(
        self,
        frames: list[tuple[str, str]],
        filter_result: FilterResult,
    ) -> list[dict[str, object]]:
        frame_count = len(frames)
        start_ts = frames[0][1]
        end_ts = frames[-1][1]
        detected = ", ".join(filter_result.detected_classes) or "none"
        user_content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": (
                    f"Analyze these {frame_count} frames from security camera footage. "
                    f"Detected objects: {detected}."
                ),
            },
            {
                "type": "text",
                "text": (
                    "TIMESTAMP CONSTRAINT: This video spans from "
                    f"{start_ts} to {end_ts}. You MUST use ONLY these exact timestamps "
                    "shown in frame labels below. Do not invent timestamps outside this range."
                ),
            },
        ]
        for idx, (frame_b64, timestamp) in enumerate(frames, start=1):
            user_content.append(
                {
                    "type": "text",
                    "text": f"Frame at {timestamp} ({idx} of {frame_count}):",
                }
            )
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_b64}",
                        "detail": "high",
                    },
                }
            )

        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": user_content,
            },
        ]

    def _build_payload(self, messages: list[dict[str, object]]) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "response_format": _create_json_schema_format(SequenceAnalysis, "sequence_analysis"),
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        payload[self.token_param] = self.max_tokens
        return payload

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _call_api(
        self, payload: dict[str, object], headers: dict[str, str]
    ) -> dict[str, object]:
        session = await self._ensure_session()
        url = f"{self.base_url}/chat/completions"

        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.debug("OpenAI API error details: %s", error_text)
                raise RuntimeError(f"OpenAI API error: HTTP {resp.status}")

            data = await resp.json()
            if not isinstance(data, dict):
                raise TypeError("OpenAI API response is not a JSON object")
            return data

    def _log_usage(self, usage: dict[str, object], start_time: float, video_path: Path) -> None:
        elapsed_s = asyncio.get_running_loop().time() - start_time
        logger.info(
            "VLM token usage",
            extra={
                "event_type": "vlm_usage",
                "provider": "openai",
                "model": self.model,
                "token_param": self.token_param,
                "clip_id": video_path.stem,
                "temperature": self.temperature,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "elapsed_s": round(elapsed_s, 3),
            },
        )

    def _extract_content(self, data: dict[str, object]) -> object:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise TypeError("OpenAI API response missing choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise TypeError("OpenAI API response choice is not an object")
        message = first.get("message")
        if not isinstance(message, dict):
            raise TypeError("OpenAI API response message is not an object")
        return message.get("content")

    def _parse_sequence_analysis(self, content: object) -> SequenceAnalysis:
        try:
            response_dict = content
            if isinstance(content, str):
                response_dict = json.loads(content)
            if not isinstance(response_dict, dict):
                raise TypeError(f"Expected JSON object, got {type(response_dict).__name__}")
            return SequenceAnalysis.model_validate(response_dict)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"VLM response is not valid JSON: {e.msg}. Raw response: {content}",
                e.doc,
                e.pos,
            ) from e
        except ValueError as e:
            raise ValueError(
                f"VLM response does not match SequenceAnalysis schema: {e}. Raw response: {content}"
            ) from e

    def _resolve_token_limit(self, llm: OpenAIConfig) -> int:
        if self.token_param == "max_completion_tokens":
            value = llm.max_completion_tokens or llm.max_tokens or 1000
        else:
            value = llm.max_tokens or llm.max_completion_tokens or 1000
        return int(value)

    def _extract_frames(
        self,
        video_path: Path,
        max_frames: int,
        max_size: int,
        quality: int,
    ) -> list[tuple[str, str]]:
        """Extract and encode frames from video.

        Returns list of (base64 JPEG, timestamp) tuples.
        """
        if cv2 is None or Image is None:
            raise RuntimeError("OpenAI VLM dependencies are not available")

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return []

        # Calculate frame indices to sample
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / max_frames
            frame_indices = [int(i * step) for i in range(max_frames)]

        frames_b64: list[tuple[str, str]] = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = self._format_timestamp(timestamp_ms)

            # Convert to PIL Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            # Resize if needed
            if max(pil_img.size) > max_size:
                pil_img = self._resize_image(pil_img, max_size)

            # Encode as JPEG
            import io

            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=quality)
            frame_bytes = buffer.getvalue()

            # Base64 encode
            frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")
            frames_b64.append((frame_b64, timestamp))

        cap.release()
        return frames_b64

    def _format_timestamp(self, timestamp_ms: float) -> str:
        total_seconds = max(0.0, timestamp_ms / 1000.0)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"

    def _resize_image(self, img: Image.Image, max_size: int) -> Image.Image:
        """Resize image maintaining aspect ratio."""
        width, height = img.size

        if width <= max_size and height <= max_size:
            return img

        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    async def shutdown(self, timeout: float | None = None) -> None:
        """Cleanup resources - close HTTP session."""
        _ = timeout
        if self._shutdown_called:
            return

        self._shutdown_called = True
        logger.info("Shutting down OpenAIVLM...")

        if self._session:
            await self._session.close()

        logger.info("OpenAIVLM shutdown complete")

    async def ping(self) -> bool:
        """Health check - verify API is reachable.

        Note: This checks if session is alive and not shut down.
        A full API connectivity check would require an API call.
        """
        if self._shutdown_called:
            return False
        # Session being None is fine - it's lazy-created
        # If session exists and is closed, that's a problem
        if self._session is not None and self._session.closed:
            return False
        return True
