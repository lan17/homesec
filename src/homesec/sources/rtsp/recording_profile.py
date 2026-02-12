from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MP4_COPY_AUDIO_CODECS: frozenset[str] = frozenset(
    {
        "aac",
        "ac3",
        "alac",
        "eac3",
        "mp2",
        "mp3",
    }
)

MP4_PASSTHROUGH_TIMING_ARGS: list[str] = ["-vsync", "0"]


class MotionProfile(BaseModel):
    """Locked ffmpeg input profile for motion detection stream."""

    model_config = {"extra": "forbid"}

    input_url: str
    ffmpeg_input_args: list[str] = Field(default_factory=list)


class RecordingProfile(BaseModel):
    """Locked ffmpeg output profile for recording stream."""

    model_config = {"extra": "forbid"}

    input_url: str
    ffmpeg_input_args: list[str] = Field(default_factory=list)
    container: Literal["mp4"] = "mp4"
    video_mode: Literal["copy"] = "copy"
    audio_mode: Literal["copy", "aac", "none"]
    ffmpeg_output_args: list[str]
    output_extension: Literal["mp4"] = "mp4"

    def profile_id(self) -> str:
        base = f"{self.container}:v={self.video_mode}:a={self.audio_mode}"
        if self.uses_wallclock_timestamps():
            return f"{base}:ts=wallclock"
        return base

    def uses_wallclock_timestamps(self) -> bool:
        args = self.ffmpeg_input_args
        for idx, arg in enumerate(args):
            if arg != "-use_wallclock_as_timestamps":
                continue
            if idx + 1 < len(args):
                return args[idx + 1] == "1"
            return True
        return False


def is_mp4_audio_copy_compatible(audio_codec: str | None) -> bool:
    if audio_codec is None:
        return False
    return audio_codec.strip().lower() in MP4_COPY_AUDIO_CODECS


def build_recording_profile_candidates(
    *,
    input_url: str,
    audio_codec: str | None,
) -> list[RecordingProfile]:
    """Build startup-only negotiation candidates in deterministic order."""

    candidates: list[RecordingProfile] = []

    if is_mp4_audio_copy_compatible(audio_codec):
        candidates.append(
            RecordingProfile(
                input_url=input_url,
                audio_mode="copy",
                ffmpeg_output_args=[
                    "-c:v",
                    "copy",
                    "-c:a",
                    "copy",
                    *MP4_PASSTHROUGH_TIMING_ARGS,
                    "-f",
                    "mp4",
                ],
            )
        )

    if audio_codec is not None:
        candidates.append(
            RecordingProfile(
                input_url=input_url,
                audio_mode="aac",
                ffmpeg_output_args=[
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    *MP4_PASSTHROUGH_TIMING_ARGS,
                    "-f",
                    "mp4",
                ],
            )
        )

    candidates.append(
        RecordingProfile(
            input_url=input_url,
            audio_mode="none",
            ffmpeg_output_args=[
                "-c:v",
                "copy",
                "-an",
                *MP4_PASSTHROUGH_TIMING_ARGS,
                "-f",
                "mp4",
            ],
        )
    )

    return candidates


def build_default_recording_profile(input_url: str) -> RecordingProfile:
    """Default profile for compatibility before startup preflight runs."""

    return RecordingProfile(
        input_url=input_url,
        audio_mode="copy",
        ffmpeg_output_args=[
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            *MP4_PASSTHROUGH_TIMING_ARGS,
            "-f",
            "mp4",
        ],
    )
