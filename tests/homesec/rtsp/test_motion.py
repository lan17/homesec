"""Replay the language-neutral corpus through the production motion detector."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
import numpy.typing as npt
import pytest
from pydantic import BaseModel, ConfigDict, Field

from homesec.sources.rtsp.motion import MotionDetector

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "native/webrtc/tests/fixtures/motion"


class FrameStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["frame"]
    input_offset: int = Field(ge=0)
    blurred_offset: int = Field(ge=0)
    threshold: float | None
    changed_pixels: int = Field(ge=0)
    changed_pct: float = Field(ge=0, le=100)
    motion: bool


class ResetStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["reset"]


class MotionCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    pixel_threshold: int = Field(ge=0)
    min_changed_pct: float = Field(ge=0)
    blur_kernel: int = Field(ge=0)
    steps: list[FrameStep | ResetStep] = Field(min_length=1)


class MotionCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    reference_opencv: str
    pixel_format: Literal["gray8"]
    data_file: Literal["frames.gray"]
    cases: list[MotionCase] = Field(min_length=1)


CORPUS = MotionCorpus.model_validate_json((FIXTURE_DIR / "corpus.json").read_text())


@pytest.fixture(scope="module")
def frame_data() -> bytes:
    return (FIXTURE_DIR / CORPUS.data_file).read_bytes()


@pytest.mark.parametrize("case", CORPUS.cases, ids=lambda case: case.name)
def test_motion_characterization(case: MotionCase, frame_data: bytes) -> None:
    # Given: Frozen synthetic gray frames and results from the production Python detector.
    detector = MotionDetector(
        pixel_threshold=case.pixel_threshold,
        min_changed_pct=case.min_changed_pct,
        blur_kernel=case.blur_kernel,
        debug=False,
    )
    frame_size = case.width * case.height

    for index, step in enumerate(case.steps):
        context = f"{case.name} step {index}"
        if isinstance(step, ResetStep):
            # When: The source resets its detector after a stream lifecycle transition.
            detector.reset()

            # Then: Observations clear immediately; the next frame establishes a new baseline.
            assert detector.last_changed_pixels == 0, context
            assert detector.last_changed_pct == 0.0, context
            continue

        pixels = frame_data[step.input_offset : step.input_offset + frame_size]
        expected_blur = frame_data[step.blurred_offset : step.blurred_offset + frame_size]
        assert len(pixels) == len(expected_blur) == frame_size, context
        frame = np.frombuffer(pixels, dtype=np.uint8).reshape(case.height, case.width)

        # When: Applying the current blur and processing the frame with its threshold override.
        blurred: npt.NDArray[np.uint8] = frame
        if case.blur_kernel > 1:
            blurred = cast(
                npt.NDArray[np.uint8],
                cv2.GaussianBlur(frame, (case.blur_kernel, case.blur_kernel), 0),
            )
        motion = detector.detect(frame, threshold=step.threshold)

        # Then: Every blurred byte and public motion observation matches the frozen corpus.
        assert blurred.tobytes(order="C") == expected_blur, context
        assert detector.last_changed_pixels == step.changed_pixels, context
        assert detector.last_changed_pct == pytest.approx(step.changed_pct, rel=0, abs=1e-12), (
            context
        )
        assert motion is step.motion, context
