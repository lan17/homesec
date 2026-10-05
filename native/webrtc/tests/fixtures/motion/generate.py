"""Capture the current Python motion behavior using synthetic grayscale frames."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from homesec.sources.rtsp.motion import MotionDetector

GrayFrame = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class Step:
    frame: GrayFrame | None  # None means reset the detector.
    threshold: float | None = None


@dataclass(frozen=True)
class Case:
    name: str
    steps: list[Step]
    pixel_threshold: int = 45
    min_changed_pct: float = 1.0
    blur_kernel: int = 0


def patterns(height: int, width: int) -> list[GrayFrame]:
    """Use arithmetic noise so inputs do not depend on an RNG implementation."""
    y, x = np.indices((height, width))
    center = np.zeros((height, width), dtype=np.uint8)
    center[height // 2, width // 2] = 255
    corner = np.zeros_like(center)
    corner[0, 0] = 255
    edge = np.where(x < width // 2, 0, 255).astype(np.uint8)
    checker = ((x + y) % 2 * 255).astype(np.uint8)
    noise = ((x * 73 + y * 151 + x * y * 19 + 37) % 256).astype(np.uint8)
    return [np.zeros_like(center), center, corner, edge, checker, noise]


def cases() -> list[Case]:
    zero = np.zeros((2, 2), dtype=np.uint8)
    at_threshold = np.array([[45, 0], [0, 0]], dtype=np.uint8)
    over_threshold = np.array([[91, 0], [0, 0]], dtype=np.uint8)
    changed = np.array([[46, 0], [0, 0]], dtype=np.uint8)
    full = np.full_like(zero, 255)
    result = [
        Case(
            "strict_pixel_threshold_and_inclusive_percentage",
            [Step(zero), Step(at_threshold), Step(over_threshold)],
            min_changed_pct=25.0,
        ),
        Case(
            "threshold_overrides_and_zero_first_frame",
            [
                Step(zero, 0.0),
                Step(changed),
                Step(zero, 15.0),
                Step(changed, 25.0),
                Step(zero, 25.00001),
                Step(zero, 0.0),
                Step(zero, -1.0),
                Step(zero),
            ],
            min_changed_pct=30.0,
        ),
        Case("zero_configured_threshold", [Step(zero), Step(zero)], min_changed_pct=0.0),
        Case("previous_frame_baseline", [Step(zero), Step(changed), Step(changed), Step(zero)]),
        Case(
            "reset_after_motion",
            [Step(zero), Step(changed), Step(None), Step(changed), Step(zero)],
        ),
        Case("pixel_threshold_255", [Step(zero), Step(full)], pixel_threshold=255),
        Case("pixel_threshold_above_255", [Step(zero), Step(full)], pixel_threshold=256),
        Case(
            "zero_pixel_threshold",
            [Step(zero), Step(np.array([[1, 0], [0, 0]], dtype=np.uint8))],
            pixel_threshold=0,
        ),
        Case("percentage_above_100", [Step(zero), Step(full)], min_changed_pct=100.1),
    ]
    for kernel in (0, 1, 3, 5, 7, 9, 11, 31):
        result.append(
            Case(
                f"patterns_kernel_{kernel}",
                [Step(frame) for frame in patterns(7, 11)],
                pixel_threshold=5,
                min_changed_pct=2.0,
                blur_kernel=kernel,
            )
        )
    for height, width, kernel in ((1, 1, 5), (1, 9, 5), (9, 1, 5), (2, 3, 11)):
        result.append(
            Case(
                f"small_shape_{height}x{width}_kernel_{kernel}",
                [Step(frame) for frame in patterns(height, width)],
                pixel_threshold=5,
                blur_kernel=kernel,
            )
        )
    for kernel in (255, 1001):
        result.append(
            Case(
                f"large_kernel_{kernel}_on_3x5",
                [Step(frame) for frame in patterns(3, 5)],
                pixel_threshold=5,
                blur_kernel=kernel,
            )
        )

    background = patterns(240, 320)[-1]
    foreground = background.copy()
    foreground[70:150, 100:220] = 255
    result.append(
        Case(
            "production_320x240_default_settings",
            [Step(background), Step(foreground), Step(foreground), Step(background)],
            blur_kernel=5,
        )
    )
    return result


def generate() -> tuple[bytes, bytes]:
    data = bytearray()
    offsets: dict[bytes, int] = {}

    def store(frame: GrayFrame) -> int:
        pixels = frame.tobytes(order="C")
        if pixels not in offsets:
            offsets[pixels] = len(data)
            data.extend(pixels)
        return offsets[pixels]

    recorded_cases: list[dict[str, object]] = []
    for case in cases():
        first_frame = case.steps[0].frame
        assert first_frame is not None
        height, width = first_frame.shape
        detector = MotionDetector(
            pixel_threshold=case.pixel_threshold,
            min_changed_pct=case.min_changed_pct,
            blur_kernel=case.blur_kernel,
            debug=False,
        )
        steps: list[dict[str, object]] = []
        for step in case.steps:
            if step.frame is None:
                detector.reset()
                steps.append({"operation": "reset"})
                continue
            frame = step.frame
            assert frame.shape == (height, width)
            blurred = frame
            if case.blur_kernel > 1:
                blurred = cast(
                    GrayFrame, cv2.GaussianBlur(frame, (case.blur_kernel, case.blur_kernel), 0)
                )
            motion = detector.detect(frame, threshold=step.threshold)
            steps.append(
                {
                    "operation": "frame",
                    "input_offset": store(frame),
                    "blurred_offset": store(blurred),
                    "threshold": step.threshold,
                    "changed_pixels": detector.last_changed_pixels,
                    "changed_pct": detector.last_changed_pct,
                    "motion": motion,
                }
            )
        recorded_cases.append(
            {
                "name": case.name,
                "width": width,
                "height": height,
                "pixel_threshold": case.pixel_threshold,
                "min_changed_pct": case.min_changed_pct,
                "blur_kernel": case.blur_kernel,
                "steps": steps,
            }
        )
    manifest = {
        "schema_version": 1,
        "reference_opencv": cv2.__version__,
        "pixel_format": "gray8",
        "data_file": "frames.gray",
        "cases": recorded_cases,
    }
    return (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode(), bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check without rewriting the corpus")
    args = parser.parse_args()
    manifest, data = generate()
    directory = Path(__file__).resolve().parent
    for name, content in (("corpus.json", manifest), ("frames.gray", data)):
        path = directory / name
        if args.check:
            if path.read_bytes() != content:
                raise SystemExit(f"{name} differs from current Python/OpenCV behavior")
        else:
            path.write_bytes(content)
    print(f"{'Verified' if args.check else 'Wrote'} {len(cases())} cases; {len(data)} raw bytes")


if __name__ == "__main__":
    main()
