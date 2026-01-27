from __future__ import annotations

import logging
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class MotionDetector:
    def __init__(
        self,
        *,
        pixel_threshold: int,
        min_changed_pct: float,
        blur_kernel: int,
        debug: bool,
    ) -> None:
        self._pixel_threshold = int(pixel_threshold)
        self._min_changed_pct = float(min_changed_pct)
        self._blur_kernel = int(blur_kernel)
        self._debug = bool(debug)

        self._prev_frame: npt.NDArray[np.uint8] | None = None
        self._last_changed_pct = 0.0
        self._last_changed_pixels = 0
        self._debug_frame_count = 0

    @property
    def last_changed_pct(self) -> float:
        return self._last_changed_pct

    @property
    def last_changed_pixels(self) -> int:
        return self._last_changed_pixels

    def reset(self) -> None:
        self._prev_frame = None
        self._last_changed_pct = 0.0
        self._last_changed_pixels = 0
        self._debug_frame_count = 0

    def detect(self, frame: npt.NDArray[np.uint8], *, threshold: float | None = None) -> bool:
        if frame.ndim == 3:
            gray = cast(npt.NDArray[np.uint8], cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            gray = frame

        if self._blur_kernel > 1:
            gray = cast(
                npt.NDArray[np.uint8],
                cv2.GaussianBlur(gray, (self._blur_kernel, self._blur_kernel), 0),
            )

        if self._prev_frame is None:
            self._prev_frame = gray
            self._last_changed_pct = 0.0
            self._last_changed_pixels = 0
            return False

        diff = cv2.absdiff(self._prev_frame, gray)
        _, mask = cv2.threshold(diff, self._pixel_threshold, 255, cv2.THRESH_BINARY)
        changed_pixels = int(cv2.countNonZero(mask))

        total_pixels = int(gray.shape[0]) * int(gray.shape[1])
        changed_pct = (changed_pixels / total_pixels * 100.0) if total_pixels else 0.0

        self._prev_frame = gray
        self._last_changed_pct = changed_pct
        self._last_changed_pixels = changed_pixels

        if threshold is None:
            threshold = self._min_changed_pct
        if threshold < 0:
            threshold = 0.0

        motion = changed_pct >= float(threshold)

        if self._debug:
            self._debug_frame_count += 1
            if self._debug_frame_count % 100 == 0:
                logger.debug(
                    "Motion check: changed_pct=%.3f%% changed_px=%s pixel_threshold=%s min_changed_pct=%.3f%% blur=%s",
                    changed_pct,
                    changed_pixels,
                    self._pixel_threshold,
                    self._min_changed_pct,
                    self._blur_kernel,
                )

        return motion
