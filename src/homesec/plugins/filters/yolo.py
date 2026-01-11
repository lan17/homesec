"""YOLOv8 object detection filter plugin."""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import cast

import cv2
import torch
from ultralytics import YOLO  # type: ignore[attr-defined]

from homesec.interfaces import ObjectFilter
from homesec.models.filter import FilterConfig, FilterOverrides, FilterResult, YoloFilterSettings

logger = logging.getLogger(__name__)

# COCO classes for humans and animals
HUMAN_ANIMAL_CLASSES = {
    0: "person",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}

_MODEL_CACHE: dict[tuple[str, str], YOLO] = {}
_YOLO_CACHE_DIR = Path.cwd() / "yolo_cache"


def _resolve_requested_path(model_path: str) -> Path:
    requested = Path(model_path)
    if not requested.is_absolute() and requested.parent == Path("."):
        return _YOLO_CACHE_DIR / requested.name
    return requested


def _resolve_weights_path(model_path: str) -> Path:
    requested_path = _resolve_requested_path(model_path)
    if requested_path.exists():
        return requested_path

    resolved: Path | None = None
    try:
        from ultralytics.utils.checks import check_file

        check_file_fn = cast(Callable[[str], str | None], check_file)

        for key in (str(requested_path), requested_path.name):
            try:
                candidate = check_file_fn(key)
            except Exception:
                continue
            if candidate and Path(candidate).exists():
                resolved = Path(candidate)
                break

        if resolved is None and requested_path.suffix.lower() == ".pt":
            try:
                _ = YOLO(requested_path.name)
                candidate = check_file_fn(requested_path.name)
                if candidate and Path(candidate).exists():
                    resolved = Path(candidate)
            except Exception:
                resolved = None
    except Exception:
        resolved = None

    if resolved is None:
        return requested_path

    if requested_path.suffix.lower() == ".pt" and resolved != requested_path:
        try:
            requested_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved, requested_path)
            return requested_path
        except Exception:
            return resolved

    return resolved


def _get_model(model_path: str, device: str) -> YOLO:
    key = (model_path, device)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = YOLO(model_path).to(device)
        _MODEL_CACHE[key] = model
    return model


class YOLOv8Filter(ObjectFilter):
    """YOLO-based object detection filter.

    Uses ProcessPoolExecutor internally for CPU/GPU-bound inference.
    Supports frame sampling and early exit on detection.
    Bare model filenames resolve under ./yolo_cache and auto-download if missing.
    """

    def __init__(self, config: FilterConfig) -> None:
        """Initialize YOLO filter with config validation.

        Required config:
            model_path: Path to .pt model file

        Optional config:
            classes: List of class names to detect (default: person)
            min_confidence: Minimum confidence threshold (default: 0.5)
            sample_fps: Frame sampling rate (default: 2)
            min_box_h_ratio: Minimum box height ratio (default: 0.1)
            min_hits: Minimum detections to confirm (default: 1)
        """
        match config.config:
            case YoloFilterSettings() as settings:
                cfg = settings
            case _:
                raise ValueError("YOLOv8Filter requires YoloFilterSettings config")

        self._settings = cfg
        self.model_path = _resolve_weights_path(str(cfg.model_path))
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self._class_id_cache: dict[tuple[str, ...], list[int]] = {}

        # Initialize executor
        self._executor = ProcessPoolExecutor(max_workers=config.max_workers)
        self._shutdown_called = False

        logger.info(
            "YOLOv8Filter initialized: model=%s, classes=%s, confidence=%.2f",
            self.model_path,
            self._settings.classes,
            self._settings.min_confidence,
        )

    async def detect(
        self,
        video_path: Path,
        overrides: FilterOverrides | None = None,
    ) -> FilterResult:
        """Detect objects in video clip.

        Runs inference in ProcessPoolExecutor to avoid blocking event loop.
        Samples frames at configured rate and exits early on first detection.
        """
        if self._shutdown_called:
            raise RuntimeError("Filter has been shut down")

        # Run blocking work in executor
        loop = asyncio.get_running_loop()
        effective = self._apply_overrides(overrides)
        target_class_ids = self._class_ids_for(effective.classes)

        result = await loop.run_in_executor(
            self._executor,
            _detect_worker,
            str(video_path),
            str(self.model_path),
            target_class_ids,
            float(effective.min_confidence),
            int(effective.sample_fps),
            float(effective.min_box_h_ratio),
            int(effective.min_hits),
        )

        return result

    async def shutdown(self, timeout: float | None = None) -> None:
        """Cleanup resources - shutdown executor."""
        _ = timeout
        if self._shutdown_called:
            return

        self._shutdown_called = True
        logger.info("Shutting down YOLOv8Filter...")
        self._executor.shutdown(wait=True, cancel_futures=False)
        logger.info("YOLOv8Filter shutdown complete")

    def _apply_overrides(self, overrides: FilterOverrides | None) -> YoloFilterSettings:
        if overrides is None:
            return self._settings
        update = overrides.model_dump(exclude_none=True)
        return self._settings.model_copy(update=update)

    def _class_ids_for(self, classes: list[str]) -> list[int]:
        key = tuple(classes)
        cached = self._class_id_cache.get(key)
        if cached is not None:
            return cached
        target_class_ids = [cid for cid, name in HUMAN_ANIMAL_CLASSES.items() if name in classes]
        if not target_class_ids:
            raise ValueError(f"No valid classes found in config: {classes}")
        self._class_id_cache[key] = target_class_ids
        return target_class_ids


def _detect_worker(
    video_path: str,
    model_path: str,
    target_class_ids: list[int],
    min_confidence: float,
    sample_fps: int,
    min_box_h_ratio: float,
    min_hits: int,
) -> FilterResult:
    """Worker function for video analysis (must be at module level for pickling).

    This runs in a separate process, so it needs to load the model fresh.
    """
    # Determine device
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Load model (cached per process)
    model = _get_model(model_path, device)

    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    min_box_h = frame_h * min_box_h_ratio

    detected_classes: list[str] = []
    max_confidence = 0.0
    sampled_frames = 0

    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        sampled_frames += 1

        # Run inference
        results = model(frame, verbose=False, conf=min_confidence, classes=target_class_ids)

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                class_name = HUMAN_ANIMAL_CLASSES.get(cls)
                if not class_name:
                    continue

                # Check box height
                xyxy = box.xyxy[0].tolist()
                box_h = xyxy[3] - xyxy[1]
                if box_h < min_box_h:
                    continue

                # Track detection
                confidence = float(box.conf[0])
                if class_name not in detected_classes:
                    detected_classes.append(class_name)
                max_confidence = max(max_confidence, confidence)

                # Early exit if we have enough hits
                if len(detected_classes) >= min_hits:
                    cap.release()
                    return FilterResult(
                        detected_classes=detected_classes,
                        confidence=max_confidence,
                        model=Path(model_path).name,
                        sampled_frames=sampled_frames,
                    )

        frame_idx += sample_fps

    cap.release()

    return FilterResult(
        detected_classes=detected_classes,
        confidence=max_confidence if detected_classes else 0.0,
        model=Path(model_path).name,
        sampled_frames=sampled_frames,
    )


# Plugin registration
from homesec.plugins.filters import FilterPlugin, filter_plugin


@filter_plugin(name="yolo")
def yolo_filter_plugin() -> FilterPlugin:
    """YOLO filter plugin factory.

    Returns:
        FilterPlugin for YOLOv8 object detection
    """

    def factory(cfg: FilterConfig) -> ObjectFilter:
        return YOLOv8Filter(cfg)

    return FilterPlugin(
        name="yolo",
        config_model=YoloFilterSettings,
        factory=factory,
    )
