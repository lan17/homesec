"""YOLO object detection filter plugin."""

from __future__ import annotations

import asyncio
import logging
import shutil
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

cv2: Any
torch: Any
YOLO_CLASS: Any

try:
    import cv2 as _cv2
    import torch as _torch
    from ultralytics import YOLO as _YOLO  # type: ignore[attr-defined]
except Exception:
    cv2 = None
    torch = None
    YOLO_CLASS = None
else:
    cv2 = _cv2
    torch = _torch
    YOLO_CLASS = _YOLO

from homesec.interfaces import ObjectFilter
from homesec.models.filter import FilterOverrides, FilterResult, YoloFilterSettings
from homesec.plugins.registry import PluginType, plugin

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

_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_YOLO_CACHE_DIR = Path.cwd() / "yolo_cache"


def _ensure_yolo_dependencies() -> None:
    """Fail fast with a clear error if YOLO dependencies are missing."""
    if cv2 is None or torch is None or YOLO_CLASS is None:
        raise RuntimeError(
            "Missing dependency for YOLO filter. "
            "Install with: uv pip install ultralytics opencv-python"
        )


def _resolve_requested_path(model_path: str) -> Path:
    requested = Path(model_path)
    if not requested.is_absolute() and requested.parent == Path("."):
        return _YOLO_CACHE_DIR / requested.name
    return requested


def _check_file_safe(key: str) -> str | None:
    """Call ultralytics check_file with type-safe result handling.

    Returns the resolved path string if found, None otherwise.
    """
    try:
        from ultralytics.utils.checks import check_file

        result = check_file(key)  # type: ignore[no-untyped-call]
        # Validate result is string (ultralytics may return various types)
        if isinstance(result, str):
            return result
        return None
    except Exception:
        return None


def _resolve_weights_path(model_path: str) -> Path:
    requested_path = _resolve_requested_path(model_path)
    if requested_path.exists():
        return requested_path

    resolved: Path | None = None
    try:
        for key in (str(requested_path), requested_path.name):
            candidate = _check_file_safe(key)
            if candidate and Path(candidate).exists():
                resolved = Path(candidate)
                break

        if resolved is None and requested_path.suffix.lower() == ".pt":
            try:
                if YOLO_CLASS is None:
                    raise RuntimeError("YOLO dependencies are not available")
                _ = YOLO_CLASS(requested_path.name)
                candidate = _check_file_safe(requested_path.name)
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


def _get_model(model_path: str, device: str) -> Any:
    """Get or create a cached YOLO model instance.

    Thread-safe: uses a lock to prevent duplicate model loading when
    multiple threads access the cache simultaneously.
    """
    key = (model_path, device)
    # Fast path: check without lock (safe for reads)
    model = _MODEL_CACHE.get(key)
    if model is not None:
        return model

    # Slow path: acquire lock for potential write
    with _MODEL_CACHE_LOCK:
        # Double-check after acquiring lock
        model = _MODEL_CACHE.get(key)
        if model is None:
            if YOLO_CLASS is None:
                raise RuntimeError("YOLO dependencies are not available")
            model = YOLO_CLASS(model_path).to(device)
            _MODEL_CACHE[key] = model
        return model


@plugin(plugin_type=PluginType.FILTER, name="yolo")
class YOLOFilter(ObjectFilter):
    """YOLO-based object detection filter.

    Uses ProcessPoolExecutor internally for CPU/GPU-bound inference.
    Supports frame sampling and early exit on detection.
    Bare model filenames resolve under ./yolo_cache and auto-download if missing.
    """

    config_cls = YoloFilterSettings

    @classmethod
    def create(cls, config: YoloFilterSettings) -> ObjectFilter:
        return cls(config)

    def __init__(self, settings: YoloFilterSettings) -> None:
        """Initialize YOLO filter with validated settings.

        Args:
            settings: YOLO-specific configuration (model_path, classes, thresholds)
                      Also assumes settings.max_workers is populated.
        """
        _ensure_yolo_dependencies()
        self._settings = settings
        self.model_path = _resolve_weights_path(str(settings.model_path))
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self._class_id_cache: dict[tuple[str, ...], list[int]] = {}

        # Initialize executor
        self._executor = ProcessPoolExecutor(max_workers=settings.max_workers)
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

    async def ping(self) -> bool:
        """Health check - verify executor is alive and model path exists."""
        if self._shutdown_called:
            return False
        if not self.model_path.exists():
            return False
        # Executor is considered healthy if not shut down
        return True

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
    if cv2 is None or torch is None:
        raise RuntimeError("YOLO dependencies are not available")

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
