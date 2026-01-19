"""Tests for YOLO filter worker logic."""

from __future__ import annotations

import pytest

from homesec.models.filter import FilterResult
from homesec.plugins.filters import yolo as yolo_module


class _FakeXYXY:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


class _FakeBox:
    def __init__(self, cls_id: int, conf: float, xyxy: list[float]) -> None:
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [_FakeXYXY(xyxy)]


class _FakeResult:
    def __init__(self, boxes: list[_FakeBox]) -> None:
        self.boxes = boxes


class _FakeModel:
    def __init__(self, boxes: list[_FakeBox]) -> None:
        self._boxes = boxes

    def __call__(self, _frame: object, **_kwargs: object) -> list[_FakeResult]:
        return [_FakeResult(self._boxes)]


class _FakeCapture:
    def __init__(self, _path: str, total_frames: int = 3, height: int = 100) -> None:
        self._total_frames = total_frames
        self._height = height
        self._pos = 0

    def get(self, prop: int) -> int:
        if prop == yolo_module.cv2.CAP_PROP_FRAME_COUNT:
            return self._total_frames
        if prop == yolo_module.cv2.CAP_PROP_FRAME_HEIGHT:
            return self._height
        return 0

    def set(self, prop: int, value: int) -> None:
        if prop == yolo_module.cv2.CAP_PROP_POS_FRAMES:
            self._pos = value

    def read(self) -> tuple[bool, object]:
        if self._pos >= self._total_frames:
            return False, object()
        return True, object()

    def release(self) -> None:
        return None


def _patch_env(
    monkeypatch: pytest.MonkeyPatch,
    model: _FakeModel,
    total_frames: int = 3,
    height: int = 100,
) -> None:
    monkeypatch.setattr(yolo_module, "_get_model", lambda *_args: model)
    monkeypatch.setattr(
        yolo_module.cv2,
        "VideoCapture",
        lambda _path: _FakeCapture(_path, total_frames=total_frames, height=height),
    )
    monkeypatch.setattr(yolo_module.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(yolo_module.torch.cuda, "is_available", lambda: False)


def test_detect_worker_returns_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detect worker should return a FilterResult when detection matches."""
    # Given a model that returns a person detection
    model = _FakeModel([_FakeBox(cls_id=0, conf=0.9, xyxy=[0, 0, 10, 60])])
    _patch_env(monkeypatch, model)

    # When running the worker
    result = yolo_module._detect_worker(
        video_path="video.mp4",
        model_path="model.pt",
        target_class_ids=[0],
        min_confidence=0.5,
        sample_fps=1,
        min_box_h_ratio=0.1,
        min_hits=1,
    )

    # Then the detection is returned
    assert isinstance(result, FilterResult)
    assert result.detected_classes == ["person"]
    assert result.confidence == 0.9
    assert result.sampled_frames == 1


def test_detect_worker_filters_small_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detect worker should ignore boxes smaller than min_box_h_ratio."""
    # Given a model with a tiny detection
    model = _FakeModel([_FakeBox(cls_id=0, conf=0.9, xyxy=[0, 0, 10, 5])])
    _patch_env(monkeypatch, model, total_frames=2, height=100)

    # When running the worker with strict box height
    result = yolo_module._detect_worker(
        video_path="video.mp4",
        model_path="model.pt",
        target_class_ids=[0],
        min_confidence=0.5,
        sample_fps=1,
        min_box_h_ratio=0.5,
        min_hits=1,
    )

    # Then no classes are detected
    assert result.detected_classes == []
    assert result.confidence == 0.0


def test_detect_worker_respects_min_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detect worker should not early-exit until min_hits is reached."""
    # Given a model that returns only one class repeatedly
    model = _FakeModel([_FakeBox(cls_id=0, conf=0.7, xyxy=[0, 0, 10, 60])])
    _patch_env(monkeypatch, model, total_frames=3, height=100)

    # When running with min_hits=2
    result = yolo_module._detect_worker(
        video_path="video.mp4",
        model_path="model.pt",
        target_class_ids=[0],
        min_confidence=0.5,
        sample_fps=1,
        min_box_h_ratio=0.1,
        min_hits=2,
    )

    # Then the worker processes all frames without early exit
    assert result.detected_classes == ["person"]
    assert result.sampled_frames == 3
