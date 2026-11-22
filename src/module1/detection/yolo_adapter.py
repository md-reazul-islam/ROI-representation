from __future__ import annotations
import numpy as np
from .base import BaseDetector, Detection
from .yolo_detector import YOLODetector


class YOLODetectorAdapter(BaseDetector):
    """Wraps the existing YOLODetector so it conforms to the shared Detection
    interface, without modifying yolo_detector.py."""

    def __init__(self, *args, **kwargs):
        self._detector = YOLODetector(*args, **kwargs)

    @property
    def inner(self) -> YOLODetector:
        return self._detector

    def load(self): self._detector.load()

    @property
    def model(self): return self._detector.model

    def detect(self, frame: np.ndarray, frame_index: int = 0, timestamp: float = 0.0) -> list[Detection]:
        result=self._detector.predict(frame)
        candidates=self._detector.extract_candidates(result)
        return [Detection(bbox=(c["x1"],c["y1"],c["x2"],c["y2"]),confidence=c["confidence"],
                           class_id=c["class_id"],class_name=c["class_name"],
                           frame_index=frame_index,timestamp=timestamp) for c in candidates]
