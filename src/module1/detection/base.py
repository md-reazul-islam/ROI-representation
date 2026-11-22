from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class Detection:
    """Detector-agnostic detection result shared by every detector adapter."""
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    frame_index: int
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        x1, y1, x2, y2 = self.bbox
        return {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": self.confidence, "class_id": self.class_id, "class_name": self.class_name,
            "frame_index": self.frame_index, "timestamp": self.timestamp,
        }


class BaseDetector(ABC):
    """Shared interface every infant detector adapter must conform to."""

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_index: int = 0, timestamp: float = 0.0) -> list[Detection]:
        """Run inference on a single frame and return Detection results sorted by confidence descending."""
        raise NotImplementedError
