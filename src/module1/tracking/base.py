from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from module1.detection.base import Detection


@dataclass(frozen=True)
class TrackingFrame:
    """Original image and timeline metadata, including frames with no detections."""
    image: np.ndarray
    frame_index: int
    timestamp: float


@dataclass(frozen=True)
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    frame_index: int
    timestamp: float
    velocity: tuple[float, float]
    age: int
    number_of_hits: int
    number_of_misses: int
    track_state: str
    consecutive_misses: int
    is_primary: bool
    observed: bool
    reliability: float
    class_id: int


class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections: list[Detection], frame: TrackingFrame) -> list[Track]:
        """Advance once per processed frame. Call even when detections is empty."""

    @abstractmethod
    def reset(self):
        """Clear all temporal state before starting a different video."""
